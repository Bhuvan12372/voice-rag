"""The question-answering pipeline (everything after speech-to-text).

    question
      -> [understand]  follow-up rewrite + sub-questions + intent   (1 LLM call)
      -> [retrieve]    hybrid dense+BM25, RRF            (per query, local)
      -> [rerank]      cross-encoder, keep top chunks     (per query, local)
      -> [override]    force relevant "later update" chunks into the context
      -> [guard]       skip the LLM if nothing relevant was found
      -> [answer]      grounded JSON answer               (1 LLM call)
      -> [verify]      citations must come from the shown chunks
      -> [retry]       if "not found": decompose once, retrieve again, answer again

Bounded on purpose: at most 2 LLM calls normally and 4 on the retry path, so
cost and latency are predictable and easy to explain.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .config import Settings
from .cost_tracker import QuestionMetrics, SessionLedger
from .generator import AnswerGenerator
from .guardrails import GuardrailResult, apply_guardrails, below_relevance_floor
from .llm_client import JSONLLM, LLMOutputError
from .models import NOT_FOUND_MESSAGE, Chunk, FinalAnswer, ScoredChunk, Turn, Understanding
from .query_understanding import QueryUnderstanding

Pool = dict[str, ScoredChunk]       # chunk_id -> scored chunk
Query = tuple[str, int]             # (query text, how many reranked chunks to keep)


@dataclass
class Session:
    """State of one conversation: recent turns, last sources, cost ledger."""

    history: deque = field(default_factory=deque)           # deque[Turn], bounded
    last_context: list[ScoredChunk] = field(default_factory=list)
    ledger: SessionLedger = field(default_factory=SessionLedger)


def merge_pools(a: Pool, b: Pool) -> Pool:
    """Union of two pools keeping the higher score; 'override' origin wins."""
    merged = dict(a)
    for cid, sc in b.items():
        if cid not in merged:
            merged[cid] = sc
            continue
        old = merged[cid]
        origin = "override" if "override" in (old.origin, sc.origin) else old.origin
        merged[cid] = ScoredChunk(old.chunk, max(old.score, sc.score), origin)
    return merged


class VoiceRAGPipeline:
    """Orchestrates retrieval, generation and guardrails for one question at a time."""

    def __init__(self, settings: Settings, retriever, reranker, llm: JSONLLM) -> None:
        self.cfg = settings
        self.retriever = retriever
        self.reranker = reranker
        self.understanding = QueryUnderstanding(llm, settings)
        self.generator = AnswerGenerator(llm, settings)

    # ------------------------------------------------------------------ public
    def new_session(self) -> Session:
        """Fresh conversation state (history is a bounded window)."""
        return Session(history=deque(maxlen=self.cfg.history_turns))

    def ask(self, question: str, session: Session) -> FinalAnswer:
        """Answer one (already transcribed) question and update the session."""
        metrics = QuestionMetrics()
        path: list[str] = []
        warnings: list[str] = []
        question = question.strip()

        # 1. understand ------------------------------------------------------
        with metrics.stage("understand"):
            u = self.understanding.understand(question, list(session.history), metrics)
        path.append(self._describe_understanding(u))

        # 2. retrieve (or reuse the previous sources for "say that again") -----
        pool: Pool = {}
        best: float | None = None
        reuse = u.intent == "rephrase_previous" and bool(session.last_context) and bool(session.history)
        if reuse:
            context = list(session.last_context)
            path.append("rephrase request: reused the previous sources, no retrieval")
        else:
            pool, best = self._gather(self._queries_for(u), metrics)
            context = self._compose(pool)
            path.append(f"retrieved {len(context)} chunks (best rerank score {best:.3f})")

        # 3. answer, with the cheap short-circuit ------------------------------
        if not reuse and (not context or below_relevance_floor(best or 0.0, self.cfg)):
            result = GuardrailResult(False, "", [])
            path.append("short-circuit: nothing relevant retrieved, answer LLM call skipped")
        else:
            previous = session.history[-1].answer if reuse else ""
            result = self._generate(u.standalone_question, context, metrics, warnings, "answer", previous)

            # 4. one retry with decomposition when the answer was "not found" --
            if not result.found and not reuse and self.cfg.retry_on_not_found:
                result, context = self._retry(u, pool, context, result, metrics, path, warnings)

        # 5. package -----------------------------------------------------------
        warnings.extend(result.warnings)
        if not result.found and result.missing:
            path.append(f"model reported missing information: {result.missing}")
        text = result.answer if result.found else NOT_FOUND_MESSAGE
        session.history.append(Turn(question=question, answer=text))
        if result.found:
            session.last_context = context
        metrics.finish()
        session.ledger.add(metrics)
        return FinalAnswer(
            question=question,
            found=result.found,
            answer=text,
            pages=result.pages,
            sources=context,
            understanding=u,
            path=path,
            warnings=warnings,
            metrics=metrics,
            best_score=best,
        )

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _describe_understanding(u: Understanding) -> str:
        text = f"understood as: {u.standalone_question!r}"
        if u.sub_questions:
            text += f" with sub-questions {u.sub_questions}"
        if u.intent != "new_question":
            text += f" [{u.intent}]"
        return text

    def _queries_for(self, u: Understanding) -> list[Query]:
        """Main question gets the full top-N; each sub-question a smaller top-N."""
        queries: list[Query] = [(u.standalone_question, self.cfg.rerank_top_n)]
        queries += [(s, self.cfg.sub_query_top_n) for s in u.sub_questions]
        return queries

    def _gather(self, queries: list[Query], metrics: QuestionMetrics) -> tuple[Pool, float]:
        """Hybrid search + rerank for each query, then the override pass.

        Returns (pool of scored chunks, best reranker score among retrieved chunks).
        """
        pool: Pool = {}
        best = 0.0
        for text, keep in queries:
            with metrics.stage("retrieve"):
                candidates = self.retriever.search(text)
            with metrics.stage("rerank"):
                ranked = self.reranker.rerank(text, candidates)
            if ranked:
                best = max(best, ranked[0].score)
            for sc in ranked[:keep]:
                if sc.score < self.cfg.context_min_rerank_score:
                    continue
                if sc.chunk.chunk_id not in pool or sc.score > pool[sc.chunk.chunk_id].score:
                    pool[sc.chunk.chunk_id] = sc
        with metrics.stage("override"):
            pool = merge_pools(pool, self._override_pass(queries, pool))
        return pool, best

    def _override_pass(self, queries: list[Query], pool: Pool) -> Pool:
        """Make sure a relevant "later update" chunk can reach the LLM.

        An update chunk and the older chunk it corrects are about the same topic,
        so ordinary retrieval may surface only one of them. Candidates:
          * update chunks in chapters that retrieved chunks point to ("see Chapter 9")
          * the nearest update chunks per query (metadata-filtered vector search)
        The reranker then decides which are relevant enough (`update_min_score`).
        """
        chapters = self.retriever.update_chapters()
        if not chapters:
            return {}
        referenced = {ch for sc in pool.values() for ch in sc.chunk.see_also} & chapters
        candidates: dict[str, Chunk] = {c.chunk_id: c for c in self.retriever.update_chunks_in(referenced)}
        for text, _ in queries:
            for chunk in self.retriever.search_updates(text, self.cfg.update_candidates):
                candidates.setdefault(chunk.chunk_id, chunk)
        for cid in pool:                      # already in the context: nothing to force
            candidates.pop(cid, None)
        if not candidates:
            return {}

        best: Pool = {}
        for text, _ in queries:
            for sc in self.reranker.rerank(text, list(candidates.values())):
                cid = sc.chunk.chunk_id
                if cid not in best or sc.score > best[cid].score:
                    best[cid] = ScoredChunk(sc.chunk, sc.score, "override")
        ranked = sorted(best.values(), key=lambda sc: sc.score, reverse=True)
        chosen = [sc for sc in ranked if sc.score >= self.cfg.update_min_score][: self.cfg.override_n]
        return {sc.chunk.chunk_id: sc for sc in chosen}

    def _compose(self, pool: Pool) -> list[ScoredChunk]:
        """Pick the final context: all overrides + best retrieved chunks up to the cap.

        Returned in document order, so an update appears after the chapter it corrects.
        """
        overrides = [sc for sc in pool.values() if sc.origin == "override"]
        retrieved = sorted(
            (sc for sc in pool.values() if sc.origin != "override"), key=lambda sc: sc.score, reverse=True
        )[: max(1, self.cfg.max_context_chunks - len(overrides))]
        return sorted(retrieved + overrides, key=lambda sc: (sc.chunk.page, sc.chunk.chunk_id))

    def _generate(
        self,
        question: str,
        context: list[ScoredChunk],
        metrics: QuestionMetrics,
        warnings: list[str],
        stage: str,
        previous_answer: str = "",
    ) -> GuardrailResult:
        """Call the answer LLM and apply the guardrails; bad JSON becomes 'not found'."""
        with metrics.stage(stage):
            try:
                raw = self.generator.generate(
                    question, context, metrics, stage=stage, previous_answer=previous_answer
                )
            except LLMOutputError as exc:
                warnings.append(f"Answer model returned unusable output: {exc}")
                return GuardrailResult(False, "", [], missing="unusable model output")
        return apply_guardrails(raw, context)

    def _retry(
        self,
        u: Understanding,
        pool: Pool,
        context: list[ScoredChunk],
        first: GuardrailResult,
        metrics: QuestionMetrics,
        path: list[str],
        warnings: list[str],
    ) -> tuple[GuardrailResult, list[ScoredChunk]]:
        """One decomposition retry. Returns the (possibly improved) result and context."""
        with metrics.stage("retry_decompose"):
            subs = self.understanding.decompose(u.standalone_question, first.missing, metrics)
        known = {u.standalone_question.lower(), *(s.lower() for s in u.sub_questions)}
        new = [s for s in subs if s.lower() not in known]
        if not new:
            path.append("retry skipped: decomposition produced no new sub-questions")
            return first, context

        path.append(f"retry: decomposed into {new}")
        extra_pool, _ = self._gather([(s, self.cfg.sub_query_top_n) for s in new], metrics)
        merged = self._compose(merge_pools(pool, extra_pool))
        second = self._generate(u.standalone_question, merged, metrics, warnings, "retry_answer")
        if second.found:
            path.append("retry succeeded: found after decomposition")
            return second, merged
        path.append("retry failed: still not found")
        return first, context
