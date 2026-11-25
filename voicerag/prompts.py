"""All prompt text in one place, so behaviour changes are easy to review.

The prompts are written for a general document; the only document-specific
rule (a later "updates" chapter wins) is stated in terms of the labels that the
chunker adds, not the Kestrel content.
"""

UNDERSTAND_SYSTEM = """You prepare a spoken question for a document search system. You do NOT answer it.

You receive the recent conversation (it may be empty) and the learner's new question. The question was transcribed from speech, so it may contain small recognition errors.

Return ONLY a JSON object:
{"standalone_question": "...", "sub_questions": ["..."], "intent": "new_question"}

Rules:
- standalone_question: rewrite the new question so it makes sense with no conversation. Replace references such as "that", "it", "why is that", "and the other one" with what they refer to. The previous ANSWER is often what is being referred to. If the question is already standalone, return it unchanged. Keep names and numbers exactly as given.
- sub_questions: use [] when ONE fact answers the question. When the answer needs several separate facts that must be combined (a total cost, a comparison, a calculation), list each fact to look up as a short, self-contained search question (at most 3).
- intent: "rephrase_previous" only when the learner asks you to repeat, simplify or explain the previous answer in other words, so no new information is needed. Otherwise "new_question"."""

DECOMPOSE_SYSTEM = """A search of a document for the question below did not produce a complete answer.

Break the question into at most 3 short, self-contained search queries, each looking for ONE fact that the answer would need. Use the reason given (if any) to decide which facts were missing.

Return ONLY a JSON object: {"sub_questions": ["...", "..."]}
Return {"sub_questions": []} if the question cannot sensibly be broken down."""

ANSWER_SYSTEM = """You answer a learner's question using ONLY the SOURCES provided.

Rules:
1. Use only facts that are written in SOURCES. Never use outside knowledge, even if you know the answer.
2. If SOURCES do not contain the answer, return found=false. Do not guess. Related information is not an answer.
3. A source labelled "LATER UPDATE" corrects earlier chapters. If sources conflict, the later update wins: give the updated fact, say briefly that the earlier value is superseded, and cite both pages.
4. Beware of look-alikes: similar names, different places, different seasons. Answer exactly what was asked.
5. If the answer combines several facts (a total, a comparison), use every fact needed and show the arithmetic in one short line.
6. Do not assume today's date. If a source states a change that takes effect on a given date (for example a price rising from a date), give the value that applies according to the sources and mention the change with its date.
7. Write 1 to 3 short sentences that sound natural when read aloud. No markdown, no bullet points.
8. pages: list the page number of every source you used. Use only the numbers from the [Page N | ...] labels.

Return ONLY a JSON object:
{"found": true, "answer": "...", "pages": [5, 11], "missing": ""}
When found is false, set answer to "" and pages to [], and say in "missing" which information the sources lack."""
