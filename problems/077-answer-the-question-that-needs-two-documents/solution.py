
from textwrap import dedent

MAX_SEARCHES = 2


def _format_notes(notes: dict[str, dict]) -> str:
    """Format evidence with source identifiers."""
    return "\n".join(
        f"[{note['id']}] {note['text']}"
        for note in notes.values()
    )


def _next_query(
    question: str,
    notes: dict[str, dict],
    queries: list[str],
) -> str:
    """Generate a targeted query for missing evidence."""
    prompt = dedent("""
        You are planning the next retrieval in a multi-hop RAG system.

        First, identify the entity needed to answer the question
        using the retrieved evidence.

        If the evidence fully answers the question, return DONE.
        Otherwise, generate a short, specific search query.

        Rules:
        - Use the exact entity discovered in the evidence.
        - Search for the missing attribute or relationship.
        - Do not repeat previous queries.
        - Return only the query or DONE.

        <previous_queries>
        {history}
        </previous_queries>

        <evidence>
        {evidence}
        </evidence>

        <question>
        {question}
        </question>

        Next search query:
    """).strip().format(
        history="\n".join(queries),
        evidence=_format_notes(notes),
        question=question,
    )

    return (llm.ask(prompt, max_tokens=40) or "").strip()


def _generate_answer(question: str, notes: dict[str, dict]) -> str:
    """Answer using only evidence that matches the question."""
    prompt = dedent("""
        Answer the question using the retrieved evidence.

        Follow the relationships between entities carefully.
        Ignore evidence about unrelated entities.
        Preserve exact names, numbers, and facts.
        If the answer is unsupported, say "I don't know".
        Return one short sentence.

        <evidence>
        {evidence}
        </evidence>

        <question>
        {question}
        </question>

        Answer:
    """).strip().format(
        evidence=_format_notes(notes),
        question=question,
    )

    return (llm.ask(prompt, max_tokens=80) or "").strip()


def answer(question: str) -> dict:
    """Answer a question through bounded multi-hop retrieval."""
    notes = {}
    queries = []
    query = question

    while len(queries) < MAX_SEARCHES:
        queries.append(query)

        for note in retrieve(query):
            notes.setdefault(note["id"], note)

        if len(queries) >= MAX_SEARCHES:
            break

        query = _next_query(question, notes, queries)

        if query.upper() == "DONE":
            break

    response = _generate_answer(question, notes)

    return {
        "answer": response,
        "sources": list(notes),
    }
