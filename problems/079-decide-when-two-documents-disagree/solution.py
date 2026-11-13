from functools import lru_cache

@lru_cache(maxsize=256)
def ask_cached(prompt: str, max_tokens: int = 60) -> str:
    return llm.ask(prompt, max_tokens=max_tokens)


def disagree(question, a, b):
    """Whether two notes give different answers to the question."""
    return a["text"] != b["text"]


def pick(notes):
    """Choose the most authoritative and recent note."""
    return max(
        notes,
        key=lambda n: (n["source"] == "policy", n["date"]),
    )


def disagree(question: str, a: dict, b: dict) -> bool:
    """Check whether two notes give conflicting answers."""
    response = ask_cached(
        f"Question: {question}\n"
        f"Note A: {a['text']}\n"
        f"Note B: {b['text']}\n\n"
        "Do BOTH notes answer the question with different, "
        "incompatible answers? Equivalent wording is not a conflict. "
        "An unrelated note is not a conflict.\n"
        "Reply YES or NO only.",
        max_tokens=5,
    )
    return (response or "").strip().upper() == "YES"


def resolve(question: str, notes: list[dict]):
    """Answer using the trusted note and report conflicts."""
    chosen = pick(notes)

    conflict = any(
        disagree(question, chosen, note)
        for note in notes
        if note["id"] != chosen["id"]
    )

    source = f"{chosen['date']} {chosen['source']}"
    prompt = (
        f"Question: {question}\n"
        f"Trusted note: {chosen['text']}\n\n"
        "Answer the question using only the trusted note.\n"
        "Preserve exact numbers, dates, frequencies, and policy terms.\n"
        "Do not mention alternative answers or superseded rules.\n"
        "Do not add information not present in the trusted note.\n"
        + (
            f"Begin with 'Under the {chosen['date']} {chosen['source']},' "
            if conflict else
            "Answer directly without mentioning the source. "
        )
        + "Return exactly one sentence."
    )

    answer = ask_cached(prompt, max_tokens=60)

    return {
        "answer": (answer or "").strip(),
        "chosen": chosen["id"],
        "conflict": conflict,
    }
