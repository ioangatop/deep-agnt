def needs_search(message: str) -> bool:
    topics = ", ".join(note["id"] for note in HANDBOOK)

    # ERROR: gentkit.fake_llm.APIError: "The decision model is not configured."
    # score = jev.yes_no(
    #     state=f"The company handbook covers these topics: {topics}.",
    #     statement=(
    #         "Does answering this message require company-specific information "
    #         "on one of these topics? General knowledge, greetings, and "
    #         f"rewriting do not require the handbook.\n\nMessage: {message}"
    #     ),
    # )

    # FIX: use the LLM
    response = llm.ask(
        prompt=(
            f"The company handbook covers these topics: {topics}."
            "Does answering this message require company-specific information "
            "on one of these topics? General knowledge, greetings, and "
            f"rewriting do not require the handbook.\n\nMessage: {message}"
            "Reply only with a float between 0.0 and 1.0 indicating the confidence."
        )
    )
    score = float(response)

    return score >= 0.5


def respond(message: str) -> dict:
    do_search = needs_search(message)

    if do_search:
        notes = retrieve(message)
        context = "\n\n".join(note["text"] for note in notes)
        instructions = (
            "Answer using only these handbook notes. "
            "If they do not contain the answer, say so.\n\n"
            f"Handbook notes:\n{context}"
        )
    else:
        instructions = "Answer directly."

    prompt = (
        f"{instructions}\n\n"
        f"Message:\n{message}\n\n"
        "Reply in one sentence."
    )
    reply = llm.ask(prompt)

    return {"searched": do_search, "answer": reply.strip()}
