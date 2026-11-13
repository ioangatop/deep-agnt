def _execute_searches(response, searches: list[str]) -> list[dict]:
    """Execute tool calls and return their results."""
    results = []

    for call in response.tool_calls:
        query = call.input["query"]
        notes = retrieve(query)
        searches.append(query)

        results.append({
            "type": "tool_result",
            "tool_use_id": call.id,
            "content": "\n".join(note["text"] for note in notes),
        })

    return results


def answer_with_search(llm, question, max_turns=4):
    """Let the model search. Returns {"answer", "searches"}."""
    system = (
        "You are a retrieval assistant. "
        "Search for information before answering. "
        "If a retrieved note names an entity whose details are needed, "
        "search for that specific entity before answering. "
        "Use only retrieved evidence. "
        "Do not guess missing facts. "
        "Answer in one short sentence."
    )

    messages = [{"role": "user", "content": question}]
    searches = []
    turns = 0

    while turns < max_turns:
        response = llm.messages.create(
            system=system,
            messages=messages,
            tools=[SEARCH_TOOL],
            max_tokens=200,
        )
        turns += 1

        if response.stop_reason != "tool_use":
            return {
                "answer": response.text,
                "searches": searches,
            }

        messages.append({
            "role": "assistant",
            "content": [block.to_dict() for block in response.content],
        })

        results = _execute_searches(response, searches)
        messages.append({
            "role": "user",
            "content": results,
        })


    return {"answer": "", "searches": searches}
