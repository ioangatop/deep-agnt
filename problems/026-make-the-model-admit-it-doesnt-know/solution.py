import inspect

DONT_KNOW = "I don't know."


def grounded_answer(llm, question: str, context: list[str]) -> str:
    """Answer only from `context`, or return DONT_KNOW."""
    prompt_template = inspect.cleandoc(f"""
        Answer from the context. If the context does not contain the answer,
        reply with {DONT_KNOW}

        QUESTION:
        {{question}}

        CONTEXT:
        {{context}}
    """)

    prompt = prompt_template.format(question=question, context=context)
    response = llm.ask(prompt)

    if DONT_KNOW in response:
        return response

    return response
