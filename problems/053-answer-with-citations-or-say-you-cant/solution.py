import json
import re


def _json_reply(text):
    """Parse the JSON in a model reply, fenced or not. None if there is none."""
    text = text or ""
    fence = chr(96) * 3
    if fence in text:
        text = text.split(fence)[1]
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    match = re.search(r"[\[{].*[\]}]", text, re.S)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except ValueError:
        return None


from agentkit.llm import LLM

llm = LLM()

DOCS = [
    "Parking permits are issued by the facilities office on the second floor.",
    "The cafeteria serves breakfast from seven until ten every weekday morning.",
    "Meeting rooms can be booked up to two weeks ahead through the calendar app.",
    "Refunds are accepted within 30 days of purchase with the original receipt.",
]


from inspect import cleandoc


def answer_with_citations(llm: LLM, question: str, docs: list[str]):
    """Answer from the docs with citations, or say the docs don't cover it."""

    prompt_template = cleandoc("""
        Answer the question using only the grounding documents.
        Cite supporting documents by their zero-based indices.
        If unsupported, return answer="I can't answer that from these documents",
        citations=[], and supported=false.

        Return only valid JSON with this structure:
        {{
            "answer": "Your answer",
            "citations": [0],
            "supported": true
        }}

        Question:
        {question}

        Documents:
        {documents}
    """)

    grounding_docs = "\n".join([f"{i}: {doc}" for i, doc in enumerate(docs)])
    prompt = prompt_template.format(question=question, documents=grounding_docs)

    response = llm.ask(prompt)
    result = _json_reply(response)

    # filter citations and update supported
    citations = [
        index
        for index in result.get("citations", [])
        if type(index) is int and 0 <= index < len(docs)
    ]
    supported = result.get("supported") is True and bool(citations)

    return {
        "answer": (
            result["answer"]
            if supported
            else "I can't answer that from these documents"
        ),
        "citations": citations if supported else [],
        "supported": supported,
    }
