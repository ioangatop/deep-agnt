def chunk_text(text: str, size: int = 160) -> list[str]:
    """Split text into token-limited chunks at line boundaries."""
    chunks = []
    current = ""

    for line in text.splitlines(keepends=True):
        if current and count_tokens(current + line) > size:
            chunks.append(current)
            current = ""

        current += line

    if current:
        chunks.append(current)

    return chunks


def chunk_repo(repo: dict[str, str]) -> list[dict[str, str]]:
    """Split the repo into chunks of at most CHUNK_TOKENS tokens, each from one file."""
    chunks = []
    for filename, context in repo.items():
        text_chunks = chunk_text(context, size=CHUNK_TOKENS)
        file_chunks = [{"path": filename, "text": text} for text in text_chunks]
        chunks.extend(file_chunks)
    return chunks


def build_index(chunks: list[dict[str, str]]) -> list[dict]:
    """Embed every chunk once. Return an index that keeps each chunk with its vector."""
    vectors = embedder.embed([chunk["text"] for chunk in chunks])
    return [
        {
            "path": chunk["path"],
            "text": chunk["text"],
            "vector": vector,
        }
        for chunk, vector in zip(chunks, vectors)
    ]


INDEX: list[dict[str, str | list[float]]] = None


def answer(question):
    """Answer from the chunks that fit the window. Returns {"answer", "files"}."""
    global INDEX

    if INDEX is None:
        INDEX = build_index(chunk_repo(REPO))

    query = embedder.embed([question])[0]
    ranked = sorted(
        INDEX,
        key=lambda chunk: cosine(query, chunk["vector"]),
        reverse=True,
    )

    prompt_template = (
        "Answer the question in one sentence using only the provided context.\n"
        "\nContext: {context}"
        f"\nQuestion: {question}"
    )

    context = ""
    files = []
    for chunk in ranked:
        section = (
            f"\nFile: {chunk['path']}"
            f"\n{chunk['text']}"
            "\n"
        )
        candidate = prompt_template.format(context=context + section)

        if count_tokens(candidate) > model.window:
            break

        context += section
        if chunk["path"] not in files:
            files.append(chunk["path"])

    reply = model.ask(
        prompt=prompt_template.format(context=context),
    )
    return {"answer": reply, "files": files}
