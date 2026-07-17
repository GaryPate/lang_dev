import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings
from .models import build_embeddings


def load_markdown_docs(data_dir: str) -> list[Document]:
    docs: list[Document] = []
    manifest = load_manifest(data_dir)

    for path in Path(data_dir).glob("*.md"):
        text = path.read_text(encoding="utf-8")
        header_metadata, body = parse_header_metadata(text)
        docs.append(
            Document(
                page_content=body,
                metadata={
                    "path": str(path),
                    "filename": path.name,
                    **manifest.get(path.name, {}),
                    **header_metadata,
                },
            )
        )

    return docs


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(documents)

    enriched: list[Document] = []
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["chunk_id"] = f"{chunk.metadata.get('filename', 'doc')}::{idx}"
        enriched.append(chunk)

    return enriched


def build_vector_store() -> Chroma:
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=build_embeddings(),
        persist_directory=settings.chroma_dir,
    )


def ingest_documents(data_dir: str) -> int:
    raw_docs = load_markdown_docs(data_dir)
    chunks = split_documents(raw_docs)
    store = build_vector_store()
    store.add_documents(chunks)
    return len(chunks)


def load_manifest(data_dir: str) -> dict[str, dict]:
    manifest_path = Path(data_dir) / "manifest.json"
    if not manifest_path.exists():
        return {}

    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {entry["filename"]: entry for entry in entries}


def parse_header_metadata(text: str) -> tuple[dict, str]:
    metadata: dict[str, str] = {}
    body_lines: list[str] = []

    for line in text.splitlines():
        if line.startswith("- ") and ":" in line:
            key, value = line[2:].split(":", 1)
            metadata[key.strip()] = value.strip()
        else:
            body_lines.append(line)

    return metadata, "\n".join(body_lines).strip()
