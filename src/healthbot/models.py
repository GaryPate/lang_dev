from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .config import settings


def build_chat_llm(model_name: str | None = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_name or settings.model_name,
        temperature=0.0,
        base_url=settings.base_url,
        api_key=settings.api_key,
    )


def build_embeddings(model_name: str | None = None) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=model_name or settings.embedding_model,
        base_url=settings.base_url,
        api_key=settings.api_key,
    )
