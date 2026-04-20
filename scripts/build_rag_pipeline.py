#!/usr/bin/env python3
"""Task 3.3: Core RAG pipeline over an existing Chroma vector store.

This script:
- loads a persisted Chroma vector store,
- resolves collection naming safely (prefers kb_store, falls back to legacy kb0),
- retrieves top-k documents with all-mpnet-base-v2-compatible embeddings,
- generates an answer using a selectable backend:
    - Hugging Face causal LM (default: Meta-Llama 3.1 8B Instruct), or
    - local Ollama API (default Ollama model: llama3.1:8b),
- prints both answer and retrieved source snippets.

Note:
- TinyLlama (or other small public models) can be used for smoke testing only.
- Final project validation for Task 3.3 should run with Meta-Llama-3.1-8B-Instruct
    (with proper Hugging Face gated-model access).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import chromadb
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

LOGGER = logging.getLogger("build_rag_pipeline")

PREFERRED_PHYSICAL = "kb_store"
LEGACY_PHYSICAL = "kb0"
LOGICAL_DEFAULT = "kb"

GEN_BACKEND_HF = "huggingface"
GEN_BACKEND_OLLAMA = "ollama"


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def list_collection_names(client: chromadb.PersistentClient) -> List[str]:
    raw = client.list_collections()
    names: List[str] = []
    for item in raw:
        if isinstance(item, str):
            names.append(item)
        else:
            name = getattr(item, "name", None)
            if name:
                names.append(name)
    return names


def resolve_collection_name(requested_name: str, available: Sequence[str]) -> str:
    available_set = set(available)

    candidates: List[str] = []
    if requested_name == LOGICAL_DEFAULT:
        candidates = [PREFERRED_PHYSICAL, LEGACY_PHYSICAL, LOGICAL_DEFAULT]
    elif requested_name == PREFERRED_PHYSICAL:
        candidates = [PREFERRED_PHYSICAL, LEGACY_PHYSICAL]
    else:
        candidates = [requested_name]

    for name in candidates:
        if name in available_set:
            if name != requested_name:
                LOGGER.warning(
                    "Requested collection '%s' resolved to '%s' based on availability.",
                    requested_name,
                    name,
                )
            return name

    raise ValueError(
        f"Could not resolve a collection for requested='{requested_name}'. "
        f"Available collections: {sorted(available_set)}"
    )


def load_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    LOGGER.info("Loading embeddings model for retrieval: %s", model_name)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_llm(model_path: str, max_new_tokens: int, temperature: float) -> HuggingFacePipeline:
    LOGGER.info("Loading generation model: %s", model_path)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    except Exception as exc:
        exc_text = str(exc).lower()
        is_gated_or_auth = any(
            token in exc_text
            for token in (
                "gated",
                "401",
                "unauthorized",
                "access to model",
                "cannot access gated repo",
                "hf_token",
            )
        )

        if is_gated_or_auth:
            raise RuntimeError(
                "Failed to load the requested model because access/authentication is missing. "
                "For meta-llama/Meta-Llama-3.1-8B-Instruct, ensure your HF account has accepted "
                "the model license and authenticate on the machine (e.g., `huggingface-cli login`) "
                "or set `HF_TOKEN` in the environment before running."
            ) from exc

        raise RuntimeError(
            "Failed to load generation model due to a non-auth error. "
            "Check model path validity, transformers/torch compatibility, and hardware memory. "
            f"Original error: {exc}"
        ) from exc

    text_gen = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        return_full_text=False,
    )
    return HuggingFacePipeline(pipeline=text_gen)


class OllamaGenerator:
    def __init__(self, base_url: str, model: str, max_new_tokens: int, temperature: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        endpoint = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": self.max_new_tokens,
                "temperature": self.temperature,
            },
        }

        req = Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(req, timeout=300) as resp:
                body = resp.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(
                f"Ollama HTTP error from {endpoint}: status={exc.code}, body={detail}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                "Failed to reach Ollama API. Ensure Ollama is installed and running locally "
                f"at {self.base_url}."
            ) from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON response from Ollama: {body[:400]}") from exc

        response_text = parsed.get("response", "")
        if not response_text:
            raise RuntimeError(f"Ollama returned empty response payload: {parsed}")
        return str(response_text).strip()


def format_docs(docs: Sequence[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for i, d in enumerate(docs, start=1):
        meta = d.get("metadata", {})
        title = meta.get("title", "Unknown")
        source = meta.get("source", "unknown")
        chunk_index = meta.get("chunk_index", "?")
        content = d.get("text", "")
        parts.append(
            f"[Doc {i}] title={title} source={source} chunk_index={chunk_index}\n{content}"
        )
    return "\n\n".join(parts)


def retrieve_documents(
    collection: Any,
    embeddings: HuggingFaceEmbeddings,
    question: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    q_emb = embeddings.embed_query(question)
    result = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs: List[Dict[str, Any]] = []
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    for text, metadata, distance in zip(documents, metadatas, distances):
        docs.append(
            {
                "text": text or "",
                "metadata": metadata or {},
                "distance": distance,
            }
        )
    return docs


def run_query(
    collection: Any,
    embeddings: HuggingFaceEmbeddings,
    generation_backend: str,
    llm: Optional[HuggingFacePipeline],
    ollama_generator: Optional[OllamaGenerator],
    question: str,
    top_k: int,
) -> tuple[str, List[Dict[str, Any]]]:
    docs = retrieve_documents(collection, embeddings, question, top_k=top_k)

    prompt = PromptTemplate.from_template(
    """You are a QA extraction system.
Use only the provided context.
Return only a short answer span (prefer 1-6 words).
Do not provide explanations.
Do not copy full sentences unless absolutely necessary.
If the answer is not in the context, return exactly: unknown

Question:
{question}

Context:
{context}

Answer:"""
    )

    prompt_text = prompt.format(question=question, context=format_docs(docs))

    if generation_backend == GEN_BACKEND_HF:
        if llm is None:
            raise RuntimeError("Hugging Face backend selected but model is not initialized.")
        chain = llm | StrOutputParser()
        answer = chain.invoke(prompt_text)
    elif generation_backend == GEN_BACKEND_OLLAMA:
        if ollama_generator is None:
            raise RuntimeError("Ollama backend selected but Ollama client is not initialized.")
        answer = ollama_generator.generate(prompt_text)
    else:
        raise ValueError(f"Unsupported generation backend: {generation_backend}")

    return answer.strip(), list(docs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build/run core RAG pipeline over Chroma vectorstore.")
    parser.add_argument("--vectorstore-path", type=str, default="data/vectorstore/clean")
    parser.add_argument("--collection-name", type=str, default="kb")
    parser.add_argument(
        "--generation-backend",
        type=str,
        default=GEN_BACKEND_HF,
        choices=[GEN_BACKEND_HF, GEN_BACKEND_OLLAMA],
        help="Generation backend to use.",
    )
    parser.add_argument("--model-path", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--ollama-model", type=str, default="llama3.1:8b")
    parser.add_argument("--ollama-base-url", type=str, default="http://localhost:11434")
    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--question", type=str, default="")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    vec_path = Path(args.vectorstore_path)
    if not vec_path.exists():
        raise FileNotFoundError(f"Vectorstore path not found: {vec_path}")

    client = chromadb.PersistentClient(path=str(vec_path))
    available = list_collection_names(client)
    LOGGER.info("Available collections in '%s': %s", vec_path, available)

    resolved_collection = resolve_collection_name(args.collection_name, available)
    LOGGER.info(
        "Resolved collection name for retrieval: requested='%s' -> using='%s'",
        args.collection_name,
        resolved_collection,
    )

    embeddings = load_embeddings(args.embedding_model)
    collection = client.get_collection(name=resolved_collection)

    llm: Optional[HuggingFacePipeline] = None
    ollama_generator: Optional[OllamaGenerator] = None

    if args.generation_backend == GEN_BACKEND_HF:
        llm = load_llm(
            model_path=args.model_path,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    elif args.generation_backend == GEN_BACKEND_OLLAMA:
        LOGGER.info(
            "Using Ollama backend with model='%s' at base_url='%s'",
            args.ollama_model,
            args.ollama_base_url,
        )
        ollama_generator = OllamaGenerator(
            base_url=args.ollama_base_url,
            model=args.ollama_model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    else:
        raise ValueError(f"Unsupported generation backend: {args.generation_backend}")

    question = args.question.strip()
    if not question:
        question = input("Enter question: ").strip()

    if not question:
        raise ValueError("No question provided.")

    answer, docs = run_query(
        collection=collection,
        embeddings=embeddings,
        generation_backend=args.generation_backend,
        llm=llm,
        ollama_generator=ollama_generator,
        question=question,
        top_k=args.top_k,
    )

    print("\n=== Generated Answer ===")
    print(answer)

    print("\n=== Retrieved Sources ===")
    for i, doc in enumerate(docs, start=1):
        title = doc.get("metadata", {}).get("title", "Unknown")
        preview = doc.get("text", "").replace("\n", " ")[:180]
        print(f"{i}. {title}")
        print(f"   {preview}")


if __name__ == "__main__":
    main()
