"""RAG (Retrieval-Augmented Generation) pipeline implementation."""

import logging
from pathlib import Path
from typing import Any

from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    CSVLoader,
    DirectoryLoader,
    TextLoader,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models import BaseLLM
from omegaconf import DictConfig

from ..models import BaseModel
from .react import LangChainModelWrapper

logger = logging.getLogger(__name__)


DEFAULT_QA_TEMPLATE = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""


class RAGPipeline:
    """RAG pipeline with document ingestion and retrieval."""

    def __init__(
        self,
        cfg: DictConfig,
        model: BaseModel | None = None,
        llm: BaseLLM | None = None,
    ):
        """Initialize RAG pipeline.

        Args:
            cfg: Agent configuration from Hydra.
            model: Optional BaseModel instance.
            llm: Optional LangChain LLM (overrides model).
        """
        self.cfg = cfg
        self._model = model
        self._llm = llm
        self._embeddings = None
        self._vectorstore = None
        self._retriever = None
        self._qa_chain = None

    def setup(self) -> None:
        """Set up embeddings, vector store, and QA chain."""
        self._embeddings = self._create_embeddings()
        self._vectorstore = self._load_or_create_vectorstore()
        self._retriever = self._create_retriever()

        if self._llm is None:
            self._llm = self._create_llm()

        self._qa_chain = self._create_qa_chain()

        logger.info("RAG pipeline initialized")

    def _create_embeddings(self) -> HuggingFaceEmbeddings:
        """Create embedding model."""
        emb_cfg = self.cfg.embeddings

        return HuggingFaceEmbeddings(
            model_name=emb_cfg.model,
            model_kwargs={"device": emb_cfg.get("device", "cpu")},
            encode_kwargs={"batch_size": emb_cfg.get("batch_size", 32)},
        )

    def _load_or_create_vectorstore(self) -> Chroma:
        """Load existing vector store or create empty one."""
        vs_cfg = self.cfg.vector_store
        persist_dir = Path(vs_cfg.persist_directory)

        if persist_dir.exists() and any(persist_dir.iterdir()):
            logger.info(f"Loading existing vector store from {persist_dir}")
            return Chroma(
                collection_name=vs_cfg.collection_name,
                embedding_function=self._embeddings,
                persist_directory=str(persist_dir),
            )
        else:
            logger.info(f"Creating new vector store at {persist_dir}")
            persist_dir.mkdir(parents=True, exist_ok=True)
            return Chroma(
                collection_name=vs_cfg.collection_name,
                embedding_function=self._embeddings,
                persist_directory=str(persist_dir),
            )

    def _create_retriever(self):
        """Create retriever with configured settings."""
        ret_cfg = self.cfg.retrieval

        search_kwargs = {"k": ret_cfg.k}

        if ret_cfg.score_threshold:
            search_kwargs["score_threshold"] = ret_cfg.score_threshold

        if ret_cfg.search_type == "mmr":
            search_kwargs["fetch_k"] = ret_cfg.fetch_k
            search_kwargs["lambda_mult"] = ret_cfg.lambda_mult

        return self._vectorstore.as_retriever(
            search_type=ret_cfg.search_type,
            search_kwargs=search_kwargs,
        )

    def _create_llm(self) -> BaseLLM:
        """Create LangChain LLM from model."""
        if self._model is None:
            raise ValueError("Either model or llm must be provided")

        self._model.load()
        return LangChainModelWrapper(self._model)

    def _create_qa_chain(self) -> RetrievalQA:
        """Create the QA chain."""
        prompt_template = self.cfg.prompts.get("qa_template", DEFAULT_QA_TEMPLATE)
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"],
        )

        chain_type = self.cfg.chain.type

        return RetrievalQA.from_chain_type(
            llm=self._llm,
            chain_type=chain_type,
            retriever=self._retriever,
            return_source_documents=self.cfg.chain.return_source_documents,
            chain_type_kwargs={"prompt": prompt},
            verbose=self.cfg.verbose,
        )

    def ingest_documents(
        self,
        source: str | Path,
        file_type: str = "txt",
    ) -> int:
        """Ingest documents from a file or directory.

        Args:
            source: Path to file or directory.
            file_type: Type of files to load (txt, csv).

        Returns:
            Number of documents ingested.
        """
        source = Path(source)
        documents = []

        if source.is_file():
            documents = self._load_file(source, file_type)
        elif source.is_dir():
            documents = self._load_directory(source, file_type)
        else:
            raise ValueError(f"Invalid source: {source}")

        chunks = self._split_documents(documents)

        if self._vectorstore is None:
            self.setup()

        self._vectorstore.add_documents(chunks)
        logger.info(f"Ingested {len(chunks)} chunks from {len(documents)} documents")

        return len(chunks)

    def _load_file(self, path: Path, file_type: str) -> list[Document]:
        """Load a single file."""
        if file_type == "csv":
            loader = CSVLoader(str(path))
        else:
            loader = TextLoader(str(path))

        return loader.load()

    def _load_directory(self, path: Path, file_type: str) -> list[Document]:
        """Load all files from a directory."""
        glob_pattern = f"**/*.{file_type}"

        if file_type == "csv":
            loader = DirectoryLoader(
                str(path),
                glob=glob_pattern,
                loader_cls=CSVLoader,
            )
        else:
            loader = DirectoryLoader(
                str(path),
                glob=glob_pattern,
                loader_cls=TextLoader,
            )

        return loader.load()

    def _split_documents(self, documents: list[Document]) -> list[Document]:
        """Split documents into chunks."""
        doc_cfg = self.cfg.documents

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=doc_cfg.chunk_size,
            chunk_overlap=doc_cfg.chunk_overlap,
            separators=list(doc_cfg.separators),
        )

        return splitter.split_documents(documents)

    def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Add texts directly to the vector store.

        Args:
            texts: List of text strings.
            metadatas: Optional metadata for each text.

        Returns:
            List of document IDs.
        """
        if self._vectorstore is None:
            self.setup()

        return self._vectorstore.add_texts(texts, metadatas=metadatas)

    def query(self, question: str) -> dict[str, Any]:
        """Query the RAG pipeline.

        Args:
            question: User question.

        Returns:
            Answer and source documents.
        """
        if self._qa_chain is None:
            self.setup()

        result = self._qa_chain.invoke({"query": question})

        return {
            "answer": result.get("result", ""),
            "source_documents": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                }
                for doc in result.get("source_documents", [])
            ],
        }

    def similarity_search(
        self,
        query: str,
        k: int = 4,
    ) -> list[dict[str, Any]]:
        """Perform similarity search without LLM.

        Args:
            query: Search query.
            k: Number of results.

        Returns:
            List of similar documents with scores.
        """
        if self._vectorstore is None:
            self.setup()

        results = self._vectorstore.similarity_search_with_score(query, k=k)

        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score,
            }
            for doc, score in results
        ]

    def clear_vectorstore(self) -> None:
        """Clear all documents from the vector store."""
        if self._vectorstore is not None:
            self._vectorstore.delete_collection()
            logger.info("Vector store cleared")
