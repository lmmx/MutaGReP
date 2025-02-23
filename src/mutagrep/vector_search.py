import json
from typing import Any, Generic, Optional, Protocol, Sequence, Type, TypeVar

import lancedb
from lancedb.pydantic import pydantic_to_schema
from openai import OpenAI
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct, ScoredPoint, UpdateResult
from typing_extensions import Self

from mutagrep.plan_search.domain_models import SymbolRetrievalScoreType
from mutagrep.plan_search.typing_utils import implements

BaseModelT = TypeVar("BaseModelT", bound=BaseModel)


class Embeddable(BaseModel, Generic[BaseModelT]):
    key: str
    payload: BaseModelT


class NullModel(BaseModel):
    pass


class RetrievedEmbeddable(BaseModel, Generic[BaseModelT]):
    score: float
    payload: BaseModelT = Field(default_factory=lambda: NullModel())  # type: ignore
    score_type: SymbolRetrievalScoreType = SymbolRetrievalScoreType.SIMILARITY

    @classmethod
    def from_scored_point(
        cls, scored_point: ScoredPoint, model: Type[BaseModelT]
    ) -> Self:
        attrs = scored_point.model_dump()
        # The payload is actually the Embeddable.
        embeddable = attrs.pop("payload")
        # The actual payload is the in the "payload" field of the Embeddable.
        payload = embeddable["payload"]
        structured_payload = model.model_validate(payload)
        return cls(score=scored_point.score, payload=structured_payload)


class OpenAIEmbedder:
    def __init__(self, embedding_model="text-embedding-ada-002", batch_size=10):
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.client = OpenAI()

    @property
    def embedding_size(self) -> int:
        return len(self(["foo"])[0])

    def batch_embed_openai(
        self,
        docs_to_embed: list[str],
        batch_size: int = 10,
        embedding_model="text-embedding-ada-002",
    ) -> list[list[float]]:
        embeddings = []
        for batch_start in range(0, len(docs_to_embed), batch_size):
            batch_end = batch_start + batch_size
            batch = docs_to_embed[batch_start:batch_end]
            response = self.client.embeddings.create(
                model=embedding_model,
                input=batch,
            )
            # Double check embeddings are in same order as input
            for i, be in enumerate(response.data):
                assert i == be.index
            batch_embeddings = [e.embedding for e in response.data]
            embeddings.extend(batch_embeddings)
        return embeddings

    def __call__(self, docs_to_embed: list[str]) -> list[list[float]]:
        return self.batch_embed_openai(
            docs_to_embed,
            batch_size=self.batch_size,
            embedding_model=self.embedding_model,
        )


class Embedder(Protocol):
    def __call__(self, docs_to_embed: list[str]) -> list[list[float]]: ...

    @property
    def embedding_size(self) -> int: ...


class QDrantVectorDatabase:
    def __init__(
        self, embedder: Optional[Embedder] = None, vector_database_url: str = ":memory:"
    ):
        self.vector_database_url = vector_database_url
        self.client = QdrantClient(self.vector_database_url)
        self.collection_names: set[str] = set()
        self.embedder = embedder or OpenAIEmbedder()

    def create_collection(self, collection_name: str) -> bool:
        success = self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=self.embedder.embedding_size,
                distance=models.Distance.COSINE,
            ),
        )
        return success

    def insert(
        self, docs: list[dict[str, Any]], collection_name: str = "default"
    ) -> UpdateResult:
        """
        Insert a list of documents into a QDrant vector database.

        Parameters:
        -----------
        docs: list[dict[str, Any]]
            A list of documents to insert into the vector database. Each document should have a "key" field.
            The key field will be embedded and used for similarity search.
        collection_name: str
            The name of the collection to insert the documents into.

        Returns:
        --------
        operation_info: UpdateResult
            Information about the operation.
        """
        if collection_name not in self.collection_names:
            if self.create_collection(collection_name):
                self.collection_names.add(collection_name)
            else:
                raise RuntimeError(f"Could not create collection {collection_name}")

        embeddings = self.embedder([doc["key"] for doc in docs])
        operation_info = self.client.upsert(
            collection_name=collection_name,
            wait=True,
            points=[
                PointStruct(id=idx, vector=embedding, payload=payload)
                for idx, (embedding, payload) in enumerate(zip(embeddings, docs))
            ],
        )
        return operation_info

    def search(
        self,
        query: str,
        collection_name: str = "default",
        limit: int = 10,
    ) -> list[ScoredPoint]:
        embeddings = self.embedder([query])
        search_result = self.client.search(
            collection_name=collection_name,
            query_vector=embeddings[0],
            limit=limit,
        )
        return search_result


class ObjectVectorDatabase(Protocol[BaseModelT]):
    def insert(self, docs: Sequence[Embeddable[BaseModelT]]) -> Any: ...

    def search(
        self, query: str, limit: int = 10
    ) -> Sequence[RetrievedEmbeddable[BaseModelT]]: ...


class PydanticQdrantVectorDatabase(Generic[BaseModelT]):
    def __init__(self, vector_database: QDrantVectorDatabase, model: Type[BaseModelT]):
        self.vector_database = vector_database
        self.model = model

    def insert(self, docs: Sequence[Embeddable[BaseModelT]]) -> Any:
        docs_unstructured = [doc.model_dump() for doc in docs]
        return self.vector_database.insert(docs_unstructured)

    def search(
        self, query: str, limit: int = 10
    ) -> Sequence[RetrievedEmbeddable[BaseModelT]]:
        scored_points = self.vector_database.search(query, limit=limit)
        return [
            RetrievedEmbeddable.from_scored_point(sp, self.model)
            for sp in scored_points
        ]


class PydanticLancedbVectorDatabase(Generic[BaseModelT]):
    def __init__(
        self,
        database_url: str,
        model: Type[BaseModelT],
        embedder: Optional[Embedder] = None,
        table_name: str = "default.lancedb",
    ):
        self.database_url = database_url
        self.model = model
        self.embedder = embedder or OpenAIEmbedder()
        self.db = lancedb.connect(self.database_url)
        self.table_name = table_name

    def insert(self, docs: Sequence[Embeddable[BaseModelT]]) -> Any:
        embeddings = self.embedder([doc.key for doc in docs])
        data = [
            {"vector": embedding, "payload": json.loads(doc.model_dump_json())}
            for embedding, doc in zip(embeddings, docs)
        ]
        try:
            tbl = self.db.open_table(self.table_name)
            tbl.add(data)
        except FileNotFoundError:
            tbl = self.db.create_table(self.table_name, data=data)

    def search(
        self, query: str, limit: int = 10
    ) -> Sequence[RetrievedEmbeddable[BaseModelT]]:
        embedding = self.embedder([query])[0]
        results = self.db[self.table_name].search(embedding).limit(limit).to_list()
        retrieved_embeddables: list[RetrievedEmbeddable[BaseModelT]] = []
        for result in results:
            retrieved_embeddable = RetrievedEmbeddable(
                score=result["_distance"],
                payload=self.model.model_validate(result["payload"]["payload"]),
                score_type=SymbolRetrievalScoreType.DISTANCE,
            )
            retrieved_embeddables.append(retrieved_embeddable)
        return retrieved_embeddables


implements(ObjectVectorDatabase)(PydanticLancedbVectorDatabase)
