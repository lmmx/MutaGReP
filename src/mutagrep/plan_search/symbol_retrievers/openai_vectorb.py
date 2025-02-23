from pathlib import Path
from typing import Sequence

from typing_extensions import Self

from mutagrep.coderec.v3.symbol_mining import Symbol
from mutagrep.plan_search.domain_models import RetrievedSymbol, SymbolRetriever
from mutagrep.plan_search.typing_utils import implements
from mutagrep.vector_search import (Embeddable, ObjectVectorDatabase,
                                    OpenAIEmbedder,
                                    PydanticLancedbVectorDatabase,
                                    RetrievedEmbeddable)


class OpenAiVectorSearchSymbolRetriever:
    def __init__(
        self,
        vector_database: ObjectVectorDatabase[Symbol],
    ):
        """
        Parameters:
        -----------
        vector_database: ObjectVectorDatabase[Symbol]
            The vector database to use for storing and retrieving symbols.
        """
        self.vector_database = vector_database

    @classmethod
    def instantiate_from_path(
        cls, path: Path, deduplicate_retrievals: bool = False
    ) -> Self:
        """
        Instantiate from a filepath (db need not exist) using the right embedder and model.
        """
        vector_database = PydanticLancedbVectorDatabase(
            database_url=str(path), model=Symbol, embedder=OpenAIEmbedder()
        )
        return cls(vector_database)

    def index_single(self, key: str, symbol: Symbol) -> None:
        embeddable = Embeddable(key=key, payload=symbol)
        self.index([embeddable])

    def index(self, embeddables: Sequence[Embeddable[Symbol]]) -> None:
        self.vector_database.insert(embeddables)

    def __call__(
        self, queries: Sequence[str], n_results: int = 5
    ) -> Sequence[RetrievedSymbol]:
        retrieved_embeddables: list[RetrievedEmbeddable[Symbol]] = []
        for query in queries:
            retrieved_embeddables.extend(self.vector_database.search(query, n_results))

        retrieved_symbols: list[RetrievedSymbol] = []
        for retrieved_embeddable in retrieved_embeddables:
            retrieved_symbols.append(
                RetrievedSymbol(
                    symbol=retrieved_embeddable.payload,
                    score=retrieved_embeddable.score,
                    score_type=retrieved_embeddable.score_type,
                )
            )
        return retrieved_symbols


implements(SymbolRetriever)(OpenAiVectorSearchSymbolRetriever)
