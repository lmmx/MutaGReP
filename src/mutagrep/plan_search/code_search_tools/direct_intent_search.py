from mutagrep.plan_search.domain_models import (
    CodeSearchInstrumentation,
    CodeSearchTool,
    CodeSearchToolOutput,
    RetrievedSymbol,
    SymbolRetriever,
)
from mutagrep.plan_search.typing_utils import implements


class DirectIntentSearchTool:
    def __init__(self, symbol_retriever: SymbolRetriever, symbols_to_retrieve: int = 5):
        self.symbol_retriever = symbol_retriever
        self.symbols_to_retrieve = symbols_to_retrieve

    def __call__(self, intention: str) -> CodeSearchToolOutput:
        retrieved_symbols = self.symbol_retriever(
            queries=[intention], n_results=self.symbols_to_retrieve
        )
        return CodeSearchToolOutput(
            satisfies_intention=True,
            symbol_name=retrieved_symbols[0].symbol.full_path,
            justification="These symbols are capable of satisfying the intention",
            instrumentation=CodeSearchInstrumentation(
                symbols_considered=list(retrieved_symbols),
                completion_tokens=0,
                prompt_tokens=0,
                total_tokens=0,
            ),
        )


implements(CodeSearchTool)(DirectIntentSearchTool)


class NoDuplicatesDirectIntentSearchTool:
    def __init__(
        self,
        symbol_retriever: SymbolRetriever,
        symbols_to_retrieve: int = 5,
        overretrieve_factor: int = 5,
    ):
        """
        This tool will retrieve more symbols than requested and then deduplicate them.

        Parameters:
            symbol_retriever: The symbol retriever to use.
            symbols_to_retrieve: The number of symbols to retrieve.
            overretrieve_factor: The factor by which to overretrieve symbols.
        """
        self.symbol_retriever = symbol_retriever
        self.symbols_to_retrieve = symbols_to_retrieve
        self.overretrieve_factor = overretrieve_factor

    def deduplicate_output(self, output: CodeSearchToolOutput) -> CodeSearchToolOutput:
        assert output.instrumentation is not None
        # Get the total number of symbols we retrieved.
        num_symbols = len(output.instrumentation.symbols_considered)
        # Ensure that they are sorted by relevance.
        top_n_symbols = output.get_top_n_symbols(num_symbols)
        # Now we need to deduplicate them.
        seen_symbols: set[str] = set()
        unique_symbols: list[RetrievedSymbol] = []
        for retrieved_symbol in top_n_symbols:
            if retrieved_symbol.symbol.full_path in seen_symbols:
                continue
            seen_symbols.add(retrieved_symbol.symbol.full_path)
            unique_symbols.append(retrieved_symbol)
            if len(unique_symbols) == self.symbols_to_retrieve:
                break
        return CodeSearchToolOutput(
            satisfies_intention=True,
            symbol_name=unique_symbols[0].symbol.full_path,
            justification="These symbols are capable of satisfying the intention",
            instrumentation=CodeSearchInstrumentation(
                symbols_considered=unique_symbols,
                completion_tokens=0,
                prompt_tokens=0,
                total_tokens=0,
            ),
        )

    def __call__(self, intention: str) -> CodeSearchToolOutput:
        num_symbols_to_retrieve = self.symbols_to_retrieve * self.overretrieve_factor
        retrieved_symbols = self.symbol_retriever(
            queries=[intention], n_results=num_symbols_to_retrieve
        )
        output_maybe_duplicates = CodeSearchToolOutput(
            satisfies_intention=True,
            symbol_name=retrieved_symbols[0].symbol.full_path,
            justification="These symbols are capable of satisfying the intention",
            instrumentation=CodeSearchInstrumentation(
                symbols_considered=list(retrieved_symbols),
                completion_tokens=0,
                prompt_tokens=0,
                total_tokens=0,
            ),
        )

        return self.deduplicate_output(output_maybe_duplicates)


implements(CodeSearchTool)(NoDuplicatesDirectIntentSearchTool)
