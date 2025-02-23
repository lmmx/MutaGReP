# Repository Structure
#### 1. `coderec/`
- **`symbol_mining.py`**: Defines structures for representing and extracting code symbols.
- **`intent_generation.py`**: Implements intent generation for symbols via OpenAI models.

#### 2. `plan_search/`
- **`domain_models.py`**: Core data structures and protocols.
- **`generic_search.py`**: Implements the main search algorithm for plan exploration.
- **`code_search_tools/`**: Tools for searching a codebase to find symbols that satisfy a query. Code search tools generally make use of a `SymbolRetriever` to find candidate symbols.
- **`symbol_retrievers/`**: Implementations of different symbol retrievers. We have a BM25s based retriever and a vector based retriever that uses OpenAI embeddings.
- **`rankers/`**: Implementations of ways to rank plans.  
- **`successor_functions/`**: Successor function implementations. The ones used in the paper correspond to `xml_like_sampling_unconstrained.py` and `xml_like_sampling_constrained.py` for unconstrained and monotonic mutation respectively.
- **`lca_benchmark/`**: Utilities for scoring and evaluating plans on the LongCodeArena dataset.
#### 3. `vector_search.py`
- Implements vector-based search, defining interfaces and specific database backends.
#### 4. General Utilities
- **`utilities.py`**: Includes JSONL readers and writers.