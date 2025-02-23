from typing import Iterable, Literal, Optional, Sequence

import instructor
import jinja2
import openai
from loguru import logger
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from mutagrep.coderec.v3.symbol_mining import Symbol
from mutagrep.plan_search.domain_models import (CodeSearchInstrumentation,
                                                CodeSearchTool,
                                                CodeSearchToolOutput,
                                                RetrievedSymbol,
                                                SymbolRetriever)
from mutagrep.plan_search.typing_utils import implements

GENERATE_KEYWORDS_TEMPLATE = jinja2.Template(
    """
    You are a code search tool that transforms natural language goal into a list of keywords that can be used to search for Python symbols relevant to accomplish that goal.

    You will be given a goal expressed in natural language.
    Look at the goal and think about what keywords would be useful to find the Python symbols that are relevant to the goal.
    The keywords will be used to search for Python symbols in a repository.
    
    You do not need to list variants of the keywords. For example, if you think of the keyword "plot", you do not also need to include "plots" or "plotting".
    List as many keywords as you can that are relevant to the goal, but do not repeat keywords or list unnecessary keywords.

    {% if repo_description %}
    # Repository Description
    {{ repo_description }}
    {% endif %}

    # Goal
    {{ goal }}
    """,
    undefined=jinja2.StrictUndefined,
)

FILTER_SYMBOLS_TEMPLATE = jinja2.Template(
    """
    You are a code search tool that examines a list of symbols from a Python repository, and compares them against a goal to determine if any of the symbols can satisfy the goal.
    The set of symbols was retrieved using keyword-based search from a repository.

    # Symbols
    {% for symbol in symbols %}
    Import Path: {{ symbol.full_path }}
    Symbol Type: {{ symbol.symbol_type }}
    Code:
    ```python
    {{ symbol.code|truncate(length=1000, end='#... truncated due to length') }}
    ```
    {% endfor %}

    # Keywords
    {{ keywords }}

    # Goal 
    {{ goal }}

    # Guidelines
    - If there is a symbol that could plausibly satisfy the goal, select it and provide a justification for why that symbol is the best choice.
    - If there is no symbol that could plausibly satisfy the goal, find the closest symbol (if any) and provide a justification for why the goal is not satisfied by the available symbols.
    - Be generous in your interpretation of what a symbol could plausibly satisfy the goal.
    - Pay attention to the code and docstrings of the symbols to determine if they can be used to satisfy the goal.
    - There may be multiple symbols that could plausibly satisfy the goal. If so, return the most specific, narrowest symbol that satisfies the goal.
    """,
    undefined=jinja2.StrictUndefined,
)


class FilterSymbolsResponseModel(BaseModel):
    symbol_name: Optional[str]
    justification: Optional[str]
    satisfies_intention: bool


class OpenAiOneStepCodeSearchTool:
    def __init__(
        self,
        retriever: SymbolRetriever,
        results_per_keyword: int = 10,
        repo_description: str | None = None,
        return_cached_outputs: bool = False,
        model: Literal["gpt-4o-mini", "gpt-4o"] = "gpt-4o-mini",
    ):
        self.client = instructor.from_openai(openai.OpenAI())
        self.retriever = retriever
        self.results_per_keyword = results_per_keyword
        self.repo_description = repo_description
        self.return_cached_outputs = return_cached_outputs
        self.cache: dict[str, CodeSearchToolOutput] = {}
        self.model = model

    def cache_code_search_output(
        self, intention: str, output: CodeSearchToolOutput
    ) -> None:
        self.cache[intention] = output

    def get_cached_code_search_output(
        self, intention: str
    ) -> CodeSearchToolOutput | None:
        return self.cache.get(intention)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    def generate_keywords(self, goal: str) -> list[str]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": GENERATE_KEYWORDS_TEMPLATE.render(
                        goal=goal, repo_description=self.repo_description
                    ),
                }
            ],
            response_model=Iterable[str],  # type: ignore
        )
        keywords = list(response)
        logger.info(f"Searching for goal={goal} with keywords={keywords}")
        return keywords

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    def filter_symbols(
        self,
        retrieved_symbols: Sequence[RetrievedSymbol],
        keywords: list[str],
        goal: str,
    ) -> CodeSearchToolOutput:
        retrieved_symbols = sorted(
            retrieved_symbols, key=lambda x: x.score or 0, reverse=True
        )
        symbols = [rs.symbol for rs in retrieved_symbols]
        prompt = FILTER_SYMBOLS_TEMPLATE.render(
            symbols=symbols, keywords=keywords, goal=goal
        )
        response, completion = self.client.chat.completions.create_with_completion(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            response_model=FilterSymbolsResponseModel,
        )

        instrumentation = CodeSearchInstrumentation(
            symbols_considered=retrieved_symbols,
            completion_tokens=completion.usage.completion_tokens,
            prompt_tokens=completion.usage.prompt_tokens,
            total_tokens=completion.usage.total_tokens,
        )

        logger.info(f"Used {instrumentation.total_tokens} tokens for filtering symbols")

        code_search_output = CodeSearchToolOutput(
            symbol_name=response.symbol_name,
            justification=response.justification,
            satisfies_intention=response.satisfies_intention,
            instrumentation=instrumentation,
        )
        return code_search_output

    def retrieve_symbols_with_keywords(
        self, keywords: list[str]
    ) -> Sequence[RetrievedSymbol]:
        try:
            symbols = self.retriever(
                queries=keywords, n_results=self.results_per_keyword
            )
        except Exception:  # noqa: E722
            logger.opt(exception=True).error(
                f"Error retrieving symbols with keywords {keywords}"
            )
            return []

        logger.info(f"Retrieved {len(symbols)} symbols with keywords {keywords}")
        return symbols

    def get_unique_symbols(
        self, retrieved_symbols: Sequence[RetrievedSymbol]
    ) -> Sequence[RetrievedSymbol]:
        unique_symbols = []
        seen_symbols = set()
        for retrieved_symbol in retrieved_symbols:
            if retrieved_symbol.symbol.full_path not in seen_symbols:
                unique_symbols.append(retrieved_symbol)
                seen_symbols.add(retrieved_symbol.symbol.full_path)
        return unique_symbols

    def __call__(self, intention: str) -> CodeSearchToolOutput:
        if self.return_cached_outputs and (
            cached_output := self.get_cached_code_search_output(intention)
        ):
            logger.info(f"Returning cached output for intention {intention}")
            return cached_output

        keywords = self.generate_keywords(intention)
        symbols = self.retrieve_symbols_with_keywords(keywords)
        unique_symbols = self.get_unique_symbols(symbols)
        code_search_output = self.filter_symbols(unique_symbols, keywords, intention)
        self.cache_code_search_output(intention, code_search_output)
        return code_search_output


implements(CodeSearchTool)(OpenAiOneStepCodeSearchTool)
