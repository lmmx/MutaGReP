from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Collection, Iterable

import click
import instructor
import jinja2
import tiktoken
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from mutagrep.coderec.v3.symbol_mining import RankableSymbol, Symbol
from mutagrep.repository_accounting import known_repos
from mutagrep.utilities import PydanticJSONLinesReader, PydanticJSONLinesWriter


# Instructor _needs_ a model to be defined in order to work. Our model is very simple, just a string.
class Intent(BaseModel):
    intent: str


# This is all the information needed to uniquely identify a symbol.
class IntentForSymbol(Intent):
    symbol_full_path: str
    repo_name: str


class MetadataForPromptConstruction(BaseModel):
    repo_name: str


def generate_intents_for_symbols(
    symbols: Collection[Symbol],
    output_writer: Callable[[IntentForSymbol], None],
    intent_generator: Callable[
        [Symbol, MetadataForPromptConstruction], Iterable[Intent]
    ],
    metadata: MetadataForPromptConstruction,
    parallel: bool = True,
    max_workers: int = 8,
):
    def process_symbol(symbol):
        intents = intent_generator(symbol, metadata)
        return [
            IntentForSymbol(
                intent=intent.intent,
                symbol_full_path=symbol.full_path,
                repo_name=metadata.repo_name,
            )
            for intent in intents
        ]

    if parallel:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_symbol, symbol): symbol for symbol in symbols
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Generating intents for symbols",
                unit="symbol",
            ):
                intents_for_symbol = future.result()
                for intent in intents_for_symbol:
                    output_writer(intent)
    else:
        for symbol in tqdm(
            symbols, desc="Generating intents for symbols", unit="symbol"
        ):
            intents_for_symbol = process_symbol(symbol)
            for intent in intents_for_symbol:
                output_writer(intent)


def is_test_heuristic(symbol: Symbol) -> bool:
    symbol_full_path_parts = symbol.full_path.split(".")
    for _ in symbol_full_path_parts:
        if "test" in _:
            return True

    return False


def num_tokens_from_string(
    string: str | None, encoding_name: str = "cl100k_base"
) -> int:
    """Returns the number of tokens in a text string."""
    if string is None:
        return 0
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def truncate_string_if_too_long(
    string: str, max_tokens: int, encoding_name: str = "cl100k_base", log: bool = False
) -> str:
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(string)
    if len(tokens) <= max_tokens:
        return string
    if log:
        logger.warning(
            "Truncating string because it is too long. Original length: {original_length}, truncated length: {truncated_length}",
            original_length=len(tokens),
            truncated_length=max_tokens,
        )
    truncated_tokens = tokens[:max_tokens]
    truncated_string = encoding.decode(truncated_tokens)
    return truncated_string


class OpenAIIntentGenerator:
    def __init__(self, model: str = "gpt-4o-mini", num_intents_per_symbol: int = 5):
        self.model = model
        self.num_intents_per_symbol = num_intents_per_symbol
        self.client = instructor.from_openai(OpenAI())
        self.intent_generation_template = jinja2.Template(
            """You are an expert Python developer. 
You will be teaching junior developers learn to use a specific codebase well.
You will be helping them memorize the symbols (functions, classes) in the codebase.
The specific skill you will be teaching them is to be able to recall the right symbol to use when given an intent to perform a specific task within the codebase.

You will be given the fully qualified name of the symbol in the codebase, and the name of the codebase itself.
You will be asked to generate a list of intents. 
Each intent should be written in the first person, as if you are giving the intent to the junior developer.
Each intent should be unique and different from the other intents.

For example, if the symbol is "torch.optim.Adam" in the codebase "pytorch", you might generate the following intents:
- "I want to optimize the parameters of my neural network using an optimizer that has adaptive learning rates and momentum."

As another example, if the symbol is a function "torch.nn.functional.relu" in the codebase "pytorch", you might generate the following intents:
- "I need an activation function that will set all negative values to zero."

As another example, if the symbol is a method "django.db.models.Model.save" in the codebase "django", you might generate the following intents:
- "I want to save an instance of a model to the database."

Please generate a list of {{ num_intents }} intents for the following symbol:

Codebase: {{ repo_name }}
Symbol: {{ symbol_fully_qualified_name }}
Symbol Type: {{ symbol_type }}
Symbol Code: 
```python
{{ symbol_code }}
```"""
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def __call__(
        self, symbol: Symbol, metadata: MetadataForPromptConstruction
    ) -> Iterable[Intent]:

        prompt = self.intent_generation_template.render(
            repo_name=metadata.repo_name,
            symbol_fully_qualified_name=symbol.full_path,
            symbol_code=truncate_string_if_too_long(
                symbol.code or "", max_tokens=10_000, log=True
            ),
            symbol_type=symbol.symbol_type.value,
            num_intents=self.num_intents_per_symbol,
            lstrip_blocks=True,
            trim_blocks=True,
        )

        intents = self.client.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_model=Iterable[Intent],  # type: ignore
        )

        return intents


def generate_intents_for_all_symbols(
    symbols_dir: Path,
    output_dir: Path,
    intent_generator: Callable[
        [Symbol, MetadataForPromptConstruction], Iterable[Intent]
    ] = OpenAIIntentGenerator(),
    idempotent: bool = True,
) -> None:
    with logging_redirect_tqdm():
        for repo_name in tqdm(
            known_repos, desc="Generating intents for all repos", unit="repo"
        ):
            path_to_symbol_jsonl = symbols_dir / f"{repo_name}.jsonl"
            if not path_to_symbol_jsonl.exists():
                logger.warning(f"Skipping {repo_name} because it doesn't have symbols")
                continue

            logger.info("Generating intents for {repo_name}", repo_name=repo_name)

            symbol_reader = PydanticJSONLinesReader(
                str(path_to_symbol_jsonl), RankableSymbol
            )
            symbols = list(symbol_reader())

            non_test_symbols = [
                _.symbol for _ in symbols if not is_test_heuristic(_.symbol)
            ]

            logger.info(
                "Repo {repo_name} has {num_symbols} symbols, {num_non_test_symbols} of which are non-test",
                repo_name=repo_name,
                num_symbols=len(symbols),
                num_non_test_symbols=len(non_test_symbols),
            )

            path_to_intent_jsonl = output_dir / f"{repo_name}.jsonl"

            if idempotent and path_to_intent_jsonl.exists():
                logger.info(
                    "Repo {repo_name} already has intents generated",
                    repo_name=repo_name,
                )

                # Read the intents so we we can skip generating any we have already generated.
                intent_reader = PydanticJSONLinesReader(
                    str(path_to_intent_jsonl), IntentForSymbol
                )
                existing_intents = list(intent_reader())

                # Remove any symbols from non_test_symbols that we have already generated intents for.
                symbols_already_generated = {
                    _.symbol_full_path for _ in existing_intents
                }

                logger.info(
                    "Found {num_existing_intents} existing intents",
                    num_existing_intents=len(existing_intents),
                )

                non_test_symbols = [
                    _
                    for _ in non_test_symbols
                    if _.full_path not in symbols_already_generated
                ]

                logger.info(
                    "Repo {repo_name} has {num_non_test_symbols} non-test symbols that don't have intents generated yet",
                    repo_name=repo_name,
                    num_non_test_symbols=len(non_test_symbols),
                )

            path_to_intent_jsonl.parent.mkdir(parents=True, exist_ok=True)

            intent_writer = PydanticJSONLinesWriter(str(path_to_intent_jsonl))

            generate_intents_for_symbols(
                symbols=non_test_symbols,
                output_writer=intent_writer,
                intent_generator=intent_generator,
                metadata=MetadataForPromptConstruction(repo_name=repo_name),
            )


@click.group()
def cli():
    pass


@cli.command("generate-all")
@click.option(
    "--symbols-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    required=True,
)
@click.option(
    "--idempotent",
    type=bool,
    default=True,
    help="If True, will only generate intents for symbols that don't already have intents generated",
)
def generate_all(symbols_dir: Path, output_dir: Path, idempotent: bool):
    generate_intents_for_all_symbols(symbols_dir, output_dir, idempotent=idempotent)


if __name__ == "__main__":
    cli()
