import datetime as dt
import json
import random
from functools import cached_property
from pathlib import Path
from typing import Literal, Optional

import jinja2
import numpy as np
from gitingest import ingest
from loguru import logger
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from tqdm.auto import tqdm
from typing_extensions import assert_never
from ulid import ULID

from mutagrep.coderec.v3.intent_generation import (
    IntentForSymbol,
    MetadataForPromptConstruction,
    OpenAIIntentGenerator,
    generate_intents_for_symbols,
)
from mutagrep.coderec.v3.symbol_mining import (
    Symbol,
    extract_all_symbols_under_directory,
)
from mutagrep.plan_search.code_search_tools.direct_intent_search import (
    NoDuplicatesDirectIntentSearchTool,
)
from mutagrep.plan_search.components import (
    AlwaysReturnsGoalTestFalse,
    GoalTest,
    PlanStep,
)
from mutagrep.plan_search.containers import PriorityQueueSearchContainer
from mutagrep.plan_search.domain_models import (
    Node,
    Plan,
    RankingFunction,
    SymbolRetriever,
)
from mutagrep.plan_search.generic_search import PlanSearcher, SearchResult
from mutagrep.plan_search.rankers.most_unique_symbols import MostUniqueSymbolsRanker
from mutagrep.plan_search.rankers.step_level_likert_llm_judge_v2 import (
    JudgeResponse,
    StepLevelLikertLlmJudge,
)
from mutagrep.plan_search.successor_functions.xml_like_sampling_unconstrained import (
    UnconstrainedXmlOutputSuccessorFunction,
)
from mutagrep.plan_search.symbol_retrievers.openai_vectorb import (
    OpenAiVectorSearchSymbolRetriever,
)
from mutagrep.utilities import (
    PydanticJSONLinesReader,
    PydanticJSONLinesWriter,
    SqliteKeyValueStore,
)
from mutagrep.vector_search import (
    Embeddable,
    ObjectVectorDatabase,
    OpenAIEmbedder,
    PydanticLancedbVectorDatabase,
)


def mine_symbols(repo_path: Path, cache_path: Path) -> list[Symbol]:
    save_path = cache_path / repo_path.name / "symbols.jsonl"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists():
        reader = PydanticJSONLinesReader(str(save_path), model=Symbol)
        symbols = list(reader())
        logger.info(f"Loaded {len(symbols)} symbols from {save_path}")
        return symbols
    else:
        symbols = extract_all_symbols_under_directory(repo_path)
        writer = PydanticJSONLinesWriter(str(save_path))
        writer.write_many(symbols)
        logger.info(f"Extracted {len(symbols)} symbols and saved to {save_path}")
        return symbols


def generate_synthetic_intents(
    symbols: list[Symbol], cache_path: Path, repo_name: str
) -> list[IntentForSymbol]:
    save_path = cache_path / repo_name / "synthetic_intents.jsonl"
    if save_path.exists():
        reader = PydanticJSONLinesReader(str(save_path), model=IntentForSymbol)
        intents = list(reader())
        logger.info(f"Loaded {len(intents)} intents from {save_path}")
        return intents
    else:
        intents = generate_intents_for_symbols(
            symbols=symbols,
            output_writer=PydanticJSONLinesWriter(str(save_path)),
            intent_generator=OpenAIIntentGenerator(),
            metadata=MetadataForPromptConstruction(repo_name=repo_name),
        )
        reader = PydanticJSONLinesReader(str(save_path), model=IntentForSymbol)
        intents = list(reader())
        logger.info(f"Generated {len(intents)} intents and saved to {save_path}")
        return intents


def embed_synthetic_intents(
    intents: list[IntentForSymbol],
    symbols: list[Symbol],
    cache_path: Path,
    repo_name: str,
    core_file_names: Optional[set[str]] = None,
) -> PydanticLancedbVectorDatabase[Symbol]:

    # First we create a cache to keep track of what has already been successfully embedded.
    progress_cache_path = cache_path / repo_name / "progress_cache.db"
    progress_cache_path.parent.mkdir(parents=True, exist_ok=True)

    batch_size = 100
    vector_store = PydanticLancedbVectorDatabase(
        str(cache_path / "lancedb"),
        Symbol,
        embedder=OpenAIEmbedder(batch_size=batch_size),
        table_name=f"{repo_name}",
    )
    completion_marker = (
        cache_path / repo_name / "embed_synthetic_intents.completion_marker"
    )

    core_file_names_path = cache_path / repo_name / "core_file_names.json"

    def can_reuse_vector_store() -> bool:
        # If the core file names have changed, we need to reindex.
        # We've never embedded this repo before.
        if not core_file_names_path.exists():
            return False

        last_used_core_file_names = json.loads(core_file_names_path.read_text())
        can_reuse = set(last_used_core_file_names) == core_file_names
        if not can_reuse:
            logger.info(f"Core file names have changed for {repo_name}, reindexing")
        return can_reuse

    if completion_marker.exists() and can_reuse_vector_store():
        logger.info(f"Skipping {repo_name} because already embedded")
        return vector_store
    else:
        logger.info(f"Embedding {repo_name}")
        core_file_names_path.write_text(json.dumps(list(core_file_names or {})))
        # Delete the progress cache.
        progress_cache_path.unlink(missing_ok=True)

    symbols_in_core_files: list[Symbol] = []
    for symbol in symbols:
        path_parts = symbol.filepath.split("/")
        repo_idx = path_parts.index(repo_name)
        file_name = "/".join(path_parts[repo_idx:])
        if core_file_names is None:
            symbols_in_core_files.append(symbol)
        else:
            if file_name in core_file_names:
                symbols_in_core_files.append(symbol)

    import_paths_for_symbols_in_core_files = {
        _.full_path for _ in symbols_in_core_files
    }

    # Now we grab all the intents for the symbols in the core files.
    intents_for_symbols_in_core_files: list[IntentForSymbol] = [
        intent
        for intent in intents
        if intent.symbol_full_path in import_paths_for_symbols_in_core_files
    ]

    if len(intents_for_symbols_in_core_files) == 0:
        logger.warning(
            "No intents found for symbols in files: {files_to_search}. "
            " Are you sure the core paths are correct?",
            files_to_search=import_paths_for_symbols_in_core_files,
        )
    else:
        logger.info(
            f"Found {len(intents_for_symbols_in_core_files)} intent for symbols in core files for {repo_name}"
        )

    # Now we embed the intents.
    progress_cache = SqliteKeyValueStore(f"sqlite:///{progress_cache_path}")

    symbol_full_path_to_symbol: dict[str, Symbol] = {
        _.full_path: _ for _ in symbols_in_core_files
    }

    # We do the embedding in batches.
    embeddables: list[Embeddable] = [
        Embeddable(
            key=intent.intent,
            payload=symbol_full_path_to_symbol[intent.symbol_full_path],
        )
        for intent in intents_for_symbols_in_core_files
        if progress_cache.get(intent.symbol_full_path) is None
    ]

    if not embeddables:
        logger.info(f"No unindexed embeddables found for {repo_name}")
        completion_marker.touch()
        return vector_store

    # We need to sort the embeddables here so that the first ones to be
    # indexed do not have any null fields. This is because we vector store
    # implementation will infer a schema from the first row. If we have any
    # null fields, it will infer that the entire column will be all null.
    # This is a bug in our vector store implementation.
    def rank_not_null_embeddables_first(embeddable: Embeddable) -> int:
        return sum(
            [
                embeddable.payload.docstring is not None,
                embeddable.payload.code is not None,
            ]
        )

    embeddables.sort(key=rank_not_null_embeddables_first, reverse=True)

    for i in tqdm(
        range(0, len(embeddables), batch_size),
        desc=f"Embedding intents for {repo_name}",
        unit="batches",
    ):
        batch = embeddables[i : i + batch_size]
        vector_store.insert(batch)
        for embeddable in batch:
            progress_cache.set(embeddable.payload.full_path, "done")

    # Do a quick sanity check. We sample 25 random intents and check that
    # the returned symbol has the same full path as the intent.
    sampled_intents = random.sample(intents_for_symbols_in_core_files, 25)
    for intent in sampled_intents:
        results = vector_store.search(intent.intent)
        assert len(results) > 0
        assert results[0].payload.full_path == intent.symbol_full_path

    completion_marker.touch()

    return vector_store


def render_human_readable_plan(plan: Plan) -> str:
    template = jinja2.Template(
        """# User Query
{{ plan.user_query }}

# Reasoning
{{ plan.reasoning }}

# Plan
{% for step in plan.steps %}
- {{ step.content }}
    {% if step.search_result.instrumentation and step.search_result.instrumentation.symbols_considered %}
    {% for symbol in step.search_result.instrumentation.symbols_considered -%}
    * {{ symbol.symbol.full_path }} (score: {{ "%.3f"|format(symbol.score) if symbol.score else "N/A" }})
    {% endfor -%}
    {% endif %}
{% endfor %}""",
        undefined=jinja2.StrictUndefined,
    )
    return template.render(plan=plan, strip_blocks=True, lstrip_blocks=True)


def build_retriever_for_repo(
    vector_store: ObjectVectorDatabase[Symbol],
) -> SymbolRetriever:
    retriever = OpenAiVectorSearchSymbolRetriever(vector_database=vector_store)
    return retriever


def get_starting_symbols(
    retriever: SymbolRetriever, user_query: str, num_starting_symbols: int
) -> list[Symbol]:
    # We have to over-retrieve because each symbol is embedded 5x with different intents.
    retrieved_symbols = retriever([user_query], n_results=num_starting_symbols * 5)
    seen_symbols: set[str] = set()
    starting_symbols: list[Symbol] = []
    for symbol in retrieved_symbols:
        if symbol.symbol.full_path in seen_symbols:
            continue
        seen_symbols.add(symbol.symbol.full_path)
        starting_symbols.append(symbol.symbol)
        if len(starting_symbols) == num_starting_symbols:
            break
    return starting_symbols


def get_repo_tree(repo_path: Path) -> str:
    _, tree, _ = ingest(str(repo_path), include_patterns=["*.py"])
    return tree


def make_ranker(
    ranker_type: Literal["diversity", "likert"]
) -> RankingFunction[PlanStep, GoalTest]:
    match ranker_type:
        case "diversity":
            return MostUniqueSymbolsRanker[GoalTest]()
        case "likert":
            return StepLevelLikertLlmJudge[GoalTest]()
        case _:
            assert_never(ranker_type)


def sort_nodes_with_diversity_ranker(
    nodes: list[Node[PlanStep, GoalTest]],
    ranker: MostUniqueSymbolsRanker[GoalTest],
) -> list[Node[PlanStep, GoalTest]]:
    return sorted(
        nodes,
        key=lambda node: ranker(node),
        reverse=True,
    )


def sort_nodes_with_likert_ranker(
    nodes: list[Node[PlanStep, GoalTest]],
    ranker: StepLevelLikertLlmJudge[GoalTest],
) -> list[Node[PlanStep, GoalTest]]:
    ulid_to_judge_response: dict[ULID, JudgeResponse] = {
        node.ulid: ranker.get_judge_response(node) for node in nodes
    }
    # Sort by solves_user_request, then no_unnecessary_steps, then average step scores
    sorted_ulids = sorted(
        ulid_to_judge_response.keys(),
        key=lambda ulid: (
            ulid_to_judge_response[ulid].solves_user_request,
            # Add average step score as final tiebreaker
            (
                np.mean(
                    [
                        step.achievable_with_symbols
                        for step in ulid_to_judge_response[ulid].step_judgements
                    ]
                )
                if ulid_to_judge_response[ulid].step_judgements
                else 0.0
            ),
        ),
        reverse=True,  # Sort in descending order
    )

    # Create a mapping from ULID to node for efficient lookup
    ulid_to_node: dict[ULID, Node[PlanStep, GoalTest]] = {
        node.ulid: node for node in nodes
    }

    # Return nodes in order of sorted UIDs
    return [ulid_to_node[ulid] for ulid in sorted_ulids]


def sort_nodes_for_plan_selection(
    nodes: list[Node[PlanStep, GoalTest]],
    ranker: RankingFunction[PlanStep, GoalTest],
) -> list[Node[PlanStep, GoalTest]]:

    match ranker:
        case MostUniqueSymbolsRanker():
            return sort_nodes_with_diversity_ranker(nodes, ranker)
        case StepLevelLikertLlmJudge():
            return sort_nodes_with_likert_ranker(nodes, ranker)
        case _:
            raise ValueError(f"Unknown ranker type: {type(ranker)}")


def run_plan_search_for_user_query(
    user_query: str,
    repo_path: Path,
    vector_store: ObjectVectorDatabase[Symbol],
    output_dir: Path,
    beam_width: int = 3,
    beam_depth: int = 20,
    budget: int = 160,
    ranker_type: Literal["diversity", "likert"] = "diversity",
) -> SearchResult[PlanStep, GoalTest]:

    retriever = build_retriever_for_repo(vector_store)

    starting_symbols = get_starting_symbols(retriever, user_query, 100)

    repo_tree = get_repo_tree(repo_path)

    code_search_tool = NoDuplicatesDirectIntentSearchTool(symbol_retriever=retriever)

    successor_fn = UnconstrainedXmlOutputSuccessorFunction(
        search_tool=code_search_tool,
        beam_width=beam_width,
        starting_symbols=starting_symbols,
        repo_tree=repo_tree,
    )

    check_is_goal_state_fn = AlwaysReturnsGoalTestFalse()

    initial_state = Node(
        plan=Plan(
            user_query=user_query,
            steps=[],
        ),
    )

    ranker = make_ranker(ranker_type)

    container = PriorityQueueSearchContainer[Node[PlanStep, GoalTest]](
        priority_function=ranker, max_heap=True
    )

    planner = PlanSearcher[PlanStep, GoalTest](
        initial_state=initial_state,
        check_is_goal_state_fn=check_is_goal_state_fn,
        successor_fn=successor_fn,
        container_factory=lambda: container,
        node_budget=budget,
        beam_width=beam_width,
        beam_depth=beam_depth,
    )

    logger.info(f'Starting processing for user query: "{user_query}"')

    search_result = planner.run()

    with open(output_dir / "plan_search_result.json", "w") as f:
        f.write(search_result.model_dump_json())

    search_result.nodes = sort_nodes_for_plan_selection(search_result.nodes, ranker)

    # Write the top 10 nodes to a markdown file.
    for idx, node in enumerate(search_result.nodes[:10]):
        with open(output_dir / f"rank_{idx}_score_{ranker(node)}_plan.md", "w") as f:
            f.write(render_human_readable_plan(node.plan))

    return search_result


class PlanSearchSettings(BaseSettings):
    """
    Settings for running plan search.

    Parameters
    -----------
    user_query: str
        The user query to search a plan for.
    repo_path: Path
        The path to the repository.
    working_dir: Path
        The directory that will be used to save outputs and intermediate computations.
    beam_width: int
        The branching factor for the search.
    beam_depth: int
        The depth of the search.
    budget: int
        The maximum number of nodes to expand.
    ranker_type: Literal["diversity", "likert"]
        The type of ranker to use. The "diversity" ranker is faster, while "likert"
        may be more accurate, since it uses an LLM to judge the plan quality.
    core_paths: Optional[list[Path]]
        The most important files in the repository. If provided, the search will index
        only the symbols in these files. This can be helpful to narrow down the search space.
    """

    user_query: str
    repo_path: Path
    working_dir: Path = Path("plan_search_outputs")
    beam_width: int = Field(default=3)
    beam_depth: int = Field(default=20)
    budget: int = Field(default=20)
    ranker_type: Literal["diversity", "likert"] = Field(default="diversity")
    core_paths: Optional[list[Path]] = Field(default=None)

    model_config = SettingsConfigDict(cli_parse_args=True)

    @cached_property
    def core_file_names(self) -> Optional[set[str]]:
        if self.core_paths is None:
            return None
        # Glob all the files in the core paths.
        core_file_names: set[str] = set()
        for core_path in self.core_paths:
            core_file_names.update(
                str(_.relative_to(settings.repo_path.parent))
                for _ in core_path.glob("**/*.py")
            )
        logger.info(f"Expanded core paths to {len(core_file_names)} files")
        return core_file_names


def plan_search(
    settings: PlanSearchSettings,
) -> None:
    cache_path = settings.working_dir / "cache"
    cache_path.mkdir(parents=True, exist_ok=True)

    symbols = mine_symbols(repo_path=settings.repo_path, cache_path=cache_path)
    intents = generate_synthetic_intents(
        symbols=symbols, cache_path=cache_path, repo_name=settings.repo_path.name
    )
    vector_store = embed_synthetic_intents(
        intents=intents,
        symbols=symbols,
        cache_path=cache_path,
        repo_name=settings.repo_path.name,
        core_file_names=settings.core_file_names,
    )

    output_dir = (
        settings.working_dir
        / settings.repo_path.name
        / f"{dt.datetime.now().isoformat()}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    run_plan_search_for_user_query(
        user_query=settings.user_query,
        repo_path=settings.repo_path,
        vector_store=vector_store,
        output_dir=output_dir,
        beam_width=settings.beam_width,
        beam_depth=settings.beam_depth,
        budget=settings.budget,
        ranker_type=settings.ranker_type,
    )


if __name__ == "__main__":
    # Will automatically parse command line arguments.
    settings = PlanSearchSettings()  # type: ignore

    plan_search(settings)
