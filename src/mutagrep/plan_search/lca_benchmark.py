from typing import Any, Generic, Optional, Protocol, Sequence, cast

import numpy as np
import pandas as pd
from datasets import DatasetDict, load_dataset
from pydantic import BaseModel

from mutagrep.longcodearena.types import LongCodeArenaRecord
from mutagrep.plan_search.domain_models import (
    CodeSearchToolOutput,
    GoalTestT,
    Node,
    PlanStepT,
)


def load_longcode_arena_records() -> list[LongCodeArenaRecord]:
    ds = cast(
        DatasetDict,
        load_dataset(
            "JetBrains-Research/lca-library-based-code-generation", split="test"
        ),
    )

    records = [LongCodeArenaRecord.model_validate(_) for _ in ds]

    return records


class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LongCodeArenaMetricSinglePlan(BaseModel):
    """
    The results of scoring a candidate plan for an LongCodeArenaRecord.
    """

    precision: float
    recall: float
    f1: float
    satisfiable_precision: float
    satisfiable_recall: float
    satisfiable_f1: float
    hit_symbols: list[str]
    missed_symbols: list[str]
    token_usage: Optional[TokenUsage] = None


class LongCodeArenaMetricBestPlan(LongCodeArenaMetricSinglePlan):
    """
    The results of scoring the best plan for an LongCodeArenaRecord.
    """

    nodes_expanded_to_reach: int


class PlanStep(Protocol):
    search_result: CodeSearchToolOutput


def calculate_token_usage(
    candidate_plan: Sequence[PlanStep],
) -> Optional[TokenUsage]:

    if not candidate_plan:
        return None

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    for step in candidate_plan:
        # If any steps are missing instrumentation, return None.
        # This is to ensure that runs which were missing data aren't
        # counted as having used a small amount of tokens.
        if step.search_result.instrumentation is None:
            return None

        total_prompt_tokens += step.search_result.instrumentation.prompt_tokens
        total_completion_tokens += step.search_result.instrumentation.completion_tokens
        total_tokens += step.search_result.instrumentation.total_tokens

    return TokenUsage(
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        total_tokens=total_tokens,
    )


def score_plan_for_record(
    record: LongCodeArenaRecord, candidate_plan: Sequence[PlanStep]
) -> LongCodeArenaMetricSinglePlan:
    # Extract unique APIs from the record
    unique_apis = set(record.unique_apis)

    # Extract symbol names from the candidate plan
    symbol_names = {
        step.search_result.symbol_name
        for step in candidate_plan
        if step.search_result.symbol_name
    }

    # Calculate true positives
    true_positives = sum(
        1 for api in unique_apis if any(api in symbol for symbol in symbol_names)
    )

    hit_symbols = [
        symbol for symbol in symbol_names if any(api in symbol for api in unique_apis)
    ]

    missed_symbols = [
        api for api in unique_apis if not any(api in symbol for symbol in symbol_names)
    ]

    # Calculate precision
    precision = true_positives / len(symbol_names) if symbol_names else 0.0

    # Calculate recall
    recall = true_positives / len(unique_apis) if unique_apis else 0.0

    # Calculate F1 score
    f1 = (
        (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    )

    # Extract satisfiable symbol names
    satisfiable_symbol_names = {
        step.search_result.symbol_name
        for step in candidate_plan
        if step.search_result.symbol_name and step.search_result.satisfies_intention
    }

    # Calculate true positives for satisfiable symbols
    satisfiable_true_positives = sum(
        1
        for api in unique_apis
        if any(api in symbol for symbol in satisfiable_symbol_names)
    )

    # Calculate satisfiable precision
    satisfiable_precision = (
        satisfiable_true_positives / len(satisfiable_symbol_names)
        if satisfiable_symbol_names
        else 0.0
    )

    # Calculate satisfiable recall
    satisfiable_recall = (
        satisfiable_true_positives / len(unique_apis) if unique_apis else 0.0
    )

    # Calculate satisfiable F1 score
    satisfiable_f1 = (
        (2 * satisfiable_precision * satisfiable_recall)
        / (satisfiable_precision + satisfiable_recall)
        if (satisfiable_precision + satisfiable_recall)
        else 0.0
    )

    return LongCodeArenaMetricSinglePlan(
        precision=precision,
        recall=recall,
        f1=f1,
        satisfiable_precision=satisfiable_precision,
        satisfiable_recall=satisfiable_recall,
        satisfiable_f1=satisfiable_f1,
        token_usage=calculate_token_usage(candidate_plan),
        hit_symbols=hit_symbols,
        missed_symbols=missed_symbols,
    )


class BestMetricResults(BaseModel, Generic[PlanStepT, GoalTestT]):
    """Results for the best performing plans across different metrics."""

    best_f1: tuple[LongCodeArenaMetricBestPlan, Node[PlanStepT, GoalTestT]]
    best_precision: tuple[LongCodeArenaMetricBestPlan, Node[PlanStepT, GoalTestT]]
    best_recall: tuple[LongCodeArenaMetricBestPlan, Node[PlanStepT, GoalTestT]]

    @staticmethod
    def create_human_readable_row(
        pair: tuple[LongCodeArenaMetricBestPlan, Node[PlanStepT, GoalTestT]]
    ) -> dict[str, Any]:
        metric, node = pair
        return {
            "metric": metric.model_dump(),
            "steps": len(node.plan.steps),
            "ulid": str(node.ulid),
        }

    def to_dataframe(self) -> pd.DataFrame:
        rows = [
            self.create_human_readable_row(self.best_f1),
            self.create_human_readable_row(self.best_precision),
            self.create_human_readable_row(self.best_recall),
        ]
        return pd.DataFrame(rows)


def rank_best_plans_for_record(
    plan_search_outputs: Sequence[tuple[LongCodeArenaMetricSinglePlan, Node]],
) -> BestMetricResults:
    # Sort nodes by their ULID timestamp
    sorted_nodes = sorted(plan_search_outputs, key=lambda x: x[1].ulid.timestamp)

    # Print the max f1, precision, and recall
    max_f1 = max(scores_for_metric.f1 for scores_for_metric, _ in sorted_nodes)
    max_precision = max(
        scores_for_metric.precision for scores_for_metric, _ in sorted_nodes
    )
    max_recall = max(scores_for_metric.recall for scores_for_metric, _ in sorted_nodes)
    print(f"Max F1: {max_f1}, Max Precision: {max_precision}, Max Recall: {max_recall}")

    # Initialize variables with the first node's metrics
    first_score, first_node = sorted_nodes[0]
    best_scores = {
        "f1": (first_score, first_node, 0),
        "precision": (first_score, first_node, 0),
        "recall": (first_score, first_node, 0),
    }

    # Iterate over sorted nodes to find the best scores for each metric
    for index, (scores_for_metric, node) in enumerate(sorted_nodes, start=1):
        if scores_for_metric.f1 > best_scores["f1"][0].f1:
            best_scores["f1"] = (scores_for_metric, node, index)
        if scores_for_metric.precision > best_scores["precision"][0].precision:
            best_scores["precision"] = (scores_for_metric, node, index)
        if scores_for_metric.recall > best_scores["recall"][0].recall:
            best_scores["recall"] = (scores_for_metric, node, index)

    # Convert results to BestMetricResults format
    results = {}
    for metric, (score, node, index) in best_scores.items():
        best_plan_score = LongCodeArenaMetricBestPlan(
            precision=score.precision,
            recall=score.recall,
            f1=score.f1,
            satisfiable_precision=score.satisfiable_precision,
            satisfiable_recall=score.satisfiable_recall,
            satisfiable_f1=score.satisfiable_f1,
            nodes_expanded_to_reach=index,
            hit_symbols=score.hit_symbols,
            missed_symbols=score.missed_symbols,
            token_usage=score.token_usage,
        )
        results[f"best_{metric}"] = (best_plan_score, node)

    return BestMetricResults(**results)


def score_plan_for_record_multisymbol(
    record: LongCodeArenaRecord, candidate_plan: Sequence[PlanStep]
) -> LongCodeArenaMetricSinglePlan:
    # Extract unique APIs from the record
    unique_apis = set(record.unique_apis)

    # Extract symbol names from the candidate plan
    symbol_names: set[str] = set()
    for step in candidate_plan:
        if step.search_result.instrumentation is None:
            raise ValueError(
                "Multi-symbol scoring relies on using the instrumentation object "
                "to get the symbols considered. But it was none."
            )
        symbol_names.update(
            retrieved_symbol.symbol.full_path
            for retrieved_symbol in step.search_result.instrumentation.symbols_considered
        )

    # Calculate true positives
    true_positives = sum(
        1 for api in unique_apis if any(api in symbol for symbol in symbol_names)
    )

    hit_symbols = [
        symbol for symbol in symbol_names if any(api in symbol for api in unique_apis)
    ]

    missed_symbols = [
        api for api in unique_apis if not any(api in symbol for symbol in symbol_names)
    ]

    # Calculate precision
    precision = true_positives / len(symbol_names) if symbol_names else 0.0

    # Calculate recall
    recall = true_positives / len(unique_apis) if unique_apis else 0.0

    # Calculate F1 score
    f1 = (
        (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    )

    # Extract satisfiable symbol names
    satisfiable_symbol_names = {
        step.search_result.symbol_name
        for step in candidate_plan
        if step.search_result.symbol_name and step.search_result.satisfies_intention
    }

    # Calculate true positives for satisfiable symbols
    satisfiable_true_positives = sum(
        1
        for api in unique_apis
        if any(api in symbol for symbol in satisfiable_symbol_names)
    )

    # Calculate satisfiable precision
    satisfiable_precision = (
        satisfiable_true_positives / len(satisfiable_symbol_names)
        if satisfiable_symbol_names
        else 0.0
    )

    # Calculate satisfiable recall
    satisfiable_recall = (
        satisfiable_true_positives / len(unique_apis) if unique_apis else 0.0
    )

    # Calculate satisfiable F1 score
    satisfiable_f1 = (
        (2 * satisfiable_precision * satisfiable_recall)
        / (satisfiable_precision + satisfiable_recall)
        if (satisfiable_precision + satisfiable_recall)
        else 0.0
    )

    return LongCodeArenaMetricSinglePlan(
        precision=precision,
        recall=recall,
        f1=f1,
        satisfiable_precision=satisfiable_precision,
        satisfiable_recall=satisfiable_recall,
        satisfiable_f1=satisfiable_f1,
        token_usage=calculate_token_usage(candidate_plan),
        hit_symbols=hit_symbols,
        missed_symbols=missed_symbols,
    )


def compute_aggregate_metrics_from_best_plans(
    best_plans_per_record: Sequence[BestMetricResults],
) -> pd.DataFrame:
    metrics = {
        "f1": [r.best_f1[0] for r in best_plans_per_record],
        "precision": [r.best_precision[0] for r in best_plans_per_record],
        "recall": [r.best_recall[0] for r in best_plans_per_record],
    }

    means = {
        metric_type: {
            "precision": np.mean([m.precision for m in ms]),
            "recall": np.mean([m.recall for m in ms]),
            "f1": np.mean([m.f1 for m in ms]),
            "satisfiable_precision": np.mean([m.satisfiable_precision for m in ms]),
            "satisfiable_recall": np.mean([m.satisfiable_recall for m in ms]),
            "satisfiable_f1": np.mean([m.satisfiable_f1 for m in ms]),
            "nodes_expanded_to_reach": np.mean([m.nodes_expanded_to_reach for m in ms]),
        }
        for metric_type, ms in metrics.items()
    }

    return pd.DataFrame(means).T.round(3)
