import ast
import json
import random
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import (Any, Iterable, Iterator, Literal, Optional, Sequence,
                    TypedDict, cast)

import numpy as np
import pandas as pd
import rich
from datasets import Dataset, load_dataset
from pydantic import BaseModel
from typing_extensions import Self

from mutagrep.plan_search.domain_models import Node


class MnmsRecordRaw(TypedDict):
    id: int
    user_request: str
    plan_str: str
    code_str: str
    alt_plans_str: Optional[str]


class MnmsPlanStep(BaseModel):
    id: int
    name: str
    args: Optional[dict[str, Any]]


class MnmsRecord(BaseModel):
    id: int
    user_request: str
    plan: list[MnmsPlanStep]
    code: str
    alt_plans: list[list[MnmsPlanStep]]

    @classmethod
    def from_raw(cls, raw: MnmsRecordRaw) -> Self:
        user_request = raw["user_request"]
        plan = ast.literal_eval(raw["plan_str"])
        for step in plan:
            step["name"] = step["name"].replace(" ", "_")
        code = raw["code_str"]
        if raw["alt_plans_str"] is None:
            alt_plans = None
        else:
            alt_plans = ast.literal_eval(raw["alt_plans_str"])
            for alt_plan in alt_plans:
                for step in alt_plan:
                    step["name"] = step["name"].replace(" ", "_")

        payload = {
            "user_request": user_request,
            "plan": plan,
            "code": code,
            "alt_plans": alt_plans,
            "id": raw["id"],
        }
        return cls.model_validate(payload)

    @classmethod
    def sequence_from_ds(cls, ds: Iterable[MnmsRecordRaw]) -> Sequence[Self]:
        return [cls.from_raw(record) for record in ds]


class MnmsMetric(BaseModel):
    """
    The results of scoring a candidate plan for an MnmsRecord.

    precision: Of the tools called in the plan, how many were in the ground truth plan?
    recall: Of the tools in the ground truth plan, how many were in the candidate plan?
    f1: The harmonic mean of precision and recall.
    length_error: The absolute difference between the length of the candidate plan and the ground truth plan.
    """

    precision: float
    recall: float
    f1: float
    length_error: int


class MnmsMetricsForBestPlan(MnmsMetric):
    nodes_expanded_to_reach: int


def normalize_tool_names(tool_name: str) -> str:
    """
    The tool name might have dots in it, like foo.bar.text_classification.
    We only want to keep the last part.
    """
    return tool_name.split(".")[-1]


def score_candidate_plan_against_reference_plan(
    reference_plan: list[MnmsPlanStep], candidate_plan: list[MnmsPlanStep]
) -> MnmsMetric:
    candidate_plan_tools = set(
        normalize_tool_names(step.name) for step in candidate_plan
    )
    reference_plan_tools = set(
        normalize_tool_names(step.name) for step in reference_plan
    )

    true_positives = len(candidate_plan_tools.intersection(reference_plan_tools))
    false_positives = len(candidate_plan_tools - reference_plan_tools)
    false_negatives = len(reference_plan_tools - candidate_plan_tools)

    if true_positives + false_positives > 0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = 0
    if true_positives + false_negatives > 0:
        recall = true_positives / (true_positives + false_negatives)
    else:
        recall = 0
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0
    length_error = abs(len(reference_plan) - len(candidate_plan))

    return MnmsMetric(
        precision=precision, recall=recall, f1=f1, length_error=length_error
    )


def score_plan_for_record(
    record: MnmsRecord, candidate_plan: list[MnmsPlanStep]
) -> MnmsMetric:

    score_against_ground_truth = score_candidate_plan_against_reference_plan(
        record.plan, candidate_plan
    )

    score_against_alt_plans: list[MnmsMetric] = []
    for alt_plan in record.alt_plans:
        score_against_alt_plans.append(
            score_candidate_plan_against_reference_plan(alt_plan, candidate_plan)
        )

    scores_against_all_plans = [score_against_ground_truth] + score_against_alt_plans
    best_score = max(scores_against_all_plans, key=lambda x: x.f1)
    return best_score


def stratify_by_step_count(
    records: Sequence[MnmsRecord],
) -> dict[int, list[MnmsRecord]]:
    stratifications = defaultdict(list)
    for record in records:
        # Pick only records for which the plan and all alt_plans have the same length
        plan_length = len(record.plan)
        alt_plans_lengths = {len(alt_plan) for alt_plan in record.alt_plans}
        if len(alt_plans_lengths) == 1 and plan_length in alt_plans_lengths:
            stratifications[plan_length].append(record)
    return stratifications


# TODO: Mark this as deprecated.
def load_dev_set_from_fs() -> list[MnmsRecord]:
    with open("mnms_dev_set.json", "r") as f:
        return [MnmsRecord.model_validate_json(line) for line in f]


def load_dev_set() -> list[MnmsRecord]:
    ds = load_dataset(
        "zixianma/mnms",
        split="test_human_verified_filtered",
        revision="da313260161c982eb2004bb15761d7aa2e03eb4f",
    )
    records = MnmsRecord.sequence_from_ds(cast(Iterable[MnmsRecordRaw], ds))
    stratifications = stratify_by_step_count(records)
    for step_count, records in stratifications.items():
        print(f"Step Count: {step_count}")
        print(f"Number of Records: {len(records)}")
        print()

    # Pick 5 from each step count. Don't randomly sample, just take the first 5.
    dev_set: list[MnmsRecord] = []
    for step_count, records in stratifications.items():
        dev_set.extend(records[:5])

    return dev_set


def filter_unique_tools(
    records: Sequence[MnmsRecord], length: Literal[1, 2, 3] = 3
) -> list[MnmsRecord]:
    match length:
        case 1:
            raise ValueError("Cannot filter unique tools for 1-step plans")
        case 2 | 3:
            matched_records = []
            for record in records:
                if len(record.plan) == length:
                    matched_records.append(record)
            # Now check the number of unique tools in the plan
            # is equal to the length of the plan.
            filtered_records = []
            for record in matched_records:
                if len(set(step.name for step in record.plan)) == length:
                    filtered_records.append(record)
            return filtered_records
        case _:
            raise ValueError(f"Invalid number of steps: {length}")


class MnmDevSet(Sequence[MnmsRecord]):
    def __init__(self, num_steps: Literal[1, 2, 3, "all"]):
        match num_steps:
            case 1 | 2 | 3:
                self.dev_set = [
                    record for record in load_dev_set() if len(record.plan) == num_steps
                ]
            case "all":
                self.dev_set = load_dev_set()
            case _:
                raise ValueError(f"Invalid number of steps: {num_steps}")
        self.num_steps = num_steps

    def __iter__(self) -> Iterator[MnmsRecord]:
        return iter(self.dev_set)

    def __getitem__(self, idx: int) -> MnmsRecord:
        return self.dev_set[idx]

    def __len__(self) -> int:
        return len(self.dev_set)


class MnmSplitChoices(Enum):
    # 113 problems that codenav did not get perfect recall on
    CODENAV_HARD_SLICE = "codenav_hard_slice"
    # the full set of 200 mnms problems codenav reported on in the paper
    CODENAV_EVAL_SLICE = "codenav_eval_slice"


class MnmsDataset(Sequence[MnmsRecord]):
    def __init__(self, split: MnmSplitChoices = MnmSplitChoices.CODENAV_HARD_SLICE):
        self.split = split

        match split:
            case MnmSplitChoices.CODENAV_HARD_SLICE:
                with open(Path(__file__).parent / "mnms_hard_slice_ids.json", "r") as f:
                    self.ids_to_keep = set(json.load(f))
            case MnmSplitChoices.CODENAV_EVAL_SLICE:
                with open(
                    Path(__file__).parent / "codenav_mnms_eval_slice_ids.json", "r"
                ) as f:
                    self.ids_to_keep = set(json.load(f))
            case _:
                raise ValueError(f"Invalid split choice: {split}")

        ds = load_dataset(
            "zixianma/mnms",
            split="test_human_verified_filtered",
            revision="da313260161c982eb2004bb15761d7aa2e03eb4f",
        )
        records = MnmsRecord.sequence_from_ds(cast(Iterable[MnmsRecordRaw], ds))
        self.records = [record for record in records if record.id in self.ids_to_keep]

    def __iter__(self) -> Iterator[MnmsRecord]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> MnmsRecord:
        return self.records[idx]

    def get_by_mnms_id(self, mnms_id: int) -> MnmsRecord:
        return next(record for record in self.records if record.id == mnms_id)


class BestMetricResults(BaseModel):
    best_f1: tuple[MnmsMetricsForBestPlan, Node]
    best_precision: tuple[MnmsMetricsForBestPlan, Node]
    best_recall: tuple[MnmsMetricsForBestPlan, Node]

    @staticmethod
    def create_human_readable_row(
        pair: tuple[MnmsMetricsForBestPlan, Node]
    ) -> dict[str, Any]:
        metric, node = pair
        return {
            **metric.model_dump(),
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
    plan_search_outputs: Sequence[tuple[MnmsMetric, Node]],
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
        best_plan_score = MnmsMetricsForBestPlan(
            precision=score.precision,
            recall=score.recall,
            f1=score.f1,
            length_error=score.length_error,
            nodes_expanded_to_reach=index,
        )
        results[f"best_{metric}"] = (best_plan_score, node)

    return BestMetricResults(**results)


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
            "length_error": np.mean([m.length_error for m in ms]),
            "nodes_expanded_to_reach": np.mean([m.nodes_expanded_to_reach for m in ms]),
        }
        for metric_type, ms in metrics.items()
    }

    return pd.DataFrame(means).T.round(3)
