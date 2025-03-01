import datetime as dt

from ulid import ULID

from mutagrep.plan_search.components import GoalTest, PlanStep
from mutagrep.plan_search.domain_models import CodeSearchToolOutput, Node, Plan
from mutagrep.plan_search.generic_search import PlanSearchForProblemOutput
from mutagrep.plan_search.lca_benchmark import (
    LongCodeArenaMetricSinglePlan,
    LongCodeArenaRecord,
    load_longcode_arena_records,
    rank_best_plans_for_record,
    score_plan_for_record,
)

PlanSearchLcaOutputT = PlanSearchForProblemOutput[
    PlanStep,
    GoalTest,
    LongCodeArenaMetricSinglePlan,
    LongCodeArenaRecord,
]


def test_score_plan_for_record_all_correct():
    records = load_longcode_arena_records()
    record = records[0]

    plans = [
        PlanStep(
            index=i,
            content="",
            search_result=CodeSearchToolOutput(
                symbol_name=f"module_foo.module_bar.{api}",
                satisfies_intention=True,
                justification="",
            ),
        )
        for i, api in enumerate(record.unique_apis)
    ]

    score = score_plan_for_record(record, plans)
    assert score.precision == 1.0
    assert score.recall == 1.0
    assert score.f1 == 1.0


def test_score_plan_for_record_all_incorrect():
    records = load_longcode_arena_records()
    record = records[0]

    plans = [
        PlanStep(
            index=i,
            content="",
            search_result=CodeSearchToolOutput(
                symbol_name="foo.bar.baz",
                satisfies_intention=False,
                justification="",
            ),
        )
        for i in range(len(record.unique_apis))
    ]

    score = score_plan_for_record(record, plans)
    assert score.precision == 0.0
    assert score.recall == 0.0
    assert score.f1 == 0.0


def test_ranking_best_plans_for_record():
    plan_a_id = ULID.from_datetime(dt.datetime(2024, 1, 1))
    plan_a = (
        LongCodeArenaMetricSinglePlan(
            precision=1.0,
            recall=0.5,
            f1=0.5,
            satisfiable_precision=0.5,
            satisfiable_recall=0.5,
            satisfiable_f1=0.5,
            hit_symbols=[],
            missed_symbols=[],
        ),
        Node(
            plan=Plan(steps=[], user_query=""),
            ulid=plan_a_id,
        ),
    )

    plan_b_id = ULID.from_datetime(dt.datetime(2024, 1, 2))
    plan_b = (
        LongCodeArenaMetricSinglePlan(
            precision=0.0,
            recall=1.0,
            f1=0.0,
            satisfiable_precision=0.0,
            satisfiable_recall=0.0,
            satisfiable_f1=0.0,
            hit_symbols=[],
            missed_symbols=[],
        ),
        Node(
            plan=Plan(steps=[], user_query=""),
            ulid=plan_b_id,
        ),
    )

    plan_c_id = ULID.from_datetime(dt.datetime(2024, 1, 3))
    plan_c = (
        LongCodeArenaMetricSinglePlan(
            precision=0.5,
            recall=0.5,
            f1=1.0,
            satisfiable_precision=0.5,
            satisfiable_recall=0.5,
            satisfiable_f1=0.5,
            hit_symbols=[],
            missed_symbols=[],
        ),
        Node(
            plan=Plan(steps=[], user_query=""),
            ulid=plan_c_id,
        ),
    )

    ranked_plans = rank_best_plans_for_record([plan_a, plan_b, plan_c])

    assert ranked_plans.best_f1[1].ulid == plan_c_id
    assert ranked_plans.best_precision[1].ulid == plan_a_id
    assert ranked_plans.best_recall[1].ulid == plan_b_id
