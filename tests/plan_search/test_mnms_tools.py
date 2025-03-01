from mutagrep.plan_search.mnms_benchmark import (
    MnmsPlanStep,
    MnmsRecord,
    score_plan_for_record,
)


def test_scoring_perfect_plan():
    record = MnmsRecord(
        id=0,
        user_request="I need to analyze the sentiment of a letter sent to me, then create an image inspired by the sentiment.",
        plan=[
            MnmsPlanStep(id=0, name="text_classification", args=None),
            MnmsPlanStep(id=1, name="image_generation", args=None),
        ],
        code="",
        alt_plans=[],
    )

    score = score_plan_for_record(record, record.plan)
    assert score.precision == 1
    assert score.recall == 1
    assert score.f1 == 1
    assert score.length_error == 0


def test_scoring_no_recall():
    record = MnmsRecord(
        id=0,
        user_request="I need to analyze the sentiment of a letter sent to me, then create an image inspired by the sentiment.",
        plan=[
            MnmsPlanStep(id=0, name="text_classification", args=None),
            MnmsPlanStep(id=1, name="image_generation", args=None),
        ],
        code="",
        alt_plans=[],
    )

    candidate_plan = [
        MnmsPlanStep(id=0, name="text_classification", args=None),
    ]

    score = score_plan_for_record(record, candidate_plan)
    assert score.precision == 1
    assert score.recall == 0.5
    assert score.length_error == 1


def test_scoring_empty_plan():
    record = MnmsRecord(
        id=0,
        user_request="I need to analyze the sentiment of a letter sent to me, then create an image inspired by the sentiment.",
        plan=[
            MnmsPlanStep(id=0, name="text_classification", args=None),
            MnmsPlanStep(id=1, name="image_generation", args=None),
        ],
        code="",
        alt_plans=[],
    )

    candidate_plan = []

    score = score_plan_for_record(record, candidate_plan)
    assert score.precision == 0
    assert score.recall == 0
    assert score.f1 == 0
    assert score.length_error == 2


def test_scoring_with_alt_plans():
    """We should return the best score between the candidate plan and ground truth + alt plans."""
    record = MnmsRecord(
        id=0,
        user_request="I need to analyze the sentiment of a letter sent to me, then create an image inspired by the sentiment.",
        plan=[
            MnmsPlanStep(id=0, name="text_classification", args=None),
            MnmsPlanStep(id=1, name="image_generation", args=None),
        ],
        code="",
        alt_plans=[
            [
                MnmsPlanStep(id=0, name="question_answering", args=None),
                MnmsPlanStep(id=1, name="image_generation", args=None),
            ],
        ],
    )

    candidate_plan = [
        MnmsPlanStep(id=0, name="question_answering", args=None),
    ]

    score = score_plan_for_record(record, candidate_plan)
    assert score.precision == 1
    assert score.recall == 0.5
