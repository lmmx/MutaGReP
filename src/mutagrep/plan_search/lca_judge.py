import os
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from typing import Literal

import jinja2
from openai import OpenAI
from openai.types.chat.chat_completion import Choice
from pydantic import BaseModel
from typing_extensions import Self

from mutagrep.coderec.v3.symbol_mining import Symbol
from mutagrep.plan_search.components import PlanStep
from mutagrep.plan_search.lca_benchmark import LongCodeArenaRecord

JUDGE_PLAN_VS_PLAN_TEMPLATE = jinja2.Template(
    """You are choosing which of two step-by-step plans to accomplish a user query is better.
    Each plan is a list of steps.
    Alongside each step is a list of fully-qualified symbol names (e.g. functions, classes, method) that the plan has marked for use in that step.
    You will also be provided gold-standard reference code that accomplishes the user query.
    Compare each of the plans to the reference code, and choose the plan that aligns more closely with the reference code and user query.

    Produce your output in the following format:
    ```xml
    <judgement>
    <explanation>
    Your chain of thought and reasoning for which plan is better.
    </explanation>
    <winner>Put either "A" or "B" here.</winner>
    </judgement>
    ```

    Example 1:
    ```xml
    <judgement>
    <explanation>
    Plan A is better because ...
    </explanation>
    <winner>A</winner>
    </judgement>
    ```

    Example 2:
    ```xml
    <judgement>
    <explanation>
    Plan B is better because ...
    </explanation>
    <winner>B</winner>
    </judgement>
    ```

    Do not include ANY other text in your output. Stick exactly to the required format.
    These are the only valid values for the <winner> field: "A" or "B".

    # User query
    {{ user_query }}

    # Reference code
    ```python
    {{ reference_code }}
    ```

    # Plan A
    {% for step in plan_a %}
    ## Step {{ step.index }}
    - {{ step.content }}
    ### Symbols
    {% for symbol in step.search_result.instrumentation.symbols_considered %}
    - {{ symbol.symbol.full_path }}
    {% endfor %}
    {% endfor %}

    # Plan B
    {% for step in plan_b %}
    ## Step {{ step.index }}
    - {{ step.content }}
    ### Symbols
    {% for symbol in step.search_result.instrumentation.symbols_considered %}
    - {{ symbol.symbol.full_path }}
    {% endfor %}
    {% endfor %}
    """,
    undefined=jinja2.StrictUndefined,
)


class AggregateJudgement(BaseModel):
    wins_by_A: int
    wins_by_B: int
    win_rate_A: float
    win_rate_B: float
    total_judgements: int


class PlanVsPlanJudgeRound(BaseModel):
    winner: Literal["A", "B"]
    explanation: str


class Judgement(BaseModel):
    winner: Literal["A", "B"]
    win_rate: float
    counts: dict[Literal["A", "B"], int]
    round_a_vs_b: list[PlanVsPlanJudgeRound]
    round_b_vs_a: list[PlanVsPlanJudgeRound]

    @classmethod
    def aggregate(cls, judgements: list[Self]) -> AggregateJudgement:
        win_counts: dict[Literal["A", "B"], int] = {"A": 0, "B": 0}
        for judgement in judgements:
            win_counts[judgement.winner] += 1

        total = sum(win_counts.values())

        return AggregateJudgement(
            wins_by_A=win_counts["A"],
            wins_by_B=win_counts["B"],
            win_rate_A=win_counts["A"] / total,
            win_rate_B=win_counts["B"] / total,
            total_judgements=total,
        )


class PlanVsPlanJudge:
    def __init__(self, record: LongCodeArenaRecord, num_judgements: int = 3):
        self.record = record
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.num_judgements = num_judgements

    def parse_response(self, response: Choice) -> PlanVsPlanJudgeRound:
        assert response.message.content is not None
        # Remove markdown code block if present
        content = response.message.content
        lines = content.splitlines()
        if lines and "```" in lines[0]:
            lines = lines[1:]
        if lines and "```" in lines[-1]:
            lines = lines[:-1]

        response.message.content = "\n".join(lines)

        root = ET.fromstring(response.message.content)
        assert root.tag == "judgement"
        assert (explanation := root.find("explanation")) is not None
        assert (winner := root.find("winner")) is not None

        assert explanation.text is not None
        assert winner.text in ["A", "B"]

        match winner.text:
            case "A":
                return PlanVsPlanJudgeRound(winner="A", explanation=explanation.text)
            case "B":
                return PlanVsPlanJudgeRound(winner="B", explanation=explanation.text)
            case _:
                raise ValueError(f"Invalid winner: {winner.text}")

    def judge_plan_vs_plan(
        self,
        plan_a: Sequence[PlanStep],
        plan_b: Sequence[PlanStep],
    ) -> list[PlanVsPlanJudgeRound]:
        prompt = JUDGE_PLAN_VS_PLAN_TEMPLATE.render(
            user_query=self.record.instruction,
            reference_code=self.record.clean_reference,
            plan_a=plan_a,
            plan_b=plan_b,
        )
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            n=self.num_judgements,
        )

        return [self.parse_response(choice) for choice in response.choices]

    def __call__(
        self,
        plan_a: Sequence[PlanStep],
        plan_b: Sequence[PlanStep],
    ) -> Judgement:
        # To account for positional bias, we do two rounds of judging.
        # In the first round, we judge plan A vs plan B.
        # In the second round, we judge plan B vs plan A.
        round_1 = self.judge_plan_vs_plan(plan_a, plan_b)
        round_2 = self.judge_plan_vs_plan(plan_b, plan_a)

        win_counts: dict[Literal["A", "B"], int] = {"A": 0, "B": 0}
        for judgement in round_1:
            win_counts[judgement.winner] += 1

        # In round 2, the plans are switched around, so a vote for plan A
        # is actually a vote for plan B and vice versa.
        for judgement in round_2:
            match judgement.winner:
                case "A":
                    win_counts["B"] += 1
                case "B":
                    win_counts["A"] += 1

        win_rate_a = win_counts["A"] / self.num_judgements
        win_rate_b = win_counts["B"] / self.num_judgements

        winner = "A" if win_rate_a > win_rate_b else "B"

        return Judgement(
            winner=winner,
            win_rate=win_rate_a if winner == "A" else win_rate_b,
            counts=win_counts,
            round_a_vs_b=round_1,
            round_b_vs_a=round_2,
        )


JUDGE_CODE_VS_CODE_TEMPLATE = jinja2.Template(
    """You are choosing which of two code listings to accomplish a user query is better.
    You will be given reference code that accomplishes the user query.
    Compare each code listing to the reference code, and choose the code listing that aligns more closely with the reference code.

    Produce your output in the following format:
    ```xml
    <judgement>
    <explanation>
    Your chain of thought and reasoning for which code listing is better.
    </explanation>
    <winner>Put either "A" or "B" here.</winner>
    </judgement>
    ```

    Example 1:
    ```xml
    <judgement>
    <explanation>
    Code A is better because ...
    </explanation>
    <winner>A</winner>
    </judgement>
    ```

    Example 2:
    ```xml
    <judgement>
    <explanation>
    Code B is better because ...
    </explanation>
    <winner>B</winner>
    </judgement>
    ```

    Do not include ANY other text in your output. Stick exactly to the required format.
    These are the only valid values for the <winner> field: "A" or "B".

    # User query
    {{ user_query }}

    # Reference code
    ```python
    {{ reference_code }}
    ```

    # Code A
    ```python
    {{ code_a }}
    ```

    # Code B
    ```python
    {{ code_b }}
    ```
    """,
    undefined=jinja2.StrictUndefined,
)


PLAN_TO_CODE_TEMPLATE = jinja2.Template(
    """You are given a step-by-step plan for accomplishing a user query.
    Follow the plan to write code that accomplishes the user query.
    Each step of the plan contains a list of suggested symbols to use in that step.
    You will be provided the definition of each symbol.
    Import the symbols using the fully-qualified symbol names and use the symbols by their non-qualified names.
    For example, if the symbol is `foo.bar.baz`, you should import it as `import foo.bar.baz as baz` and then use it as `baz.some_function()`.

    Produce your output in the following format:
    ```python
    # your code goes here
    ```

    Do not include any other text in your output. Stick exactly to the required format.
    Do not include any comments in your output.

    # Code Definitions
    ```python
    {% for symbol in all_symbols_used %}
    # Filepath: {{ symbol.filepath }}
    # Import Path: {{ symbol.full_path }}
    {{ truncated_code_display(symbol.code) }}
    {% endfor %}
    ```

    # User query
    {{ user_query }}

    # Plan
    {% for step in plan %}
    ## Step {{ step.index }}
    - {{ step.content }}
    ### Symbols
    {% for symbol in step.search_result.instrumentation.symbols_considered %}
    - {{ symbol.symbol.full_path }}
    {% endfor %}
    {% endfor %}
    """,
    undefined=jinja2.StrictUndefined,
)


def truncated_code_display(code: str) -> str:
    if len(code) > 512:
        return code[:512] + "\n # ... truncated"
    return code


class CodeVsCodeJudge:
    def __init__(self, record: LongCodeArenaRecord, num_judgements: int = 3):
        self.record = record
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.num_judgements = num_judgements

    def parse_response(self, response: Choice) -> PlanVsPlanJudgeRound:
        assert response.message.content is not None
        # Remove markdown code block if present
        content = response.message.content
        lines = content.splitlines()
        if lines and "```" in lines[0]:
            lines = lines[1:]
        if lines and "```" in lines[-1]:
            lines = lines[:-1]

        response.message.content = "\n".join(lines)

        root = ET.fromstring(response.message.content)
        assert root.tag == "judgement"
        assert (explanation := root.find("explanation")) is not None
        assert (winner := root.find("winner")) is not None

        assert explanation.text is not None
        assert winner.text in ["A", "B"]

        match winner.text:
            case "A":
                return PlanVsPlanJudgeRound(winner="A", explanation=explanation.text)
            case "B":
                return PlanVsPlanJudgeRound(winner="B", explanation=explanation.text)
            case _:
                raise ValueError(f"Invalid winner: {winner.text}")

    def convert_plan_to_code(self, plan: Sequence[PlanStep]) -> str:
        all_symbols_used: dict[str, Symbol] = dict()
        for step in plan:
            if step.search_result.instrumentation is None:
                raise ValueError(
                    "Code vs code judge relies on using the instrumentation object "
                    "to get the symbols considered. But it was none.",
                )
            for (
                retrieved_symbol
            ) in step.search_result.instrumentation.symbols_considered:
                all_symbols_used[retrieved_symbol.symbol.full_path] = (
                    retrieved_symbol.symbol
                )
        prompt = PLAN_TO_CODE_TEMPLATE.render(
            user_query=self.record.instruction,
            plan=plan,
            all_symbols_used=all_symbols_used.values(),
            truncated_code_display=truncated_code_display,
        )
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
        )
        assert response.choices[0].message.content is not None
        return response.choices[0].message.content

    def judge_code_vs_code(
        self,
        plan_a: Sequence[PlanStep],
        plan_b: Sequence[PlanStep],
    ) -> list[PlanVsPlanJudgeRound]:
        code_a = self.convert_plan_to_code(plan_a)
        code_b = self.convert_plan_to_code(plan_b)

        prompt = JUDGE_CODE_VS_CODE_TEMPLATE.render(
            user_query=self.record.instruction,
            reference_code=self.record.clean_reference,
            code_a=code_a,
            code_b=code_b,
        )
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            n=self.num_judgements,
        )

        return [self.parse_response(choice) for choice in response.choices]

    def __call__(
        self,
        plan_a: Sequence[PlanStep],
        plan_b: Sequence[PlanStep],
    ) -> Judgement:
        # To account for positional bias, we do two rounds of judging.
        # In the first round, we judge plan A vs plan B.
        # In the second round, we judge plan B vs plan A.
        round_1 = self.judge_code_vs_code(plan_a, plan_b)
        round_2 = self.judge_code_vs_code(plan_b, plan_a)

        win_counts: dict[Literal["A", "B"], int] = {"A": 0, "B": 0}
        for judgement in round_1:
            win_counts[judgement.winner] += 1

        # In round 2, the plans are switched around, so a vote for plan A
        # is actually a vote for plan B and vice versa.
        for judgement in round_2:
            match judgement.winner:
                case "A":
                    win_counts["B"] += 1
                case "B":
                    win_counts["A"] += 1

        win_rate_a = win_counts["A"] / self.num_judgements
        win_rate_b = win_counts["B"] / self.num_judgements

        winner = "A" if win_rate_a > win_rate_b else "B"

        return Judgement(
            winner=winner,
            win_rate=win_rate_a if winner == "A" else win_rate_b,
            counts=win_counts,
            round_a_vs_b=round_1,
            round_b_vs_a=round_2,
        )
