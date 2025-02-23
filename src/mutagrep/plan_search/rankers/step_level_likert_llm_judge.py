import os
import xml.etree.ElementTree as ET
from typing import Callable, Generic

import jinja2
import numpy as np
from openai import OpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential
from typing_extensions import Self

from mutagrep.coderec.v3.symbol_mining import Symbol
from mutagrep.plan_search.domain_models import (GoalTestT, Node, Plan,
                                                RankingFunction)
from mutagrep.plan_search.typing_utils import implements

from ..components import PlanStep


def truncated_code_display(code: str):
    if len(code) > 512:
        return code[:512] + "\n # ... truncated"
    return code


judge_prompt_template = jinja2.Template(
    """# Instructions
You will be given a user request and a plan to accomplish that user request with a codebase.
Each step of the plan will have a list of symbols that are relevant to that step.

Judge the entire plan based on the following criteria on a scale of 1 to 7:
- This plan solves the user request. 
- This plan has no unnecessary or redundant steps. 

Judge each step based on the following criteria on a scale of 1 to 7:
- This step is achievable with the symbols found.

The scale indicates your degree of agreement with the statement.
- 1: Strongly disagree
- 2: Disagree
- 3: Slightly disagree
- 4: Neutral
- 5: Slightly agree
- 6: Agree
- 7: Strongly agree

Your response must be in the following XML format:
<judgement>
    <plan_level>
        <solves_user_request>NUMBER</solves_user_request>
        <no_unnecessary_steps>NUMBER</no_unnecessary_steps>
    </plan_level>
    <steps>
        <step>
            <step_index>INDEX_OF_STEP</step_index>
            <achievable_with_symbols>NUMBER</achievable_with_symbols>
        </step>
        <!-- Repeat for each step -->
    </steps>
</judgement>

ONLY output the XML.
DO NOT wrap the XML in triple backticks.
DO NOT provide any other output.

# User Request
{{ plan.user_query }}

# Code Definitions
```python
{% for symbol in all_symbols_used %}
# Filepath: {{ symbol.filepath }}
# Import Path: {{ symbol.full_path }}
{{ truncated_code_display(symbol.code) }}
{% endfor %}
```
# Plan
{% for step in plan.steps %}
## Step {{ step.index }}
{{ step.content }}
### Symbols Found
{% for symbol in step.search_result.instrumentation.symbols_considered %}
- {{ symbol.symbol.full_path }}
{% endfor %}
{% endfor %}
""",
    undefined=jinja2.StrictUndefined,
)


def make_prompt(plan: Plan[PlanStep, GoalTestT]) -> str:
    all_symbols_used: dict[str, Symbol] = dict()
    for step in plan.steps:
        if step.search_result.instrumentation is None:
            raise ValueError(
                "Likert LLM judge relies on using the instrumentation object "
                "to get the symbols considered. But it was none."
            )
        for retrieved_symbol in step.search_result.instrumentation.symbols_considered:
            all_symbols_used[retrieved_symbol.symbol.full_path] = (
                retrieved_symbol.symbol
            )

    prompt = judge_prompt_template.render(
        plan=plan,
        all_symbols_used=all_symbols_used.values(),
        trim_blocks=True,
        lstrip_blocks=True,
        truncated_code_display=truncated_code_display,
    )
    return prompt


class StepJudgement(BaseModel):
    step_index: int
    achievable_with_symbols: int


def strip_code_backticks(text: str) -> str:
    exploded = text.split("\n")
    return "\n".join([_ for _ in exploded if "```" not in _])


class JudgeResponse(BaseModel):
    solves_user_request: int
    no_unnecessary_steps: int
    step_judgements: list[StepJudgement]

    @classmethod
    def from_xml(cls, xml_str: str) -> Self:
        xml_str = strip_code_backticks(xml_str)

        try:
            root = ET.fromstring(xml_str)
        except ET.ParseError:
            print(xml_str)
            raise

        # Parse plan level metrics
        plan_level = root.find("plan_level")
        solves_request = int(plan_level.find("solves_user_request").text)  # type: ignore
        no_unnecessary = int(plan_level.find("no_unnecessary_steps").text)  # type: ignore

        # Parse step judgements
        steps = root.find("steps")
        step_judgements = []
        for step in steps.findall("step"):  # type: ignore
            step_judgements.append(
                StepJudgement(
                    step_index=int(step.find("step_index").text),  # type: ignore
                    achievable_with_symbols=int(
                        step.find("achievable_with_symbols").text  # type: ignore
                    ),
                )
            )

        return cls(
            solves_user_request=solves_request,
            no_unnecessary_steps=no_unnecessary,
            step_judgements=step_judgements,
        )


def default_aggregation_fn(judge_response: JudgeResponse) -> float:
    # Get the average of the step judgements
    step_level_mean = float(
        np.mean(
            [step.achievable_with_symbols for step in judge_response.step_judgements]
        )
    )

    return float(
        np.mean(
            [
                judge_response.solves_user_request,
                step_level_mean,
                judge_response.no_unnecessary_steps,
            ]
        )
    )


class StepLevelLikertLlmJudge(Generic[GoalTestT]):
    def __init__(
        self, aggregation_fn: Callable[[JudgeResponse], float] = default_aggregation_fn
    ):
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.aggregation_fn = aggregation_fn

    def get_judge_response(self, state: Node[PlanStep, GoalTestT]) -> JudgeResponse:
        if len(state.plan.steps) == 0:
            return JudgeResponse(
                solves_user_request=0,
                no_unnecessary_steps=0,
                step_judgements=[],
            )

        prompt = make_prompt(state.plan)

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
        )

        assert response.choices[0].message.content is not None

        return JudgeResponse.from_xml(response.choices[0].message.content)

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def __call__(self, state: Node[PlanStep, GoalTestT]) -> float:
        judge_response = self.get_judge_response(state)
        return self.aggregation_fn(judge_response)


implements(RankingFunction)(StepLevelLikertLlmJudge)
