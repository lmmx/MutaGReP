import os
from typing import Generic

import instructor
import jinja2
from openai import OpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from mutagrep.coderec.v3.symbol_mining import Symbol
from mutagrep.plan_search.domain_models import GoalTestT, Node, Plan, RankingFunction
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

Judge the plan based on the following criteria on a scale of 1 to 5:
- Does this plan solve the user request?
- Is each step achievable with the symbols found?
- Does this plan have unnecessary or repeated steps?

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
                "to get the symbols considered. But it was none.",
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


class JudgeResponse(BaseModel):
    solves_user_request: int
    achievable_with_symbols: int
    unnecessary_or_repeated_steps: int

    def score(self) -> float:
        return (
            self.solves_user_request
            + self.achievable_with_symbols
            + self.unnecessary_or_repeated_steps
        ) / 3


class LikertLlmJudge(Generic[GoalTestT]):
    def __init__(self):
        self.client = instructor.from_openai(
            OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    def __call__(self, state: Node[PlanStep, GoalTestT]) -> float:
        prompt = make_prompt(state.plan)

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_model=JudgeResponse,
        )
        return response.score()


implements(RankingFunction)(LikertLlmJudge)
