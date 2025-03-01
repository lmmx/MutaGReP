from collections.abc import Callable, Sequence
from typing import (
    Any,
    Literal,
    cast,
)

import instructor
import jinja2
from openai import OpenAI
from pydantic import BaseModel

from mutagrep.plan_search.domain_models import Node

from ..components import BaseSuccessorFunctionInvocationLog, PlanStep
from ..domain_models import (
    CodeSearchTool,
    GoalTestT,
    # Node,
    Plan,
    SuccessorFunction,
)
from ..typing_utils import implements


class BaseAction(BaseModel):
    action_type: Literal["remove_last_step", "append_new_step"]


class AppendNewStep(BaseAction):
    action_type: Literal["append_new_step"] = "append_new_step"
    step_to_add: str


class RemoveLastStep(BaseAction):
    action_type: Literal["remove_last_step"] = "remove_last_step"
    step_to_remove: str


class ProposedPlanModifications(BaseModel):
    proposed_modifications: list[AppendNewStep | RemoveLastStep]


class ProposePossibleFirstSteps(BaseModel):
    proposed_first_steps: list[str]


class AppendNewStepOrRemoveLastStep:
    def __init__(
        self,
        search_tool: CodeSearchTool,
        fix_beam_width_to: int | None = None,
        log_sink: Callable[[BaseSuccessorFunctionInvocationLog], None] | None = None,
    ) -> None:
        self.client = instructor.from_openai(OpenAI())
        self.search_tool = search_tool
        self.fix_beam_width_to = fix_beam_width_to
        self.log_sink = log_sink
        self.client_hook_captures: dict[str, Any] = {}
        self.client.on("completion:kwargs", self.hook_client_kwargs)
        self.client.on("completion:response", self.hook_client_completion_response)

    def log(self, invocation_log: BaseSuccessorFunctionInvocationLog) -> None:
        if self.log_sink is not None:
            self.log_sink(invocation_log)

    def hook_client_kwargs(self, *args: Any, **kwargs: Any) -> None:
        self.client_hook_captures["completion:kwargs"] = kwargs

    def hook_client_completion_response(
        self,
        completion_response: dict[str, str],
    ) -> None:
        self.client_hook_captures["completion:response"] = completion_response

    @staticmethod
    def prepare_prompt(
        state: Node[PlanStep, GoalTestT],
        fix_beam_width_to: int | None,
    ) -> str:
        template = jinja2.Template(
            """# Task
You are an expert Python engineer.
You have been given a user request.
You are provided a codebase that contains functions relevant to the user request.
Your task is to determine a step-by-step plan that describes how to satisfy the user request using the codebase.

# User Request
{{ state.plan.user_query }}

{% if state.parent %}
# Edit History
For each step in the plan, you are  given feedback.
The feedback will tell you whether that step is satisfiable within the codebase.
If a step is satisfiable, that means there exists a function in the codebase to fullfill that step.
If a step is not satisfiable, there is no function in the codebase that can be used to fullfill that step.
{% for state in state.get_lineage() %}
## Version {{ loop.index }}
{% for step in state.plan.steps %}
## Step {{ loop.index }}
{{ step.content }}
### Feedback
Symbol Name: {{ step.search_result.symbol_name }}
Satisfiable: {{ step.search_result.satisfies_intention }}
Justification: {{ step.search_result.justification }}
{% endfor %}
{% endfor %}
{% endif %}

# Plan
You are currently editing the following plan:
{% if state.plan.steps %}
{% for step in state.plan.steps %}
## Step {{ loop.index }}
{{ step.content }}
### Feedback
Symbol Name: {{ step.search_result.symbol_name }}
Satisfiable: {{ step.search_result.satisfies_intention }}
Justification: {{ step.search_result.justification }}
{% endfor %}
{% else %}
The plan is currently empty. You will need to add an initial step.
{% endif %}

# Examples
Let's say the user request is "I want to count the number of red pixels in images linked to at foo.com/index.html".
Good possible first steps might be:
"Start by downloading foo.com/index.html" or "get all links from a url"

Now let's say we have the following plan:
1. "Download foo.com/index.html"
2. "Parse the HTML file to find all links"
3. "Download each of the linked files"

Good possible next steps might be:
- "Count the number of red pixels in each image"
- "Get the pixel values of each image"



# Instructions
Propose potential modifications to the plan to improve it.
We will apply your modifications to the plan to obtain new plans.
This is a process of trial and error and exploration, so make your modifications varied.
The possible modifications are:
- You can remove the last step of the plan if it is unnecessary or unsatisfiable.
- You can add a new step to the end of the plan to get closer to satisfying the user request.
{% if fix_beam_width_to %}
You must propose exactly {{ fix_beam_width_to }} possible modifications to the plan.
Each modification must be different from the others.
{% else %}
You can propose any number of modifications to the plan.
{% endif %}
Your goal is to arrive at a plan that is fully satisfiable and achieves the user request in the minimum number of steps.
The purpose of the plan is to tell an engineer what symbols to use in what order to satisfy the user request.
Therefore, when proposing new steps, focus on unearthing the next symbol needed to satisfy the user request.
Assume that an engineer is capable of gluing the symbols together to satisfy the user request.
Think of each modification as a hypothesis about what the next step in the plan should be.
""",
            undefined=jinja2.StrictUndefined,
        )

        return template.render(state=state, fix_beam_width_to=fix_beam_width_to)

    def __call__(
        self,
        state: Node[PlanStep, GoalTestT],
    ) -> Sequence[Node[PlanStep, GoalTestT]]:
        prompt = self.prepare_prompt(state, self.fix_beam_width_to)
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt},
            ],
            response_model=ProposedPlanModifications,  # type: ignore
        )
        response = cast(ProposedPlanModifications, response)

        new_nodes: list[Node[PlanStep, GoalTestT]] = []

        for proposed_modifications in response.proposed_modifications:
            if proposed_modifications.action_type == "remove_last_step":
                plan_steps = state.plan.steps[:-1]
            elif proposed_modifications.action_type == "append_new_step":
                # We need to check if the step is satisfiable.
                proposed_step_raw = proposed_modifications.step_to_add
                search_result = self.search_tool(proposed_step_raw)
                proposed_steps = [
                    PlanStep(
                        index=len(state.plan.steps),
                        content=proposed_step_raw,
                        search_result=search_result,
                    ),
                ]
                existing_plan_steps = list(state.plan.steps)
                plan_steps = existing_plan_steps + proposed_steps
            else:
                raise ValueError(
                    f"Invalid edit type: {proposed_modifications.action_type}",
                )

            edited_plan = Plan[PlanStep, GoalTestT](
                user_query=state.plan.user_query,
                steps=plan_steps,
            )
            new_node = Node(plan=edited_plan, parent=state, level=state.level + 1)
            new_nodes.append(new_node)

        invocation_log = BaseSuccessorFunctionInvocationLog(
            client_kwargs=self.client_hook_captures.get("completion:kwargs"),
            state=state,
            successors=new_nodes,
            completion_response=self.client_hook_captures.get("completion:response"),
        )
        self.log(invocation_log)

        return new_nodes


implements(SuccessorFunction[PlanStep])(AppendNewStepOrRemoveLastStep)
