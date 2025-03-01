import re
from collections.abc import Callable, Iterable, Sequence
from typing import (
    Any,
    Generic,
    Literal,
    cast,
)

import instructor
import jinja2
from openai import OpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, model_validator
from typing_extensions import Self

from mutagrep.coderec.v3.symbol_mining import Symbol
from mutagrep.plan_search.domain_models import Node

from .domain_models import (
    CodeSearchTool,
    CodeSearchToolOutput,
    GoalTestFunction,
    GoalTestT,
    HasBeenVisitedFunction,
    # Node,
    Plan,
    PlanStepT,
    SuccessorFunction,
)
from .mnms_benchmark import MnmsPlanStep
from .typing_utils import implements


class PlanStep(BaseModel):
    index: int
    content: str
    search_result: CodeSearchToolOutput

    def to_mnms_plan_step(self) -> MnmsPlanStep:
        return MnmsPlanStep(
            id=self.index,
            name=self.search_result.symbol_name or "SEARCH_RESULT_MISSING",
            args=None,
        )


def extract_symbols_used_from_plan(
    plan: Plan[PlanStep, GoalTestT],
) -> dict[str, Symbol]:
    all_symbols_used: dict[str, Symbol] = dict()
    for step in plan.steps:
        if step.search_result.instrumentation is None:
            raise ValueError(
                "extract_symbols_used_from_plan relies on using the instrumentation "
                "object to get the symbols considered. But it was none.",
            )
        for retrieved_symbol in step.search_result.instrumentation.symbols_considered:
            all_symbols_used[retrieved_symbol.symbol.full_path] = (
                retrieved_symbol.symbol
            )
    return all_symbols_used


def extract_symbols_used_from_node(
    node: Node[PlanStep, GoalTestT],
) -> dict[str, Symbol]:
    return extract_symbols_used_from_plan(node.plan)


class GoalTest(BaseModel):
    satisfies_user_request: bool
    explanation: str

    def __bool__(self) -> bool:
        return self.satisfies_user_request


class LlmPlanStep(BaseModel):
    content: str
    index: int


class LlmPlan(BaseModel):
    steps: list[LlmPlanStep]
    edit_type: Literal["remove_last_step", "add_new_step"]

    @model_validator(mode="after")
    def check_steps_not_empty_when_new_step_added(self) -> Self:
        if self.edit_type == "add_new_step" and len(self.steps) == 0:
            raise ValueError("'steps' cannot be empty when edit_type is 'add_new_step'")
        return self

    @model_validator(mode="after")
    def check_cannot_remove_last_step_when_empty(self) -> Self:
        if self.edit_type == "remove_last_step" and len(self.steps) == 0:
            raise ValueError(
                "'steps' cannot be empty when edit_type is 'remove_last_step'",
            )
        return self


class MonotonicLlmPlan(LlmPlan):
    @model_validator(mode="after")
    def check_steps_are_monotonic(self) -> Self:
        if self.edit_type != "add_new_step":
            raise ValueError("'edit_type' must be 'add_new_step'")
        return self


class SuccessorFunctionFollowHumanWrittenPlan:
    def __init__(self, plan: Sequence[str], search_tool: CodeSearchTool) -> None:
        self.plan = plan
        self.search_tool = search_tool
        self.step_index = 0

    def __call__(
        self,
        state: Node[PlanStep, GoalTestT],
    ) -> Sequence[Node[PlanStep, GoalTestT]]:
        # If we've already used all steps in the plan, return empty list
        current_step_index = self.step_index
        if current_step_index >= len(self.plan):
            raise ValueError("Already used all steps in the plan")

        # Get the next step from the human-written plan
        next_step_content = self.plan[current_step_index]

        # Search for the step in the codebase
        search_result = self.search_tool(next_step_content)

        # Create the new plan step
        new_step = PlanStep(
            index=current_step_index,
            content=next_step_content,
            search_result=search_result,
        )

        # Create new plan with all existing steps plus the new one
        plan_steps = list(state.plan.steps) + [new_step]
        edited_plan = Plan[PlanStep, GoalTestT](
            user_query=state.plan.user_query,
            steps=plan_steps,
        )

        # Create and return new node
        new_node = Node(plan=edited_plan, parent=state, level=state.level + 1)
        self.step_index += 1
        return [new_node]


class BaseSuccessorFunctionInvocationLog(BaseModel, Generic[PlanStepT, GoalTestT]):
    state: Node[PlanStepT, GoalTestT]
    successors: list[Node[PlanStepT, GoalTestT]]
    client_kwargs: dict | None = None
    completion_response: ChatCompletion | None = None


class SuccessorFunctionAddOrRemoveLastStep:
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

# Instructions
Propose new plans that are edited from the current plan.
{% if fix_beam_width_to %}
You must propose exactly {{ fix_beam_width_to }} plans.
Each plan must be different from the others.
{% else %}
You can propose any number of plans.
{% endif %}
For each step in the plan, you will be given feedback.
The feedback will tell you whether that step is satisfiable within the codebase.
The feedback will tell you whether that step is satisfiable within the codebase.
If a step is satisfiable, that means there exists a function in the codebase to fullfill that step.
If a step is not satisfiable, there is no function in the codebase that can be used to fullfill that step.
Your goal is to arrive at a plan that is fully satisfiable and achieves the user request in the minimum number of steps.

You are allowed to make the following edits:
- You can remove the last step of the plan.
- You can add a new step to the end of the plan.

## Plan Format
The plans must consist of a sequence of steps.
You must output valid JSON.
You must state whether you are removing the last step or adding a new step.
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
            response_model=Iterable[LlmPlan],  # type: ignore
        )
        response = cast(list[LlmPlan], response)

        new_nodes: list[Node[PlanStep, GoalTestT]] = []

        for llm_plan in response:
            if llm_plan.edit_type == "remove_last_step":
                plan_steps = state.plan.steps[:-1]
            elif llm_plan.edit_type == "add_new_step":
                # We need to check if the step is satisfiable.
                proposed_step_raw = llm_plan.steps[-1]
                search_result = self.search_tool(proposed_step_raw.content)
                proposed_steps = [
                    PlanStep(
                        index=proposed_step_raw.index,
                        content=proposed_step_raw.content,
                        search_result=search_result,
                    ),
                ]
                existing_plan_steps = list(state.plan.steps)
                plan_steps = existing_plan_steps + proposed_steps
            else:
                raise ValueError(f"Invalid edit type: {llm_plan.edit_type}")

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


implements(SuccessorFunction[PlanStep])(SuccessorFunctionAddOrRemoveLastStep)


class SuccessorFunctionAddOrRemoveLastStepTextOnly:
    def __init__(
        self,
        search_tool: CodeSearchTool,
        fix_beam_width_to: int | None = None,
        log_sink: Callable[[BaseSuccessorFunctionInvocationLog], None] | None = None,
    ) -> None:
        self.client = OpenAI()
        self.search_tool = search_tool
        self.fix_beam_width_to = fix_beam_width_to
        self.log_sink = log_sink
        self.plan_edit_pattern = re.compile(r"^# Plan Edit \d+$", re.MULTILINE)
        self.remove_step_pattern = re.compile(
            r"^## Edit Type\nRemove last step\.$",
            re.MULTILINE,
        )
        self.add_step_pattern = re.compile(
            r"^## Edit Type\nAdd new step: (\d+)\. (.+)$",
            re.MULTILINE,
        )

    def log(self, invocation_log: BaseSuccessorFunctionInvocationLog) -> None:
        if self.log_sink is not None:
            self.log_sink(invocation_log)

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

# Instructions
Propose new plans that are edited from the current plan.
{% if fix_beam_width_to %}
You must propose exactly {{ fix_beam_width_to }} plans.
Each plan must be different from the others.
{% else %}
You can propose any number of plans.
{% endif %}
For each step in the plan, you will be given feedback.
The feedback will tell you whether that step is satisfiable within the codebase.
The feedback will tell you whether that step is satisfiable within the codebase.
If a step is satisfiable, that means there exists a function in the codebase to fullfill that step.
If a step is not satisfiable, there is no function in the codebase that can be used to fullfill that step.
Your goal is to arrive at a plan that is fully satisfiable and achieves the user request in the minimum number of steps.

You are allowed to make the following edits:
- You can remove the last step of the plan.
- You can add a new step to the end of the plan.

## Plan Format
Each plan must be under a markdown heading that begins with "# Plan Edit <plan_number>".
Underneath the heading, write the following:
    - Add a heading "## Edit Type".
    - Under the heading "## Edit Type", write one of the following:
        - "Remove last step." if you are removing the last step of the plan.
        - "Add new step: <step_content>." if you are adding a new step to the end of the plan, where <step_content> is the content of the new step.
            - <step_content> must begin with a number, which will be the index of the new step in the plan. This number should be one more than the index of the last step in the plan you are editing..

It is critical you stick to this format exactly, and do not output anything else.
""",
            undefined=jinja2.StrictUndefined,
        )

        return template.render(state=state, fix_beam_width_to=fix_beam_width_to)

    def parse_llm_output(self, output: str) -> list[dict[str, Any]]:
        """Parse the LLM output into a structured format."""
        plans = []
        # Skip the first empty split
        for plan in self.plan_edit_pattern.split(output)[1:]:
            plan = plan.strip()
            if self.remove_step_pattern.search(plan):
                plans.append({"edit_type": "remove_last_step"})
            else:
                match = self.add_step_pattern.search(plan)
                if match:
                    plans.append(
                        {
                            "edit_type": "add_new_step",
                            "step_number": int(match.group(1)),
                            "step_content": match.group(2),
                        },
                    )
                else:
                    raise ValueError(f"Invalid plan edit: {plan}")
        return plans

    def __call__(
        self,
        state: Node[PlanStep, GoalTestT],
    ) -> Sequence[Node[PlanStep, GoalTestT]]:
        prompt = self.prepare_prompt(state, self.fix_beam_width_to)
        client_kwargs = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": prompt},
            ],
        }
        response = self.client.chat.completions.create(**client_kwargs)
        response_content = response.choices[0].message.content
        new_nodes: list[Node[PlanStep, GoalTestT]] = []

        assert response_content is not None
        parsed_plans = self.parse_llm_output(response_content)

        for plan_edit in parsed_plans:
            if plan_edit["edit_type"] == "remove_last_step":
                plan_steps = state.plan.steps[:-1]
            elif plan_edit["edit_type"] == "add_new_step":
                search_result = self.search_tool(plan_edit["step_content"])
                proposed_step = PlanStep(
                    index=plan_edit["step_number"],
                    content=plan_edit["step_content"],
                    search_result=search_result,
                )
                existing_plan_steps = list(state.plan.steps)
                plan_steps = existing_plan_steps + [proposed_step]
            else:
                raise ValueError(f"Invalid edit type: {plan_edit['edit_type']}")

            edited_plan = Plan[PlanStep, GoalTestT](
                user_query=state.plan.user_query,
                steps=plan_steps,
            )
            new_node = Node(plan=edited_plan, parent=state, level=state.level + 1)
            new_nodes.append(new_node)

        invocation_log = BaseSuccessorFunctionInvocationLog(
            state=state,
            successors=new_nodes,
            completion_response=response,
            client_kwargs=client_kwargs,
        )
        self.log(invocation_log)

        return new_nodes


implements(SuccessorFunction[PlanStep])(SuccessorFunctionAddOrRemoveLastStepTextOnly)


class GoalTestPlanSatisfiesUserRequest:
    def __init__(self) -> None:
        self.client = instructor.from_openai(OpenAI())

    @staticmethod
    def prepare_prompt(state: Node[PlanStep, GoalTest]) -> str:
        template = jinja2.Template(
            """# Task
You are an expert Python engineer.
You have been given a user request.
You are provided a codebase that contains functions relevant to the user request.
You are also provided a plan that proposes a step-by-step process to satisfy the user request.
Your task is to determine if the plan satisfies the user request.

# User Request
{{ state.plan.user_query }}

# Plan
{% for step in state.plan.steps %}
## Step {{ loop.index }}
{{ step.content }}
### Feedback
Symbol Name: {{ step.search_result.symbol_name }}
Satisfiable: {{ step.search_result.satisfies_intention }}
Justification: {{ step.search_result.justification }}
{% endfor %}

# Instructions
Feedback has been provided for each step in the plan.
The feedback will tell you whether that step is satisfiable within the codebase.
If a step is satisfiable, that means there exists a function in the codebase to fullfill that step.
If a step is not satisfiable, there is no function in the codebase that can be used to fullfill that step.

Use the following criteria to determine if the plan satisfies the user request:
- Are all steps in the plan satisfiable?
- If the plan is followed step-by-step, will the final output be a solution to the user request?
- If the plan is followed step-by-step, will there be anything missing from the final output that the user has specifically asked for in the user request?

Provide a justification for your answer. If the plan does not satisfy the user request, explain what is missing from the final output.
""",
            undefined=jinja2.StrictUndefined,
        )
        return template.render(state=state)

    def __call__(self, state: Node[PlanStep, GoalTest]) -> GoalTest:
        prompt = self.prepare_prompt(state)
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt},
            ],
            response_model=GoalTest,
        )
        return response


implements(GoalTestFunction[PlanStep, GoalTest])(GoalTestPlanSatisfiesUserRequest)


class SuccessorFunctionMonotonicAddStep:
    def __init__(self, search_tool: CodeSearchTool) -> None:
        self.client = instructor.from_openai(OpenAI())
        self.search_tool = search_tool

    @staticmethod
    def prepare_prompt(state: Node[PlanStep, GoalTestT]) -> str:
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

# Instructions
Propose new plans that are edited from the current plan by adding new steps.
You can propose any number of plans.
For each step in the plan, you will be given feedback.
The feedback will tell you whether that step is satisfiable within the codebase.
If a step is satisfiable, that means there exists a function in the codebase to fulfill that step.
If a step is not satisfiable, there is no function in the codebase that can be used to fulfill that step.
Your goal is to arrive at a plan that is fully satisfiable and achieves the user request in the minimum number of steps.

You are only allowed to make the following edit:
- You can add a new step to the end of the plan.

## Plan Format
The plans must consist of a sequence of steps.
You must output valid JSON.
The edit_type must always be "add_new_step".
""",
            undefined=jinja2.StrictUndefined,
        )

        return template.render(state=state)

    def __call__(
        self,
        state: Node[PlanStep, GoalTestT],
    ) -> Sequence[Node[PlanStep, GoalTestT]]:
        prompt = self.prepare_prompt(state)
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt},
            ],
            response_model=Iterable[MonotonicLlmPlan],  # type: ignore
        )
        response = cast(list[MonotonicLlmPlan], response)

        new_nodes: list[Node[PlanStep, GoalTestT]] = []

        for llm_plan in response:
            # We need to check if the step is satisfiable.
            proposed_step_raw = llm_plan.steps[-1]
            search_result = self.search_tool(proposed_step_raw.content)
            proposed_step = PlanStep(
                index=proposed_step_raw.index,
                content=proposed_step_raw.content,
                search_result=search_result,
            )

            existing_plan_steps = list(state.plan.steps)
            plan_steps = existing_plan_steps + [proposed_step]

            edited_plan = Plan[PlanStep, GoalTestT](
                user_query=state.plan.user_query,
                steps=plan_steps,
            )
            new_node = Node(plan=edited_plan, parent=state, level=state.level + 1)
            new_nodes.append(new_node)

        return new_nodes


implements(SuccessorFunction[PlanStep])(SuccessorFunctionMonotonicAddStep)


class AlwaysReturnsVisitedFalse(Generic[PlanStepT, GoalTestT]):
    """A stub implementation of the HasBeenVisitedFunction protocol that always
    claims that a state has not been visited.
    """

    def __call__(
        self,
        state: Node[PlanStepT, GoalTestT],
        visited: Sequence[Node[PlanStepT, GoalTestT]],
    ) -> bool:
        return False


implements(HasBeenVisitedFunction[PlanStep, GoalTest])(
    AlwaysReturnsVisitedFalse[PlanStep, GoalTest],
)


class AlwaysReturnsGoalTestTrue(Generic[PlanStepT]):
    """A stub implementation of the GoalTestFunction protocol that always claims
    that the plan satisfies the user request.
    """

    def __call__(self, state: Node[PlanStepT, GoalTest]) -> GoalTest:
        return GoalTest(satisfies_user_request=True, explanation="")


implements(GoalTestFunction[PlanStep, GoalTest])(AlwaysReturnsGoalTestTrue[PlanStep])


class AlwaysReturnsGoalTestFalse(Generic[PlanStepT]):
    """A stub implementation of the GoalTestFunction protocol that always claims
    that the plan does not satisfy the user request.
    """

    def __call__(self, state: Node[PlanStepT, GoalTest]) -> GoalTest:
        return GoalTest(satisfies_user_request=False, explanation="")


implements(GoalTestFunction[PlanStep, GoalTest])(AlwaysReturnsGoalTestFalse[PlanStep])


class EmptySetSuccessorFunction(Generic[PlanStepT]):
    """A stub implementation of the SuccessorFunction protocol that returns an empty
    list of successors.
    """

    def __call__(
        self,
        state: Node[PlanStepT, GoalTestT],
    ) -> Sequence[Node[PlanStepT, GoalTestT]]:
        return []


implements(SuccessorFunction[PlanStep])(EmptySetSuccessorFunction[PlanStep])
