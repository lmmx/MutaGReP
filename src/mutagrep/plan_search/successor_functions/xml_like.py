import xml.etree.ElementTree as ET
from enum import Enum
from typing import Sequence

import jinja2
from openai import OpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from mutagrep.plan_search.components import GoalTestT, Node, PlanStep
from mutagrep.plan_search.domain_models import (CodeSearchTool, Plan,
                                                SuccessorFunction)
from mutagrep.plan_search.typing_utils import implements


class AllowedEdit(Enum):
    REMOVE_LAST_STEP = "You can remove the last step of the plan."
    ADD_NEW_STEP = "You can add a new step to the end of the plan."


MONOTONIC_ALLOWED_ACTIONS = (AllowedEdit.REMOVE_LAST_STEP,)
ADD_OR_REMOVE_LAST_STEP_ALLOWED_ACTIONS = (
    AllowedEdit.REMOVE_LAST_STEP,
    AllowedEdit.ADD_NEW_STEP,
)


PROMPT_TEMPLATE = jinja2.Template(
    """# Example 1
<user_request>
    "I'm planning to revamp my home interior, using '417814-input.png' as reference. I want to see how the space would appear if the carpet was replaced with a wooden floor. Also, could you determine the total number of objects present in the image once the modifications are made?"
</user_request>

<plan>
</plan>

<allowed_edits>
    You are allowed to make the following edits:
    {% for edit in allowed_edits %}
    - {{ edit.value }}
    {% endfor %}
</allowed_edits>

<proposed_edits>
    <edit number="1">
        <step number="1">
            <description>Modify the image to replace the carpet with a wooden floor.</description>
        </step>
    </edit>
</proposed_edits>

# Example 2
<user_request>
    "I'm planning to revamp my home interior, using '417814-input.png' as reference. I want to see how the space would appear if the carpet was replaced with a wooden floor. Also, could you determine the total number of objects present in the image once the modifications are made?"
</user_request>

<plan>
    <step number="1">
        <description>Modify the image to replace the carpet with a wooden floor.</description>
        <feedback>
            <symbol_name>mnm.tool_api.image_editing</symbol_name>
            <satisfiable>True</satisfiable>
            <justification>
                The function `image_editing` is specifically designed to modify images based on a given prompt, and it will replace the carpet with a wooden floor as requested.
            </justification>
        </feedback>
    </step>
</plan>

<allowed_edits>
    You are allowed to make the following edits:
    {% for edit in allowed_edits %}
    - {{ edit.value }}
    {% endfor %}
</allowed_edits>

<proposed_edits>
    <edit number="1">
        <step number="2">
            <description>Detect objects in the modified image.</description>
        </step>
    </edit>

    <edit number="2">
        <step number="2">
            <description>Segment the modified image to identify distinct object regions.</description>
        </step>
    </edit>
</proposed_edits>


# Instructions
You are given an incomplete plan and your task is to propose modifications to the plan to get it closer to satisfying the user request.
Follow the format shown in the example above.
You may propose any number of edits.
Each edit should have the same step number, do not propose edits that are more than 1 step in the future.
DO NOT try to plan multiple steps ahead, only propose edits that are 1 step in the future.
There are NO exceptions to this rule.


<user_request>
{{ user_request }}
</user_request>

<plan>
{% for step in plan.steps %}
<step number="{{ step.index }}">
    <description>{{ step.content }}</description>
    <feedback>
        <symbol_name>{{ step.search_result.symbol_name }}</symbol_name>
        <satisfiable>{{ step.search_result.satisfies_intention }}</satisfiable>
        <justification>
            {{ step.search_result.justification }}
        </justification>
    </feedback>
</step>
{% endfor %}
</plan>

<allowed_edits>
    You are allowed to make the following edits:
    {% for edit in allowed_edits %}
    - {{ edit.value }}
    {% endfor %}
</allowed_edits>""",
    undefined=jinja2.StrictUndefined,
)


class ParsedPlanModification(BaseModel):
    step_number: int
    description: str


class XmlOutputSuccessorFunction:
    def __init__(
        self, allowed_actions: Sequence[AllowedEdit], search_tool: CodeSearchTool
    ) -> None:
        self.allowed_actions = allowed_actions
        self.client = OpenAI()
        self.search_tool = search_tool

    @staticmethod
    def prepare_prompt(state: Node[PlanStep, GoalTestT]) -> str:
        return PROMPT_TEMPLATE.render(
            user_request=state.plan.user_query,
            plan=state.plan,
            allowed_edits=MONOTONIC_ALLOWED_ACTIONS,
        )

    def parse_modifications_from_response(
        self, response: ChatCompletion
    ) -> list[ParsedPlanModification]:
        # Parse the XML-like response content
        root = ET.fromstring(response.choices[0].message.content)  # type: ignore

        # Find all proposed edits
        # Use './edit' to find <edit> elements directly under the root
        proposed_edits = root.findall("./edit")

        modifications = []
        for edit in proposed_edits:
            step_number = int(edit.find("step").attrib["number"])  # type: ignore
            description = edit.find("step/description").text  # type: ignore

            modification = ParsedPlanModification(
                step_number=step_number, description=description  # type: ignore
            )
            modifications.append(modification)

        return modifications

    def __call__(
        self, state: Node[PlanStep, GoalTestT]
    ) -> Sequence[Node[PlanStep, GoalTestT]]:
        prompt = self.prepare_prompt(state)
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt},
            ],
        )

        parsed_modifications = self.parse_modifications_from_response(response)

        new_nodes: list[Node[PlanStep, GoalTestT]] = []

        for proposed_modification in parsed_modifications:
            # We need to check if the step is satisfiable.
            proposed_step_raw = proposed_modification.description
            search_result = self.search_tool(proposed_step_raw)
            proposed_steps = [
                PlanStep(
                    index=len(state.plan.steps),
                    content=proposed_step_raw,
                    search_result=search_result,
                )
            ]
            existing_plan_steps = list(state.plan.steps)
            plan_steps = existing_plan_steps + proposed_steps

            edited_plan = Plan[PlanStep, GoalTestT](
                user_query=state.plan.user_query, steps=plan_steps
            )
            new_node = Node(plan=edited_plan, parent=state, level=state.level + 1)
            new_nodes.append(new_node)
        return new_nodes


implements(SuccessorFunction)(XmlOutputSuccessorFunction)
