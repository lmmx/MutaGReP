import xml.etree.ElementTree as ET
from collections.abc import Sequence

import jinja2
from loguru import logger
from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from pydantic import BaseModel

from mutagrep.plan_search.components import GoalTestT, Node, PlanStep
from mutagrep.plan_search.domain_models import (
    CodeSearchTool,
    Plan,
    SuccessorFunction,
)
from mutagrep.plan_search.typing_utils import implements

PROMPT_TEMPLATE = jinja2.Template(
    """# Example 1
<user_request>
    "I'm planning to revamp my home interior, using '417814-input.png' as reference. I want to see how the space would appear if the carpet was replaced with a wooden floor. Also, could you determine the total number of objects present in the image once the modifications are made?"
</user_request>

<plan>
</plan>

<proposed_edit>
    <step number="0">
        <description>Modify the image to replace the carpet with a wooden floor.</description>
    </step>
</proposed_edit>

# Example 2
<user_request>
    "I'm planning to revamp my home interior, using '417814-input.png' as reference. I want to see how the space would appear if the carpet was replaced with a wooden floor. Also, could you determine the total number of objects present in the image once the modifications are made?"
</user_request>

<plan>
    <step number="0">
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

<proposed_edit>
    <step number="1">
        <description>Detect objects in the modified image.</description>
    </step>
</proposed_edit>


# Instructions
You are given an incomplete plan and your task is to propose a modification to the plan to get it closer to satisfying the user request.
Follow the format shown in "Example 1" and "Example 2" above.

Ensure the step number is a single integer like 0, 1, 12, 42, etc. Do not use letters like 5a or decimals like 0.5.
Produce valid XML and do not include any other text or comments that would break XML parsing.


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
</plan>""",
    undefined=jinja2.StrictUndefined,
)


class ParsedStepFromResponse(BaseModel):
    step_number: int
    description: str


class ParsedResponse(BaseModel):
    parsed_steps: list[ParsedStepFromResponse]
    parsed_from: Choice

    def get_step_matching_index(self, step_index: int) -> ParsedStepFromResponse | None:
        """Returns the step matching the expected step index or None if not found."""
        for step in self.parsed_steps:
            if step.step_number == step_index:
                return step
        return None

    @property
    def step_indices(self) -> list[int]:
        return [step.step_number for step in self.parsed_steps]


class XmlOutputSuccessorFunction:
    def __init__(self, search_tool: CodeSearchTool, beam_width: int = 1) -> None:
        self.client = OpenAI()
        self.search_tool = search_tool
        self.beam_width = beam_width

    @staticmethod
    def prepare_prompt(state: Node[PlanStep, GoalTestT]) -> str:
        return PROMPT_TEMPLATE.render(
            user_request=state.plan.user_query,
            plan=state.plan,
        )

    @staticmethod
    def parse_steps_from_choice(choice: Choice) -> list[ParsedStepFromResponse]:
        content = choice.message.content

        # Check if content is wrapped in triple backticks
        assert content is not None
        if "```xml" in content and "```" in content[content.find("```xml") + 6 :]:
            start = content.find("```xml") + 6
            end = content.find("```", start)
            content = content[start:end].strip()

        root = ET.fromstring(content)  # type: ignore
        target_xml_nodes = root.findall("./step")

        parsed_steps = []
        for xml_node in target_xml_nodes:
            try:
                step_number = int(xml_node.attrib["number"])  # type: ignore
            except (TypeError, ValueError):
                # Occasionally the step number is something like 9a or 0a or 11b, etc.
                # Extract numeric part from string like "9a" or "11b"
                # We also need to handle the case where it is a decimal like 2.5
                raw_step_number = xml_node.attrib["number"]  # type: ignore
                numeric_part = "".join(c for c in raw_step_number if c.isdigit())
                step_num_as_float = float(numeric_part)
                step_number = int(step_num_as_float)

            try:
                description = xml_node.find("description").text  # type: ignore
            except AttributeError:
                logger.warning(
                    f"No description found for step {step_number} in XML response. Skipping.",
                )
                logger.warning(f"XML response: {content}")
                continue

            modification = ParsedStepFromResponse(
                step_number=step_number,
                description=description,  # type: ignore
            )
            parsed_steps.append(modification)

        return parsed_steps

    def parse_steps_from_response(
        self,
        response: ChatCompletion,
    ) -> list[ParsedResponse]:
        responses: list[ParsedResponse] = []
        for choice in response.choices:
            parsed_steps = self.parse_steps_from_choice(choice)
            responses.append(
                ParsedResponse(parsed_steps=parsed_steps, parsed_from=choice),
            )
        return responses

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
            n=self.beam_width,
        )

        proposed_successors = self.parse_steps_from_response(response)

        new_nodes: list[Node[PlanStep, GoalTestT]] = []

        expected_step_index = len(state.plan.steps)

        for proposed_successor in proposed_successors:
            # We need to check if the step is satisfiable.
            proposed_step = proposed_successor.get_step_matching_index(
                expected_step_index,
            )

            if proposed_step is None:
                logger.warning(
                    f"expected to find step index {expected_step_index} but only found {proposed_successor.step_indices}",
                )
                continue

            proposed_step_raw = proposed_step.description

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

            edited_plan = Plan[PlanStep, GoalTestT](
                user_query=state.plan.user_query,
                steps=plan_steps,
            )
            new_node = Node(plan=edited_plan, parent=state, level=state.level + 1)
            new_nodes.append(new_node)
        return new_nodes


implements(SuccessorFunction)(XmlOutputSuccessorFunction)
