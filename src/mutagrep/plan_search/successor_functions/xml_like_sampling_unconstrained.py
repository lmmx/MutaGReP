import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import jinja2
from loguru import logger
from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt
from typing_extensions import assert_never

from mutagrep.coderec.v3.symbol_mining import (Symbol, SymbolCategory,
                                               extract_symbol_signature)
from mutagrep.plan_search.components import GoalTestT, Node, PlanStep
from mutagrep.plan_search.domain_models import (CodeSearchTool,
                                                CodeSearchToolOutput, Plan,
                                                SuccessorFunction)
from mutagrep.plan_search.typing_utils import implements

UNCONSTRAINED_PROMPT_TEMPLATE = jinja2.Template(
    """# Instructions
You are given a plan for implementing a user request. Your task is to propose a modified version of the plan that might better satisfy the user request.

For each step in the plan, you will be given feedback from a search tool. 
The feedback consists of:
- Symbols in the codebase that are most likely to help accomplish the step
- The signatures of those symbols
Use the feedback to help you make modifications to the plan.
For example, you may be able to intuit from reading the signatures that none of the symbols are relevant to the step, and thus remove the step and replace it with a new step.
Or, you may see that it seems like the step is not necessary, and thus remove it.
Or, you may see that the step seems feasible to accomplish and requires no changes.


You can make any combination of the following modifications:
- Add new steps anywhere in the plan (e.g. to fill in missing steps)
- Remove existing steps (e.g. to remove unnecessary steps)
- Modify existing steps to be more specific or accurate (e.g. based on the feedback from the search tool)
- Reorder steps to improve the execution flow (e.g. based on the feedback from the search tool)

Format your output as an XML document with the following structure:
```xml
<thought>
<-- YOUR THOUGHT PROCESS for modifying the plan -->
</thought>
<plan>
    <step number="0">
        <description>Modify the image to replace the carpet with a wooden floor.</description>
    </step>
    <step number="1">
        <description>Detect objects in the modified image.</description>
    </step>
    <step number="stepnum">
        <description>stepdesc</description>
    </step>
</plan>
```
Here, `stepnum` is an integer and `stepdesc` is a string. 
IMPORTANT: The step number must be an integer enclosed in double quotes.
IMPORTANT: Your output must be valid XML and must not contain any other text or comments that would break XML parsing.

# Your Task
The user request is: "{{ user_request }}"

Here is a sample of some of the symbols in the codebase:
{% for symbol in starting_symbols %}
- {{ symbol.name }}: {{ symbol.filepath }}
{% endfor %}

Here is the repository tree:
{{ repo_tree }}

Remember that the list above is just a starting point to give you an idea of what functionality is available in the codebase.
There are many more symbols in the codebase that are not listed above.

{% if plan.steps | length > 0 %}
The current plan is:
<plan>
{% for step in plan.steps %}
<step number="{{ step.index }}">
    <description>{{ step.content }}</description>
    <feedback>
        {% for signature in get_signatures(step.search_result) %}
        - {{ signature }}
        {% endfor %}
    </feedback>
</step>
{% endfor %}
</plan>
Propose a modified plan that better accomplishes the user request.
{% else %}
Currently, the plan is empty. Propose an initial plan (it can be incomplete), that we can work on modifying.
{% endif %}

Remember the following guidelines:
- The step number must be an integer enclosed in double quotes.
- Your output must be valid XML and must not contain any other text or comments that would break XML parsing.
""",
    undefined=jinja2.StrictUndefined,
)


@dataclass
class PromptContext:
    user_request: str
    starting_symbols: list[Symbol]
    plan: Plan
    repo_tree: str

    def get_signatures(self, search_result: CodeSearchToolOutput) -> list[str]:
        if search_result.instrumentation is None:
            raise ValueError(
                "The search tool did not list which symbols were considered."
            )
        signatures: list[str] = []
        for symbol in search_result.instrumentation.symbols_considered:
            try:
                signatures.append(extract_symbol_signature(symbol.symbol))
            except Exception:
                logger.opt(exception=True).warning(
                    f"Could not extract signature for symbol {symbol.symbol.name}"
                )
                match symbol.symbol.symbol_type:
                    case SymbolCategory.FUNCTION:
                        signatures.append(f"def {symbol.symbol.name}(...): ...")
                    case SymbolCategory.CLASS:
                        signatures.append(f"class {symbol.symbol.name}: ..")
                    case SymbolCategory.METHOD:
                        signatures.append(f"{symbol.symbol.name}(...)")
                    case _:
                        assert_never(symbol.symbol.symbol_type)
        return signatures

    def render(self) -> str:
        return UNCONSTRAINED_PROMPT_TEMPLATE.render(
            user_request=self.user_request,
            starting_symbols=self.starting_symbols,
            plan=self.plan,
            repo_tree=self.repo_tree,
            get_signatures=self.get_signatures,
            trim_blocks=True,
            lstrip_blocks=True,
        )


class ParsedStepFromResponse(BaseModel):
    step_number: int
    description: str


class ParsedResponse(BaseModel):
    parsed_steps: list[ParsedStepFromResponse]
    parsed_from: Choice
    thought: Optional[str]


class ParseError(Exception):
    """Raised when parsing the XML response fails"""

    pass


class UnconstrainedXmlOutputSuccessorFunction:
    def __init__(
        self,
        search_tool: CodeSearchTool,
        starting_symbols: list[Symbol],
        repo_tree: str,
        beam_width: int = 1,
        max_retries: int = 3,  # New parameter for configuring max retries
    ) -> None:
        """
        Parameters
        ----------
        search_tool: The search tool to use.
        starting_symbols: A list of symbols to give the LLM an idea of what functionality is available in the codebase.
        repo_tree: A string representation of the repository tree.
        beam_width: The number of beams to use.
        max_retries: Maximum number of additional attempts to make if parsing fails.
        """
        self.client = OpenAI()
        self.search_tool = search_tool
        self.beam_width = beam_width
        self.repo_tree = repo_tree
        self.starting_symbols = starting_symbols
        self.max_retries = max_retries

    def build_prompt_context(self, state: Node[PlanStep, GoalTestT]) -> PromptContext:
        return PromptContext(
            user_request=state.plan.user_query,
            starting_symbols=self.starting_symbols,
            plan=state.plan,
            repo_tree=self.repo_tree,
        )

    @staticmethod
    def parse_steps_from_choice(
        choice: Choice,
    ) -> tuple[list[ParsedStepFromResponse], Optional[str]]:
        content = choice.message.content

        # Check if content is wrapped in triple backticks
        assert content is not None
        if "```xml" in content and "```" in content[content.find("```xml") + 6 :]:
            start = content.find("```xml") + 6
            end = content.find("```", start)
            content = content[start:end].strip()

        try:
            root = ET.fromstring(f"<root>{content}</root>")
        except ET.ParseError as e:
            logger.warning(f"Failed to parse XML response: {e}")
            raise ParseError(f"XML parsing failed: {e}")

        thought = root.find("thought")
        plan = root.find("plan")
        if thought is None or plan is None:
            raise ParseError("Missing required thought or plan elements")

        target_xml_nodes = plan.findall(".//step")
        if not target_xml_nodes:
            raise ParseError("No steps found in plan")

        parsed_steps = []
        for xml_node in target_xml_nodes:
            try:
                step_number = int(xml_node.attrib["number"])
            except (TypeError, ValueError):
                raw_step_number = xml_node.attrib["number"]
                numeric_part = "".join(c for c in raw_step_number if c.isdigit())
                step_num_as_float = float(numeric_part)
                step_number = int(step_num_as_float)

            try:
                description = xml_node.find("description").text  # type: ignore
            except AttributeError:
                logger.warning(
                    f"No description found for step {step_number} in XML response. Skipping."
                )
                logger.warning(f"XML response: {content}")
                continue

            modification = ParsedStepFromResponse(
                step_number=step_number, description=description  # type: ignore
            )
            parsed_steps.append(modification)

        if not parsed_steps:
            raise ParseError("No valid steps could be parsed")

        return parsed_steps, thought.text

    def parse_steps_from_response(
        self,
        response: ChatCompletion,
    ) -> list[ParsedResponse]:
        responses: list[ParsedResponse] = []
        for choice in response.choices:
            try:
                parsed_steps, thought = self.parse_steps_from_choice(choice)
                responses.append(
                    ParsedResponse(
                        parsed_steps=parsed_steps,
                        parsed_from=choice,
                        thought=thought,
                    )
                )
            except ParseError as e:
                logger.warning(f"Skipping invalid response: {e}")
                continue
        return responses

    def __call__(
        self, state: Node[PlanStep, GoalTestT]
    ) -> Sequence[Node[PlanStep, GoalTestT]]:
        prompt_context = self.build_prompt_context(state)
        prompt = prompt_context.render()

        proposed_successors: list[ParsedResponse] = []
        retry_count = 0

        while (
            len(proposed_successors) < self.beam_width
            and retry_count < self.max_retries
        ):
            # Calculate how many more responses we need
            remaining = self.beam_width - len(proposed_successors)

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": prompt},
                ],
                n=remaining,
            )

            # Log the number of tokens in the prompt and response
            assert response.usage is not None
            logger.info(f"Prompt tokens: {response.usage.prompt_tokens}")
            logger.info(f"Response tokens: {response.usage.completion_tokens}")

            addtl_proposed_successors = self.parse_steps_from_response(response)
            proposed_successors.extend(addtl_proposed_successors)

            if len(addtl_proposed_successors) < remaining:
                retry_count += 1
                logger.warning(
                    f"Got {len(addtl_proposed_successors)} valid responses out of {remaining} requested. Retry {retry_count}/{self.max_retries}"
                )
            else:
                logger.info(
                    f"Got {len(addtl_proposed_successors)} valid responses out of {remaining} requested. Success!"
                )
                break

        new_nodes: list[Node[PlanStep, GoalTestT]] = []

        for proposed_successor in proposed_successors:
            # Ground each step in the proposed plan
            grounded_steps: list[PlanStep] = []
            for step in proposed_successor.parsed_steps:
                search_result = self.search_tool(step.description)
                grounded_step = PlanStep(
                    index=step.step_number,
                    content=step.description,
                    search_result=search_result,
                )
                grounded_steps.append(grounded_step)

            # Create new plan with all grounded steps
            edited_plan = Plan[PlanStep, GoalTestT](
                user_query=state.plan.user_query,
                steps=grounded_steps,
                reasoning=proposed_successor.thought,
            )
            new_node = Node(plan=edited_plan, parent=state, level=state.level + 1)
            new_nodes.append(new_node)

        return new_nodes


implements(SuccessorFunction)(UnconstrainedXmlOutputSuccessorFunction)
