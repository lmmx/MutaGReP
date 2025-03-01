import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import jinja2
import numpy as np
import pandas as pd
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel, computed_field
from together import Together
from typing_extensions import assert_never

from mutagrep.coderec.v3.symbol_mining import Symbol, extract_symbol_signature
from mutagrep.longcodearena.scoring import ChrF, Overlap
from mutagrep.longcodearena.types import LongCodeArenaRecord
from mutagrep.plan_search.components import PlanStep
from mutagrep.utilities import strip_markdown_code_fence


def truncated_code_display(code: str) -> str:
    if len(code) > 1000:
        return code[:1000] + "\n # ... truncated"
    return code


PLAN_TO_CODE_TEMPLATE = jinja2.Template(
    """Your task is to write Python code that achieves a user query.
You will be provided a step-by-step plan for accomplishing the user query.
Use the plan to help you write code that accomplishes the user query.
Each step of the plan contains a list of suggested symbols to use in that step.
You will be provided the definition of each symbol mentioned in the plan.
{% if project_defined_elements is not none %}
You will also be provided a list of all symbols in the codebase.
{% endif %}
{% if repo_tree is not none %}
You will also be given a map of the codebase structure.
{% endif %}
You can use all of the above information to write code that accomplishes the user query.
{% if encourage_symbol_usage %}
It is important to stick to the plan as closely as possible and consider using the symbols provided for each step.
{% endif %}

Produce your output in the following format:
```python
# your code goes here
```
Do not include any other text in your output. Stick exactly to the required format.

{%- if repo_tree is not none -%}
# Repository Tree
{{ repo_tree }}
{%- endif -%}

{%- if project_defined_elements is not none -%}
# List of all symbols in the codebase
{%- for element in project_defined_elements %}
- {{ element }}
{%- endfor -%}
{%- endif -%}

# Code Definitions
```python
{% for symbol in all_symbols_used %}
# Filepath: {{ symbol.filename }}
# Python Path: {{ symbol.full_path }}
{{ code_display(symbol) }}
{% endfor %}
```

# User query
{{ user_query }}

# Step-by-step plan
{% for step in plan -%}
## Step {{ step.index }}
- {{ step.content }}
### Symbols
{% for symbol in step.search_result.instrumentation.symbols_considered -%}
- {{ symbol.symbol.full_path }}
{% endfor %}
{% endfor %}""",
    undefined=jinja2.StrictUndefined,
)


class CodeGenerationContext(BaseModel):
    user_query: str
    plan: list[PlanStep]
    all_symbols_used: list[Symbol]
    code_display_mode: Literal["signature", "full"] = "full"
    repo_tree: str | None = None
    project_defined_elements: list[str] | None = None
    encourage_symbol_usage: bool = False

    def code_display(self, symbol: Symbol) -> str:
        pair = (self.code_display_mode, symbol.code)
        match pair:
            case ("signature", _):
                return extract_symbol_signature(symbol)
            case ("full", None):
                return "# No code available"
            case ("full", code):
                return truncated_code_display(code)
            case _:
                assert_never(pair)

    def render(self) -> str:
        return PLAN_TO_CODE_TEMPLATE.render(
            repo_tree=self.repo_tree,
            project_defined_elements=self.project_defined_elements,
            user_query=self.user_query,
            plan=self.plan,
            all_symbols_used=self.all_symbols_used,
            encourage_symbol_usage=self.encourage_symbol_usage,
            code_display=self.code_display,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def save_context(self, path: Path) -> None:
        with open(path, "w") as f:
            f.write(self.render())


class ScoreBundle(BaseModel):
    scores: list[float]

    @computed_field
    @property
    def average_score(self) -> float:
        return float(np.mean(self.scores))

    @computed_field
    @property
    def median_score(self) -> float:
        return float(np.median(self.scores))

    @computed_field
    @property
    def spread(self) -> float:
        return float(np.abs(np.max(self.scores) - np.min(self.scores)))

    @computed_field
    @property
    def max_score(self) -> float:
        return float(np.max(self.scores))

    @computed_field
    @property
    def min_score(self) -> float:
        return float(np.min(self.scores))

    @computed_field
    @property
    def std_dev(self) -> float:
        return float(np.std(self.scores))


class CompletedCodeGenTask(BaseModel):
    record_idx: int
    repository_name: str
    code_snippets: list[str]
    context_tokens: int
    origin_plan_performance: float
    overlap_scores: ScoreBundle
    chrf_scores: ScoreBundle
    response_tokens: int | None = None

    @computed_field
    @property
    def plan_performance_delta(self) -> float:
        return self.overlap_scores.average_score - self.origin_plan_performance

    def human_readable_summary(self) -> str:
        return (
            f"Code Generation Results for {self.repository_name}:\n"
            f"  Overlap Scores:\n"
            f"    Average: {self.overlap_scores.average_score:.2f}\n"
            f"    Range:   {self.overlap_scores.min_score:.2f} - {self.overlap_scores.max_score:.2f} (spread: {self.overlap_scores.spread:.2f})\n"
            f"    StdDev:  {self.overlap_scores.std_dev:.2f}\n"
            f"  Performance:\n"
            f"    Plan Delta: {self.plan_performance_delta:+.2f}\n"
            f"    Context Tokens:     {self.context_tokens:,}\n"
            f"    Response Tokens:   {self.response_tokens:,}"
        )


def compute_score_bundles_from_code_snippets(
    code_snippets: list[str],
    record: LongCodeArenaRecord,
) -> tuple[ScoreBundle, ScoreBundle]:
    """Compute overlap and chrf scores for a list of code snippets."""
    overlap_scores: list[float] = []
    overlap_scorer = Overlap()
    for code_snippet in code_snippets:
        overlap_scores.append(
            overlap_scorer.score(
                generated_file=code_snippet,
                reference_code=record.clean_reference,
                unique_apis=record.unique_apis,
            ),
        )

    chrf_scores: list[float] = []
    chrf_scorer = ChrF()
    for code_snippet in code_snippets:
        chrf_scores.append(
            chrf_scorer.score(
                generated_file=code_snippet,
                reference_code=record.clean_reference,
                unique_apis=record.unique_apis,
            ),
        )

    return ScoreBundle(scores=overlap_scores), ScoreBundle(scores=chrf_scores)


class LongCodeArenaCodeGenTask(BaseModel):
    record_idx: int
    record: LongCodeArenaRecord
    output_directory: Path
    n_samples: int = 5
    context: CodeGenerationContext
    origin_plan_performance: float
    model: Literal[
        "gpt-4o",
        "gpt-4o-mini",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K",
        "Qwen/Qwen2.5-7B-Instruct-Turbo",
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct-Turbo",
        "deepseek-ai/DeepSeek-R1",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
        "meta-llama/Llama-3.2-3B-Instruct-Turbo",
        "o1-mini",
        "o1",
    ] = "gpt-4o"

    @classmethod
    def get_cached(
        cls,
        record_idx: int,
        output_directory: Path,
    ) -> CompletedCodeGenTask | None:
        if not (save_path := output_directory / f"{record_idx}.json").exists():
            return None
        return CompletedCodeGenTask.model_validate_json(save_path.read_text())

    @property
    def completion_marker(self) -> Path:
        return self.output_directory / f"{self.record_idx}.completed"

    def __call__(self) -> CompletedCodeGenTask:
        if self.completion_marker.exists():
            logger.info(
                f"Skipping task {self.record_idx} because it has already been completed",
            )
            return CompletedCodeGenTask.model_validate_json(
                self.task_completion_save_path.read_text(),
            )

        temperature = 0.7

        match self.model:
            case "gpt-4o":
                client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            case "o1-mini":
                client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
                temperature = 1.0  # o1-mini does not support temperature
            case "o1":
                client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
                temperature = 1.0  # o1 does not support temperature
            case "gpt-4o-mini":
                client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            case "meta-llama/Llama-3.3-70B-Instruct-Turbo":
                client = cast(OpenAI, Together(api_key=os.environ["TOGETHER_API_KEY"]))
            case "Qwen/Qwen2.5-7B-Instruct-Turbo":
                client = cast(OpenAI, Together(api_key=os.environ["TOGETHER_API_KEY"]))
            case "Qwen/Qwen2.5-Coder-32B-Instruct":
                client = cast(OpenAI, Together(api_key=os.environ["TOGETHER_API_KEY"]))
            case "Qwen/Qwen2.5-72B-Instruct-Turbo":
                client = cast(OpenAI, Together(api_key=os.environ["TOGETHER_API_KEY"]))
            case "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K":
                client = cast(OpenAI, Together(api_key=os.environ["TOGETHER_API_KEY"]))
            case "deepseek-ai/DeepSeek-R1":
                client = cast(OpenAI, Together(api_key=os.environ["TOGETHER_API_KEY"]))
            case "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free":
                client = cast(OpenAI, Together(api_key=os.environ["TOGETHER_API_KEY"]))
                temperature = 0.5
            case "meta-llama/Llama-3.2-3B-Instruct-Turbo":
                client = cast(OpenAI, Together(api_key=os.environ["TOGETHER_API_KEY"]))
            case _:
                assert_never(self.model)

        prompt = self.context.render()

        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            n=self.n_samples,
        )

        assert response.usage is not None
        context_tokens = response.usage.prompt_tokens
        response_tokens = response.usage.completion_tokens

        code_snippets: list[str] = []
        for choice in response.choices:
            assert choice.message.content is not None
            code_snippets.append(strip_markdown_code_fence(choice.message.content))

        overlap_scores, chrf_scores = compute_score_bundles_from_code_snippets(
            code_snippets=code_snippets,
            record=self.record,
        )

        task_completion = CompletedCodeGenTask(
            record_idx=self.record_idx,
            code_snippets=code_snippets,
            overlap_scores=overlap_scores,
            chrf_scores=chrf_scores,
            context_tokens=context_tokens,
            response_tokens=response_tokens,
            repository_name=self.record.repo_name,
            origin_plan_performance=self.origin_plan_performance,
        )

        logger.info(task_completion.human_readable_summary())

        with open(self.task_completion_save_path, "w") as f:
            f.write(task_completion.model_dump_json())

        # Write the code snippets to a file
        code_snippets_dir = self.output_directory / f"{self.record_idx}_code"
        code_snippets_dir.mkdir(parents=True, exist_ok=True)
        for i, code_snippet in enumerate(code_snippets):
            with open(code_snippets_dir / f"sample_{i}.py", "w") as f:
                f.write(code_snippet)

        self.completion_marker.touch()

        return task_completion

    @property
    def task_completion_save_path(self) -> Path:
        return self.output_directory / f"{self.record_idx}.json"


@dataclass
class CodeGenerationReport:
    markdown_report: str
    dataframe: pd.DataFrame
    report_save_name: str = "report.md"
    metrics_save_name: str = "metrics.csv"

    def save(self, output_dir: Path) -> None:
        with open(output_dir / self.report_save_name, "w") as f:
            f.write(self.markdown_report)
        self.dataframe.to_csv(output_dir / self.metrics_save_name, index=False)


def generate_report(
    tasks: Sequence[CompletedCodeGenTask],
) -> CodeGenerationReport:
    # Create DataFrame with basic stats
    df = pd.DataFrame(
        [
            {
                "task_idx": task.record_idx,
                "repository": task.repository_name,
                "n_samples": len(task.code_snippets),
                "context_tokens": task.context_tokens,
                "average_score": task.overlap_scores.average_score,
                "median_score": task.overlap_scores.median_score,
                "spread": task.overlap_scores.spread,
                "max_score": task.overlap_scores.max_score,
                "min_score": task.overlap_scores.min_score,
                "std_dev": task.overlap_scores.std_dev,
                "plan_performance_delta": task.plan_performance_delta,
            }
            for task in tasks
        ],
    )

    # Sort by median score
    df = df.sort_values("median_score", ascending=False)

    # Compute overall statistics across all scores
    all_scores = [score for task in tasks for score in task.overlap_scores.scores]
    overall_avg = np.mean(all_scores)
    overall_median = np.median(all_scores)
    overall_std = np.std(all_scores)
    overall_delta_avg = np.mean([task.plan_performance_delta for task in tasks])
    max_score_average = np.mean([task.overlap_scores.max_score for task in tasks])
    min_score_average = np.mean([task.overlap_scores.min_score for task in tasks])
    chrf_average = np.mean([task.chrf_scores.average_score for task in tasks])
    chrf_max_score_average = np.mean([task.chrf_scores.max_score for task in tasks])
    chrf_min_score_average = np.mean([task.chrf_scores.min_score for task in tasks])

    # Prepare markdown report
    report = [
        "# Code Generation Metrics Report\n",
        f"Overall average score: {overall_avg:.3f}  ",
        f"Overall max score: {max_score_average:.3f}  ",
        f"Overall min score: {min_score_average:.3f}  ",
        f"Overall median score: {overall_median:.3f}  ",
        f"Overall standard deviation: {overall_std:.3f}  ",
        f"Overall average performance delta: {overall_delta_avg:.3f}\n",
        f"Overall average ChrF score: {chrf_average:.3f}  ",
        f"Overall max ChrF score: {chrf_max_score_average:.3f}  ",
        f"Overall min ChrF score: {chrf_min_score_average:.3f}  ",
        "\n## Top 5 Repositories by Median Score",
        df.nlargest(5, "median_score")[["repository", "median_score"]].to_markdown(),
        "\n## Bottom 5 Repositories by Median Score",
        df.nsmallest(5, "median_score")[["repository", "median_score"]].to_markdown(),
        "\n## Top 5 Repositories by Performance Delta",
        df.nlargest(5, "plan_performance_delta")[
            ["repository", "plan_performance_delta"]
        ].to_markdown(),
        "\n## Bottom 5 Repositories by Performance Delta",
        df.nsmallest(5, "plan_performance_delta")[
            ["repository", "plan_performance_delta"]
        ].to_markdown(),
        "\n## Top 5 Repositories by Score Spread",
        df.nlargest(5, "spread")[["repository", "spread"]].to_markdown(),
        "\n## Bottom 5 Repositories by Score Spread",
        df.nsmallest(5, "spread")[["repository", "spread"]].to_markdown(),
        "\n## Top 5 Individual Scores",
        df.nlargest(5, "max_score")[["repository", "max_score"]].to_markdown(),
        "\n## Bottom 5 Individual Scores",
        df.nsmallest(5, "min_score")[["repository", "min_score"]].to_markdown(),
    ]

    return CodeGenerationReport(
        markdown_report="\n\n".join(report),
        dataframe=df,
    )
