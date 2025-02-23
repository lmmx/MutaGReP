from pydantic import BaseModel


class LongCodeArenaRecord(BaseModel):
    repo_full_name: str
    repo_name: str
    repo_owner: str
    instruction: str
    reference: str
    clean_reference: str
    path_to_reference_file: str
    path_to_examples_folder: str
    n_unique_apis: int
    unique_apis: list[str]
    project_defined_elements: list[str]
    api_calls: list[str]
    internal_apis: list[str]
