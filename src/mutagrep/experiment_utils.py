from pathlib import Path

from git import Repo
from loguru import logger


def get_experiment_path_from_script_name(script_name: str) -> Path:
    # Each experiments file is named something like
    # 001_skill_conditioned_gqa.py
    # the output should go in workspace/experiments__001_skill_conditioned_gqa
    output_dir_name = f"experiments__{Path(script_name).stem}"
    output_dir = Path("workspace") / output_dir_name
    return output_dir


def make_output_dir_for_experiment(script_name: str) -> Path:
    output_dir = get_experiment_path_from_script_name(script_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_output_dir_for_prev_experiment(experiment_identifier: str | int | Path) -> Path:
    """If a Path is given, return it directly.
    If an int is given, look in `experiments/` and return the path to the experiment starting with that number.
    Throw an error if there is not exactly one match.
    If an experiment name is given, return the path to the experiment with that name.
    """
    experiments_dir = Path("experiments")

    if isinstance(experiment_identifier, Path):
        return experiment_identifier

    if isinstance(experiment_identifier, int):
        matches = []
        for path in experiments_dir.iterdir():
            try:
                # Split at first underscore and try to parse the number
                experiment_num = int(path.name.split("_")[0])
                if experiment_num == experiment_identifier:
                    matches.append(path)
            except ValueError:
                # Skip files that don't start with a parseable number
                continue

        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one match for experiment number {experiment_identifier}, found {len(matches)}: {matches}",
            )
        script_name = matches[0].name
        experiment_path = get_experiment_path_from_script_name(script_name)
        if not experiment_path.exists():
            raise ValueError(
                f"No experiment found with number {experiment_identifier}.",
            )
        return experiment_path

    if isinstance(experiment_identifier, str):
        experiment_path = experiments_dir / experiment_identifier
        if not experiment_path.exists():
            raise ValueError(f"No experiment found with name {experiment_identifier}.")
        return get_experiment_path_from_script_name(experiment_path.name)

    raise TypeError("experiment_identifier must be a str, int, or Path.")


def write_current_commit_hash_to_file(output_dir: Path, repo_path: str = ".") -> None:
    repo = Repo(repo_path)
    commit_hash = repo.head.commit.hexsha
    with open(output_dir / "commit_hash.txt", "w") as f:
        f.write(commit_hash)


def make_output_dir_for_experiment_with_backup(script_name: str) -> Path:
    """Creates experiment directory, copies it to backup location, and creates a symlink.

    Args:
        script_name: Name of the experiment script

    Returns:
        Path to the symlinked directory in the original workspace location

    """
    logger.info(f"Setting up experiment directory for script: {script_name}")

    # Get the original workspace path
    original_dir = get_experiment_path_from_script_name(script_name)
    logger.info(f"Original directory path: {original_dir}")

    # Define backup location
    backup_root = Path("/nas-ssd2/zaidkhan/8-aluminium-chicken-workspace")
    backup_dir = backup_root / original_dir.name
    logger.info(f"Backup directory path: {backup_dir}")

    # Create the backup directory
    backup_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created backup directory at: {backup_dir}")

    # Remove original directory if it exists (needed before creating symlink)
    if original_dir.exists():
        if original_dir.is_symlink():
            logger.info(f"Removing existing symlink at: {original_dir}")
            original_dir.unlink()
        else:
            import shutil

            logger.info(
                f"Original directory exists and is not a symlink: {original_dir}",
            )
            # Copy contents to backup before removing
            if backup_dir.exists():
                logger.info("Merging contents with existing backup...")
                # Merge contents if backup already exists
                for item in original_dir.iterdir():
                    if (backup_dir / item.name).exists():
                        logger.debug(f"Skipping existing item: {item.name}")
                        continue
                    if item.is_dir():
                        logger.info(f"Copying directory: {item.name}")
                        shutil.copytree(item, backup_dir / item.name)
                    else:
                        logger.info(f"Copying file: {item.name}")
                        shutil.copy2(item, backup_dir / item.name)
            else:
                logger.info("Copying entire directory to backup location...")
                # Copy everything if backup doesn't exist
                shutil.copytree(original_dir, backup_dir)
            logger.info(f"Removing original directory: {original_dir}")
            shutil.rmtree(original_dir)

    # Create symlink from original location to backup
    logger.info(f"Creating symlink: {original_dir} -> {backup_dir}")
    original_dir.symlink_to(backup_dir, target_is_directory=True)

    logger.success("Setup complete!")
    return original_dir
