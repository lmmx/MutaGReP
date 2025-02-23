import functools
from pathlib import Path
from threading import Lock
from typing import (Generator, Generic, Iterator, MutableMapping, Optional,
                    Protocol, Sequence, Type, TypeVar, Union, cast)

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel
from sqlalchemy import Column, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker


class LogOnce:
    """
    LogOnce ensures that a message is logged only once.

    When used as a context manager, multiple log calls can be made within the same context block.
    Any subsequent uses of the LogOnce context manager will no-op.

    Args:
        logger_name (str): Name of the logger to use. Defaults to the module's __name__.

    Usage examples:
        # Single context entry with multiple log calls
        log_once = LogOnce()
        with log_once:
            log_once("This message will be logged.")
            log_once("This message will also be logged.")

        # Subsequent context usage will no-op
        with log_once:
            log_once("This message will not be logged.")

        # Subsequent log calls will no-op
        log_once("This message will not be logged.")
    """

    def __init__(self, name: str):
        self._logged = False
        self._lock = Lock()
        self._is_context_manager = False
        self._logger = logger.bind(name=name)

    def __call__(self, *args, **kwargs):
        with self._lock:
            if not self._logged:
                self._logger.info(*args, **kwargs)
                if not self._is_context_manager:
                    self._logged = True

    def reset(self):
        with self._lock:
            self._logged = False
            self._is_context_manager = False

    def __enter__(self):
        with self._lock:
            self._is_context_manager = True
            return self

    def __exit__(self, exc_type, exc_value, traceback):
        with self._lock:
            self._logged = True
            self._is_context_manager = False
            self._is_context_manager = False


T = TypeVar("T", bound=BaseModel)


class PydanticJSONLinesWriter(Generic[T]):
    def __init__(self, file_path: str | Path):
        self.file_path = file_path

    def __call__(self, serializable: T):
        with open(self.file_path, "a") as f:
            f.write(serializable.model_dump_json() + "\n")

    def write_many(self, serializables: Sequence[T]):
        with open(self.file_path, "a") as f:
            for serializable in serializables:
                f.write(serializable.model_dump_json() + "\n")


class PydanticJSONLinesReader(Generic[T]):
    def __init__(self, file_path: str | Path, model: Type[T]):
        self.file_path = file_path
        self.model = model

    def __call__(self) -> Generator[T, None, None]:
        with open(self.file_path, "r") as f:
            for line in f:
                yield self.model.parse_raw(line)

    def read_all(self) -> list[T]:
        return list(self.__call__())


# TODO: Instead of doing it this way, we should save the models in something like
# Parquet or Arrow to save space. Base64 encoding the images makes them 33% larger.
class PydanticJSONLinesIO(Generic[T]):
    def __init__(self, file_path: str | Path, model: Type[T] = BaseModel):
        self.reader: PydanticJSONLinesReader[T] = PydanticJSONLinesReader(
            file_path, model
        )
        self.writer: PydanticJSONLinesWriter[T] = PydanticJSONLinesWriter(file_path)

    def read(self) -> Generator[T, None, None]:
        return self.reader()

    def write(self, serializable: T):
        self.writer(serializable)

    def read_all(self) -> list[T]:
        return list(self.reader())

    def write_batch(self, serializables: Sequence[T]):
        for serializable in serializables:
            self.writer(serializable)


class ImmutablePydanticJSONLinesCollection(Generic[T]):
    def __init__(self, file_path: str, model: Type[T]):
        self.io = PydanticJSONLinesIO(file_path, model)
        self.items = self.io.read_all()
        logger.info(
            "Loaded {} items of class {} from {}", len(self.items), model, file_path
        )

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int):
        return self.items[index]

    def __contains__(self, item: T):
        return item in self.items


def load_hydra_config_from_paths(
    config_name: str,
    config_path: str,
    overrides: Optional[list[str]] = None,
) -> DictConfig:
    hydra.initialize(config_path=config_path, version_base=None)
    config = hydra.compose(config_name=config_name, overrides=overrides or [])
    OmegaConf.resolve(config)
    logger.info("Loaded config from {} with overrides {}", config_path, overrides)
    logger.info("Config on next line:\n{}", OmegaConf.to_yaml(config))
    return config


def hydra_instantiate_with_logging(*args, **kwargs):
    logger.info("Instantiating with args={} and kwargs={}", args, kwargs)
    instantiated = hydra.utils.instantiate(*args, **kwargs)
    logger.info("Instantiated {}", instantiated)
    return instantiated


Base = declarative_base()


class KeyValueModel(Base):
    __tablename__ = "key_value_store"
    key = Column(String, primary_key=True, nullable=False)
    value = Column(String, nullable=False)


class SqliteKeyValueStore:
    def __init__(self, db_url: str = "sqlite:///key_value_store.db"):
        self.engine = create_engine(db_url, connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        self.Session = scoped_session(sessionmaker(bind=self.engine))

    def set(self, key: str, value: str):
        session = self.Session()
        try:
            existing_entry = session.query(KeyValueModel).filter_by(key=key).first()
            if existing_entry:
                existing_entry.value = value  # type: ignore
            else:
                new_entry = KeyValueModel(key=key, value=value)
                session.add(new_entry)
            session.commit()
        finally:
            session.close()

    def get(self, key: str) -> Optional[str]:
        session = self.Session()
        try:
            entry = session.query(KeyValueModel).filter_by(key=key).first()
            return cast(str, entry.value) if entry else None
        finally:
            session.close()

    def delete(self, key: str) -> None:
        session = self.Session()
        try:
            session.query(KeyValueModel).filter_by(key=key).delete()
            session.commit()
        finally:
            session.close()

    def keys(self) -> list[str]:
        session = self.Session()
        try:
            return cast(
                list[str], [entry.key for entry in session.query(KeyValueModel).all()]
            )
        finally:
            session.close()

    def values(self) -> list[str]:
        session = self.Session()
        try:
            return cast(
                list[str], [entry.value for entry in session.query(KeyValueModel).all()]
            )
        finally:
            session.close()

    def items(self) -> list[tuple[str, str]]:
        session = self.Session()
        try:
            return cast(
                list[tuple[str, str]],
                [
                    (entry.key, entry.value)
                    for entry in session.query(KeyValueModel).all()
                ],
            )
        finally:
            session.close()


class PydanticSqliteKeyValueStore(MutableMapping[str, T], Generic[T]):
    def __init__(self, model: Type[T], db_url: str = "sqlite:///key_value_store.db"):
        self.db_url = db_url
        self.store = SqliteKeyValueStore(db_url)
        self.model = model

    def __getitem__(self, key: str) -> T:
        value = self.store.get(key)
        if value is None:
            raise KeyError(f"Key {key} not found in store")
        return self.model.model_validate_json(value)

    def __setitem__(self, key: str, value: T) -> None:
        self.store.set(key, value.model_dump_json())

    def __delitem__(self, key: str) -> None:
        self.store.delete(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.store.keys())

    def __len__(self) -> int:
        return len(self.store.keys())


def strip_markdown_code_fence(text: str) -> str:
    lines = text.split("\n")
    if "```" in lines[0]:
        lines = lines[1:]
    if "```" in lines[-1]:
        lines = lines[:-1]
    return "\n".join(lines)
