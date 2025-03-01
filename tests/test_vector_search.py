from pathlib import Path

from pydantic import BaseModel

from mutagrep.vector_search import (
    Embeddable,
    PydanticLancedbVectorDatabase,
    PydanticQdrantVectorDatabase,
    QDrantVectorDatabase,
)


class Animal(BaseModel):
    name: str
    sound: str
    vertebrate: bool


def test_qdrant_vector_database():
    db = QDrantVectorDatabase()
    docs = [
        {"key": "I am addicted to sponge cake.", "payload": {}},
        {"key": "There are 18 horses outside my home.", "payload": {}},
    ]
    db.insert(docs)
    results = db.search("I am addicted to chocolate cake.", limit=1)

    assert len(results) == 1
    assert results[0].payload == docs[0]


def test_pydantic_vector_database():
    backing_db = QDrantVectorDatabase()
    db = PydanticQdrantVectorDatabase(backing_db, Animal)
    embeddables = [
        Embeddable(
            key="Dogs enjoy playing with bones.",
            payload=Animal(name="dog", sound="woof", vertebrate=True),
        ),
        Embeddable(
            key="My cat Mittens was capable of running at 20 mph.",
            payload=Animal(name="cat", sound="meow", vertebrate=True),
        ),
        Embeddable(
            key="Worms are an important part of the food chain.",
            payload=Animal(name="worm", sound="squirm", vertebrate=False),
        ),
    ]
    db.insert(embeddables)
    results = db.search("Wombats enjoy playing with bones.", limit=1)
    assert len(results) == 1
    assert results[0].payload == embeddables[0].payload


def test_lancedb_vector_database(tmp_path: Path):
    db = PydanticLancedbVectorDatabase(str(tmp_path / "test.lancedb"), Animal)
    embeddables = [
        Embeddable(
            key="Dogs enjoy playing with bones.",
            payload=Animal(name="dog", sound="woof", vertebrate=True),
        ),
        Embeddable(
            key="My cat Mittens was capable of running at 20 mph.",
            payload=Animal(name="cat", sound="meow", vertebrate=True),
        ),
        Embeddable(
            key="Worms are an important part of the food chain.",
            payload=Animal(name="worm", sound="squirm", vertebrate=False),
        ),
    ]
    db.insert(embeddables)
    results = db.search("Wombats enjoy playing with bones.", limit=3)
    assert results[0].payload == embeddables[0].payload
