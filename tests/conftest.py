import os
from unittest.mock import MagicMock

import hydra
import pytest

TEST_ROOT = os.path.abspath(os.path.dirname(__file__))


@pytest.fixture
def mock_openai_client_factory():
    def factory(response_content):
        # Create the nested structure using MagicMock
        mock_response = MagicMock()
        mock_response.choices[0].message.content = response_content

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        return mock_client

    return factory


@pytest.fixture(autouse=True, scope="function")
def reinitialize_hydra():
    yield
    hydra.core.global_hydra.GlobalHydra.instance().clear()  # type: ignore
