from unittest.mock import MagicMock


class MockVllmLLMInstance:
    def __init__(self, *args, **kwargs):
        pass

    def generate(self, *args, **kwargs):
        return [MagicMock(outputs=[MagicMock(text="<enrichment text goes here>")])]


MockVllmLORAInstance = MagicMock
MockSamplingParams = MagicMock
