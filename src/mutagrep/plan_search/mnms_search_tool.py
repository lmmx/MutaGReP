import instructor
import jinja2
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from mutagrep.coderec.v3.symbol_mining import Symbol, SymbolCategory
from mutagrep.plan_search.symbol_retrievers.bm25_simple import Bm25SymbolRetriever

from .domain_models import CodeSearchTool, CodeSearchToolOutput
from .typing_utils import implements

MNMS_TOOLS = [
    {
        "name": "text_generation",
        "signature": "text_generation(text: str) -> dict",
        "filepath": "tool_api.py",
        "full_path": "tool_api.text_generation",
        "docstring": "It takes an input text prompt and outputs a text that is most likely to follow the input text.",
    },
    {
        "name": "text_summarization",
        "signature": "text_summarization(text: str) -> dict",
        "filepath": "tool_api.py",
        "full_path": "tool_api.text_summarization",
        "docstring": "It takes a paragraph of text and summarizes into a few sentences.",
    },
    {
        "name": "text_classification",
        "signature": "text_classification(text: str) -> dict",
        "filepath": "tool_api.py",
        "full_path": "tool_api.text_classification",
        "docstring": "It takes a text and classifies it into a category in the model's vocaburary (e.g. positive or negative based on its sentiment).",
    },
    {
        "name": "question_answering",
        "signature": "question_answering(question: str, text: str) -> dict",
        "filepath": "tool_api.py",
        "full_path": "tool_api.question_answering",
        "docstring": "It takes a text and a question, and outputs an answer to that question based on the text.",
    },
    {
        "name": "automatic_speech_recognition",
        "signature": "automatic_speech_recognition(audio: str) -> dict",
        "filepath": "tool_api.py",
        "full_path": "tool_api.automatic_speech_recognition",
        "docstring": "It takes an audio file and produces a transcription of the audio.",
    },
    {
        "name": "image_generation",
        "signature": "image_generation(text: str) -> dict",
        "filepath": "tool_api.py",
        "full_path": "tool_api.image_generation",
        "docstring": "It takes a text prompt and generates an image that matches the text description.",
    },
    {
        "name": "image_captioning",
        "signature": "image_captioning(image: Image.Image) -> dict",
        "filepath": "tool_api.py",
        "full_path": "tool_api.image_captioning",
        "docstring": "It takes an image and generates a text caption of the image.",
    },
    {
        "name": "image_editing",
        "signature": "image_editing(image: Image.Image, prompt: str) -> dict",
        "filepath": "tool_api.py",
        "full_path": "tool_api.image_editing",
        "docstring": "It takes an image and a text prompt and outputs a new image based on the text.",
    },
    {
        "name": "image_classification",
        "signature": "image_classification(image: Image.Image) -> dict",
        "filepath": "tool_api.py",
        "full_path": "tool_api.image_classification",
        "docstring": "It takes an image and classifies the subject in the image into a category such as cat or dog.",
    },
    {
        "name": "visual_question_answering",
        "signature": "visual_question_answering(image: Image.Image, question: str) -> dict",
        "filepath": "tool_api.py",
        "full_path": "tool_api.visual_question_answering",
        "docstring": "It takes an image and a question about the image, and generates an answer to the question.",
    },
    {
        "name": "object_detection",
        "signature": "object_detection(image: Image.Image) -> dict",
        "filepath": "tool_api.py",
        "full_path": "tool_api.object_detection",
        "docstring": "It takes an image and outputs rectangular bounding boxes of objects detected in the image.",
    },
    {
        "name": "image_segmentation",
        "signature": "image_segmentation(image: Image.Image) -> dict",
        "filepath": "tool_api.py",
        "full_path": "tool_api.image_segmentation",
        "docstring": "It takes an image, segments it into different parts, and outputs segmentation masks of any shape for the parts.",
    },
    {
        "name": "optical_character_recognition",
        "signature": "optical_character_recognition(image: Union[str, Image.Image]) -> dict",
        "filepath": "tool_api.py",
        "full_path": "tool_api.optical_character_recognition",
        "docstring": "It takes an image and outputs recognized texts in the image.",
    },
    {
        "name": "image_crop",
        "signature": "image_crop(image: Image.Image, object: dict) -> dict",
        "filepath": "tool_api.py",
        "full_path": "tool_api.image_crop",
        "docstring": "It takes an image and 4 numbers representing the coordinates of a bounding box and crops the image to the region within the box.",
    },
    {
        "name": "image_crop_left",
        "signature": "image_crop_left(image: Image.Image) -> dict",
        "filepath": "tool_api.py",
        "full_path": "tool_api.image_crop_left",
        "docstring": "It takes an image, crops and keeps the left part of the image.",
    },
    {
        "name": "image_crop_right",
        "signature": "image_crop_right(image: Image.Image) -> dict",
        "filepath": "tool_api.py",
        "full_path": "tool_api.image_crop_right",
        "docstring": "It takes an image, crops and keeps the right part of the image.",
    },
    {
        "name": "image_crop_top",
        "signature": "image_crop_top(image: Image.Image) -> dict",
        "filepath": "tool_api.py",
        "full_path": "tool_api.image_crop_top",
        "docstring": "It takes an image, crops and keeps the top part of the image.",
    },
    {
        "name": "image_crop_bottom",
        "signature": "image_crop_bottom(image: Image.Image) -> dict",
        "filepath": "tool_api.py",
        "full_path": "tool_api.image_crop_bottom",
        "docstring": "It takes an image, crops and keeps the bottom part of the image.",
    },
    {
        "name": "background_blur",
        "signature": "background_blur(image: Image.Image, object: dict) -> dict",
        "filepath": "tool_api.py",
        "full_path": "tool_api.background_blur",
        "docstring": "It takes an image and one or multiple objects in the foreground, and returns an image where the backgroud is blurred.",
    },
    {
        "name": "color_pop",
        "signature": "color_pop(image: Image.Image, object: dict) -> dict",
        "filepath": "tool_api.py",
        "full_path": "tool_api.color_pop",
        "docstring": "It takes an image and one or multiple objects, and returns an image where only the object is colored and the rest is black and white.",
    },
    {
        "name": "count",
        "signature": "count(objects: list) -> dict",
        "filepath": "tool_api.py",
        "full_path": "tool_api.count",
        "docstring": "It takes a list of objects and returns the count of the objects.",
    },
    {
        "name": "tag",
        "signature": "tag(image: Image.Image, objects: list) -> dict",
        "filepath": "tool_api.py",
        "full_path": "tool_api.tag",
        "docstring": "It takes an image and a list of objects with their bounding boxes and classes, and tags all the objects.",
    },
    {
        "name": "emoji",
        "signature": "emoji(image: Image.Image, object: dict, emoji: str) -> dict",
        "filepath": "tool_api.py",
        "full_path": "tool_api.emoji",
        "docstring": "It takes an image and the bounding box coordinates of one or multiple objects, and replaces the object with an emoji (e.g. angry/flushed/crying/dizzy/sleepy/grimacing/kissing/smiling_face, alien, ghost, goblin etc).",
    },
    {
        "name": "select_object",
        "signature": "select_object(objects: list, object_name: str) -> dict",
        "filepath": "tool_api.py",
        "full_path": "tool_api.select_object",
        "docstring": "It takes a list of objects, and selects the object based on the input object name.",
    },
    {
        "name": "get_date_fact",
        "signature": "get_date_fact(date: str) -> dict",
        "filepath": "tool_api.py",
        "full_path": "tool_api.get_date_fact",
        "docstring": "It provides interesting facts about dates.",
    },
    {
        "name": "get_year_fact",
        "signature": "get_year_fact(year: str) -> dict",
        "filepath": "tool_api.py",
        "full_path": "tool_api.get_year_fact",
        "docstring": "It provides interesting facts about years.",
    },
    {
        "name": "get_math_fact",
        "signature": "get_math_fact(number: str) -> dict",
        "filepath": "tool_api.py",
        "full_path": "tool_api.get_math_fact",
        "docstring": "It provides interesting math facts about numbers.",
    },
    {
        "name": "get_trivia_fact",
        "signature": "get_trivia_fact(number: str) -> dict",
        "filepath": "tool_api.py",
        "full_path": "tool_api.get_trivia_fact",
        "docstring": "It provides interesting trivia facts about number.",
    },
    {
        "name": "love_calculator",
        "signature": "love_calculator(first_name: str, second_name: str) -> dict",
        "filepath": "tool_api.py",
        "full_path": "tool_api.love_calculator",
        "docstring": "Enter your name and the name of your partner/lover/crush to find Love compatibility & chances of successful love relationship.",
    },
    {
        "name": "get_location",
        "signature": "get_location(city: str) -> dict",
        "filepath": "tool_api.py",
        "full_path": "tool_api.get_location",
        "docstring": "Convert a city name or address to geographical coordinates using OpenStreetMap's Nominatim API.",
    },
    {
        "name": "search_movie",
        "signature": "search_movie(movie_title: str, movie_year: str) -> dict",
        "filepath": "tool_api.py",
        "full_path": "tool_api.search_movie",
        "docstring": "Retrieve basic movie information, including title, year, genre, and director.",
    },
    {
        "name": "get_weather",
        "signature": "get_weather(lon: str, lat: str) -> dict",
        "filepath": "tool_api.py",
        "full_path": "tool_api.get_weather",
        "docstring": "Provides weather forecast data based on specific geographical coordinates.",
    },
    {
        "name": "wikipedia_simple_search",
        "signature": "wikipedia_simple_search(text: str) -> dict",
        "filepath": "tool_api.py",
        "full_path": "tool_api.wikipedia_simple_search",
        "docstring": "Perform a basic search query on Wikipedia to retrieve a summary of the most relevant page.",
    },
]


select_relevant_functions_prompt = jinja2.Template(
    """You are an expert Python developer. Given a intention, select the function that best satisfies the intention.

Here is the list of functions:

{% for tool in tools %}
- {{ tool.name }}: {{ tool.docstring }}
{% endfor %}

Given the intention: {{ intention }}, make a decision.
If there is a function that could plausibly satisfy the intention, select it and provide a justification for why that function is the best choice.
If there is no function that could plausibly satisfy the intention, find the closest function (if any) and provide a justification for why the intention is not satisfied by the available functions.
Be generous in your interpretation of what a function could plausibly satisfy the intention.
Don't leave the "instrumentation" field empty.
""",
    undefined=jinja2.StrictUndefined,
)


def build_retriever_for_mnms() -> Bm25SymbolRetriever:
    symbol_corpus: list[Symbol] = [
        Symbol(
            name=tool["name"],
            docstring=tool["docstring"],
            code=None,
            filename=tool["filepath"],
            filepath=tool["full_path"],
            lineno=0,
            symbol_type=SymbolCategory.FUNCTION,
            full_path=tool["full_path"],
        )
        for tool in MNMS_TOOLS
    ]
    retriever = Bm25SymbolRetriever.build_from_symbol_sequence(symbol_corpus)
    return retriever


class MnmsSimpleCodeSearchTool:
    def __init__(self) -> None:
        self.client = instructor.from_openai(OpenAI())

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    def __call__(self, intention: str) -> CodeSearchToolOutput:
        prompt = select_relevant_functions_prompt.render(
            intention=intention,
            tools=MNMS_TOOLS,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_model=CodeSearchToolOutput,
        )
        return response


implements(CodeSearchTool)(MnmsSimpleCodeSearchTool)


# def search_tool(intention: str) -> SearchToolOutput:
#     prompt = select_relevant_functions_prompt.render(
#         intention=intention, tools=MNMS_TOOLS, trim_blocks=True, lstrip_blocks=True
#     )
#     client = instructor.from_openai(OpenAI())
#     return client.chat.completions.create(
#         model="gpt-4o",
#         messages=[{"role": "user", "content": prompt}],
#         response_model=SearchToolOutput,
#     )
