# Most of this code is copied from https://github.com/JetBrains-Research/lca-baselines/tree/main/library_based_code_generation/src.
# I haven't made any semantic changes to the code.


from abc import ABC, abstractmethod
from textwrap import dedent

import tree_sitter_python as tspython
from colorama import Fore
from sacrebleu import CHRF
from tree_sitter import Language, Parser

PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)


class Queries:
    function_name_query = """
    (function_definition
    name: (identifier) @fun_name)
    """

    class_name_query = """
    (class_definition
    name: (identifier) @class_name)
    """

    called_function_name_query = """
    (call function: (identifier) @fun_name)

    (call function: (attribute attribute: (identifier) @fun_name))
    """

    all_identifiers_query = """
    (identifier) @identifier_name
    """

    imported_names_query = """
    (import_statement (dotted_name (identifier) @import_name))

    (import_from_statement (dotted_name (identifier) @import_name))


    (import_statement (aliased_import (identifier) @import_name))

    (import_from_statement (aliased_import (identifier) @import_name))
    """


class TokenColor:
    REGULAR = Fore.BLUE
    SEEN_IN_FILE = Fore.YELLOW
    SEEN_OUT_FILE = Fore.RED
    SEEN_BOTH = Fore.GREEN
    SEEN_NONE = Fore.MAGENTA
    NON_TREE = Fore.BLACK


class ParsedFile:
    def __init__(
        self,
        filepath: str | None = None,
        code: str | None = None,
        encoding: str = "utf8",
    ):
        self.filepath = filepath

        if code is None:
            assert self.filepath is not None, "Either filepath or code must be provided"
            try:
                with open(self.filepath) as code_file:
                    self.code = code_file.read()
            except UnicodeError:
                self.code = ""
            except FileNotFoundError:
                self.code = ""
        else:
            self.code = code

        self.encoding = encoding
        self.bytecode = bytes(self.code, self.encoding)

        self.tree = parser.parse(self.bytecode)
        self.root = self.tree.root_node

        self.function_names = self.make_query(Queries.function_name_query)
        self.class_names = self.make_query(Queries.class_name_query)
        self.called_functions = self.make_query(Queries.called_function_name_query)
        self.all_identifiers = self.make_query(Queries.all_identifiers_query)
        self.imported_names = self.make_query(Queries.imported_names_query)

        self.token_sequence = []
        self.collect_leaves(self.token_sequence)

    def collect_leaves(self, leaves: list, node=None):
        if node is None:
            node = self.root

        if len(node.children) == 0:
            leaves.append(node)
        else:
            for child in node.children:
                self.collect_leaves(leaves, child)

    def print_leaves(self, node=None, depth: int = 0):
        if node is None:
            node = self.root

        prefix = "--" * depth
        if len(node.children) == 0:
            print(prefix, node.type, "|", node.text.decode(self.encoding))  # type: ignore[attr-defined]
        else:
            print(prefix, node.type)
            for child in node.children:
                self.print_leaves(child, depth + 1)

    def make_query(self, query_string: str) -> list:
        query = PY_LANGUAGE.query(query_string)
        captures = query.captures(self.root)

        return self.query_to_set(captures)  # type: ignore[return-value]

    def query_to_set(self, captures: list) -> set:
        return {capture[0].text.decode("utf-8") for capture in captures}

    def __repr__(self):
        return dedent(
            f"""
            filepath: {self.filepath}
            function_names: {self.function_names}
            class_names: {self.class_names}
            called_functions: {self.called_functions}
            all_identifiers: {self.all_identifiers}
            imported_names: {self.imported_names}
        """,
        )

    def colored_code(self, other_identifiers: set):
        visited_identifiers = set()
        current_pos = 0
        result = ""
        for token in self.token_sequence:
            start_pos, end_pos = token.byte_range

            if current_pos < start_pos:
                result += TokenColor.NON_TREE + self.bytecode[
                    current_pos:start_pos
                ].decode(self.encoding)

            if token.type == "identifier":
                color = TokenColor.SEEN_NONE
                if token.text in visited_identifiers:
                    color = TokenColor.SEEN_IN_FILE
                    if token.text in other_identifiers:
                        color = TokenColor.SEEN_BOTH
                elif token.text in other_identifiers:
                    color = TokenColor.SEEN_OUT_FILE

                visited_identifiers.add(token.text)
                result += color + token.text.decode(self.encoding)
            else:
                result += TokenColor.REGULAR + token.text.decode(self.encoding)

            current_pos = end_pos

        return result + Fore.BLACK

    @staticmethod
    def filter_function_names(names):
        return {
            name
            for name in names
            if not (name.startswith("__") and name.endswith("__"))
            and not name == "super"
        }

    def clean_comments(self):
        start1 = bytes("'''", "utf8")
        start2 = bytes('"""', "utf8")

        comment_nodes = []

        def walk(node):
            if "comment" in node.type.lower():
                comment_nodes.append(node)
            elif node.type == "string" and (
                node.text.startswith(start1) or node.text.startswith(start2)
            ):
                comment_nodes.append(node)
            else:
                for child in node.children:
                    walk(child)

        walk(self.root)

        comment_positions = [(node.start_byte, node.end_byte) for node in comment_nodes]
        comment_positions.reverse()

        clean_code = self.code

        for start, end in comment_positions:
            clean_code = clean_code[:start] + clean_code[end:]

        return clean_code


class Metric(ABC):
    @abstractmethod
    def score(
        self,
        generated_file: str,
        reference_code: str,
        unique_apis: list[str],
    ) -> float:
        pass

    @abstractmethod
    def name(self) -> str:
        pass


class Overlap(Metric):
    def __init__(self):
        pass

    def score(
        self,
        generated_file: str,
        reference_code: str,
        unique_apis: list[str],
    ) -> float:
        parsed_generated_file = ParsedFile(code=generated_file)
        generated_function_calls = parsed_generated_file.called_functions
        guessed_apis = set(generated_function_calls) & set(unique_apis)
        return len(guessed_apis) / len(unique_apis)

    def name(self) -> str:
        return "API_recall"


class ChrF(Metric):
    def __init__(self):
        self.chrf = CHRF()

    def score(
        self,
        generated_file: str,
        reference_code: str,
        unique_apis: list[str],
    ) -> float:
        return self.chrf.sentence_score(generated_file, [reference_code]).score / 100

    def name(self) -> str:
        return "ChrF"
