import ast
import os
from collections import defaultdict
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import Literal, TypedDict

import networkx as nx
import pandas as pd
from loguru import logger
from pydantic import BaseModel, ConfigDict
from rich.console import Console
from rich.table import Table
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


class SymbolCategory(Enum):
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"


class Symbol(BaseModel):
    name: str
    docstring: str | None
    code: str | None
    filename: str
    filepath: str
    lineno: int
    symbol_type: SymbolCategory
    full_path: str

    # Make the class hashable.
    model_config = ConfigDict(frozen=True)


class SymbolDegree(TypedDict):
    in_degree: int
    out_degree: int


class RankableSymbol(BaseModel):
    symbol: Symbol
    usage_count: int
    reference_count: int
    inbound_calls: int
    outbound_calls: int
    topological_dependencies: int


class SymbolExtractor(ast.NodeVisitor):
    def __init__(self, file_path, file_content, base_directory):
        self.file_path = file_path
        self.file_content = file_content
        self.base_directory = base_directory
        self.symbols = []
        self.current_parents = []

        # Compute the module path from the file path
        relative_path = os.path.relpath(file_path, start=base_directory)
        self.module_path = os.path.splitext(relative_path)[0].replace(os.sep, ".")

    def visit_FunctionDef(self, node):
        docstring = ast.get_docstring(node)
        code = self.get_source_segment(node)
        parent_classes = [
            parent
            for parent in self.current_parents
            if isinstance(parent, ast.ClassDef)
        ]
        if parent_classes:
            # Construct the full path including the class name(s)
            full_path = (
                f"{self.module_path}."
                + ".".join(parent.name for parent in parent_classes)
                + f".{node.name}"
            )
            symbol_type = SymbolCategory.METHOD
        else:
            full_path = f"{self.module_path}.{node.name}"
            symbol_type = SymbolCategory.FUNCTION
        symbol = Symbol(
            name=node.name,
            full_path=full_path,
            docstring=docstring,
            code=code,
            filename=os.path.basename(self.file_path),
            filepath=self.file_path,
            lineno=node.lineno,
            symbol_type=symbol_type,
        )
        self.symbols.append(symbol)
        self.current_parents.append(node)
        self.generic_visit(node)
        self.current_parents.pop()

    def visit_ClassDef(self, node):
        full_path = f"{self.module_path}.{node.name}"
        docstring = ast.get_docstring(node)
        code = self.get_source_segment(node)
        symbol = Symbol(
            name=node.name,
            full_path=full_path,
            docstring=docstring,
            code=code,
            filename=os.path.basename(self.file_path),
            filepath=self.file_path,
            lineno=node.lineno,
            symbol_type=SymbolCategory.CLASS,
        )
        self.symbols.append(symbol)
        self.current_parents.append(node)
        self.generic_visit(node)
        self.current_parents.pop()

    def visit_Module(self, node):
        self.current_parents.append(node)
        self.generic_visit(node)
        self.current_parents.pop()

    def get_source_segment(self, node):
        return ast.get_source_segment(self.file_content, node)


def extract_symbols_from_file(
    file_path: Path | str,
    base_directory: Path,
) -> list[Symbol]:
    with open(file_path, encoding="utf-8") as file:
        file_content = file.read()
        tree = ast.parse(file_content, filename=file_path)
    extractor = SymbolExtractor(file_path, file_content, base_directory)
    extractor.visit(tree)
    return extractor.symbols


def extract_all_symbols_under_directory(directory: str | Path) -> list[Symbol]:
    if isinstance(directory, str):
        directory = Path(directory)
    python_files = get_all_pyfiles_under_directory(directory)
    all_symbols = []
    for py_file in tqdm(python_files, desc="Extracting symbols"):
        try:
            all_symbols.extend(extract_symbols_from_file(py_file, directory))
        except SyntaxError:
            with logging_redirect_tqdm():
                logger.error(f"Syntax error in {py_file}, skipping...")
    return all_symbols


def get_all_pyfiles_under_directory(directory: str | Path) -> list[str]:
    python_files = []
    for root, _, files in tqdm(os.walk(directory), desc="Finding .py files"):
        for file in files:
            if file.endswith(".py"):
                # Try parsing the file to catch syntax errors. If we cannot parse
                # the file, we skip it, because no static analysis can be done.
                with open(os.path.join(root, file), encoding="utf-8") as f:
                    try:
                        ast.parse(f.read())
                    except SyntaxError:
                        logger.warning(
                            "Skipping {} due to syntax error.",
                            os.path.join(root, file),
                        )
                        continue
                python_files.append(os.path.join(root, file))
    return python_files


def count_symbol_usage_frequency(
    symbols: list[Symbol],
    code_files: list[str],
    base_directory: str,
) -> dict[Symbol, int]:
    symbol_dict = {symbol.full_path: symbol for symbol in symbols}
    usage_count = defaultdict(int)
    known_symbols_usage_count: dict[Symbol, int] = defaultdict(int)

    for code_file in code_files:
        with open(code_file, encoding="utf-8") as f:
            content = f.read()
            tree = ast.parse(content, filename=code_file)
            import_paths = {}
            relative_path = os.path.relpath(code_file, start=base_directory)
            current_module = os.path.splitext(relative_path)[0].replace(os.sep, ".")

            class ImportVisitor(ast.NodeVisitor):
                def visit_Import(self, node):
                    for alias in node.names:
                        import_paths[alias.asname or alias.name] = alias.name

                def visit_ImportFrom(self, node):
                    module = node.module or ""
                    for alias in node.names:
                        full_name = f"{module}.{alias.name}"
                        import_paths[alias.asname or alias.name] = full_name

            ImportVisitor().visit(tree)

            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    full_path = import_paths.get(node.id, f"{current_module}.{node.id}")
                    # if full_path in symbol_dict:
                    #     usage_count[symbol_dict[full_path]] += 1
                    # usage_count[full_path] += 1
                    if full_path in symbol_dict:
                        known_symbols_usage_count[symbol_dict[full_path]] += 1
                    else:
                        usage_count[full_path] += 1
                elif isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Name):
                        value_name = import_paths.get(
                            node.value.id,
                            f"{current_module}.{node.value.id}",
                        )
                        full_path = f"{value_name}.{node.attr}"
                    if full_path in symbol_dict:
                        known_symbols_usage_count[symbol_dict[full_path]] += 1
                    else:
                        usage_count[full_path] += 1
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        full_path = import_paths.get(
                            node.func.id,
                            f"{current_module}.{node.func.id}",
                        )
                        if full_path in symbol_dict:
                            known_symbols_usage_count[symbol_dict[full_path]] += 1
                        else:
                            usage_count[full_path] += 1
                    elif isinstance(node.func, ast.Attribute):
                        if isinstance(node.func.value, ast.Name):
                            value_name = import_paths.get(
                                node.func.value.id,
                                f"{current_module}.{node.func.value.id}",
                            )
                            full_path = f"{value_name}.{node.func.attr}"
                            if full_path in symbol_dict:
                                known_symbols_usage_count[symbol_dict[full_path]] += 1
                            usage_count[full_path] += 1
    return known_symbols_usage_count


def count_symbol_references(
    symbols: list[Symbol],
    code_files: list[str],
    base_directory: str,
) -> dict[Symbol, int]:
    symbol_dict = {symbol.full_path: symbol for symbol in symbols}
    reference_counts = defaultdict(int)

    for code_file in code_files:
        with open(code_file, encoding="utf-8") as f:
            content = f.read()
            tree = ast.parse(content, filename=code_file)
            import_paths = {}
            relative_path = os.path.relpath(code_file, start=base_directory)
            current_module = os.path.splitext(relative_path)[0].replace(os.sep, ".")

            class ImportVisitor(ast.NodeVisitor):
                def visit_Import(self, node):
                    for alias in node.names:
                        import_paths[alias.asname or alias.name] = alias.name

                def visit_ImportFrom(self, node):
                    module = node.module or ""
                    for alias in node.names:
                        full_name = f"{module}.{alias.name}"
                        import_paths[alias.asname or alias.name] = full_name

            ImportVisitor().visit(tree)

            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        full_path = import_paths.get(
                            node.func.id,
                            f"{current_module}.{node.func.id}",
                        )
                        if full_path in symbol_dict:
                            reference_counts[symbol_dict[full_path]] += 1
                    elif isinstance(node.func, ast.Attribute):
                        if isinstance(node.func.value, ast.Name):
                            value_name = import_paths.get(
                                node.func.value.id,
                                f"{current_module}.{node.func.value.id}",
                            )
                            full_path = f"{value_name}.{node.func.attr}"
                            if full_path in symbol_dict:
                                reference_counts[symbol_dict[full_path]] += 1
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        full_name = f"{node.module}.{alias.name}"
                        if full_name in symbol_dict:
                            reference_counts[symbol_dict[full_name]] += 1
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in symbol_dict:
                            reference_counts[symbol_dict[alias.name]] += 1

    return reference_counts


def build_call_graph(
    symbols: list[Symbol],
    code_files: list[str],
    base_directory: str,
) -> nx.DiGraph:
    logger.debug("Starting to build call graph.")
    call_graph = nx.DiGraph()
    symbol_dict = {symbol.full_path: symbol for symbol in symbols}

    for code_file in code_files:
        try:
            with open(code_file, encoding="utf-8") as f:
                content = f.read()
            tree = ast.parse(content, filename=code_file)
            import_paths = {}
            relative_path = os.path.relpath(code_file, start=base_directory)
            current_module = os.path.splitext(relative_path)[0].replace(os.sep, ".")
            logger.debug(f"Parsing file: {code_file}, module: {current_module}")

            class ImportVisitor(ast.NodeVisitor):
                def visit_Import(self, node):
                    for alias in node.names:
                        import_paths[alias.asname or alias.name] = alias.name
                        logger.debug(
                            f"Found import: {alias.name} as {alias.asname or alias.name}",
                        )

                def visit_ImportFrom(self, node):
                    module = node.module or ""
                    for alias in node.names:
                        full_name = f"{module}.{alias.name}"
                        import_paths[alias.asname or alias.name] = full_name
                        logger.debug(
                            f"Found import from {module}: {alias.name} as {alias.asname or alias.name}",
                        )

            ImportVisitor().visit(tree)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_name = node.name
                    func_full_path = f"{current_module}.{func_name}"
                    call_graph.add_node(func_full_path)
                    logger.debug(f"Added function node: {func_full_path}")

                    for subnode in ast.walk(node):
                        if isinstance(subnode, ast.Call):
                            if isinstance(subnode.func, ast.Name):
                                callee_name = subnode.func.id
                                if callee_name in import_paths:
                                    callee_full_path = import_paths[callee_name]
                                else:
                                    callee_full_path = f"{current_module}.{callee_name}"
                                if callee_full_path in symbol_dict:
                                    call_graph.add_edge(
                                        func_full_path,
                                        callee_full_path,
                                    )
                                    logger.debug(
                                        f"Added edge from {func_full_path} to {callee_full_path}",
                                    )
                                else:
                                    logger.debug(
                                        f"Unresolved call: {callee_full_path} in {func_full_path}",
                                    )
                            elif isinstance(subnode.func, ast.Attribute):
                                if isinstance(subnode.func.value, ast.Name):
                                    value_name = import_paths.get(
                                        subnode.func.value.id,
                                        subnode.func.value.id,
                                    )
                                    callee_full_path = (
                                        f"{value_name}.{subnode.func.attr}"
                                    )
                                    if callee_full_path in symbol_dict:
                                        call_graph.add_edge(
                                            func_full_path,
                                            callee_full_path,
                                        )
                                        logger.debug(
                                            f"Added edge from {func_full_path} to {callee_full_path}",
                                        )
                                    else:
                                        logger.debug(
                                            f"Unresolved attribute call: {callee_full_path} in {func_full_path}",
                                        )
        except Exception as e:
            logger.error(f"Error processing file {code_file}: {e}")

    logger.debug("Finished building call graph.")
    return call_graph


def compute_symbol_dependencies(call_graph: nx.DiGraph):
    # Perform a topological sort
    topological_order = list(nx.topological_sort(call_graph))

    # Initialize the dependency count dictionary
    dependency_count = {node: 0 for node in call_graph.nodes()}

    # Traverse the nodes in reverse topological order to count dependencies
    for node in reversed(topological_order):
        for successor in call_graph.successors(node):
            dependency_count[node] += 1 + dependency_count[successor]

    # Sort the symbols based on the number of dependencies in descending order
    sorted_symbols = sorted(
        dependency_count.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return sorted_symbols


def compute_symbol_dependencies_with_cycles(call_graph: nx.DiGraph) -> dict[str, int]:
    # Find the strongly connected components (SCCs)
    sccs = list(nx.strongly_connected_components(call_graph))

    # Create a new graph where each SCC is a single node
    scc_graph = nx.DiGraph()
    scc_map = {}

    for i, scc in enumerate(sccs):
        scc_node = f"SCC_{i}"
        scc_graph.add_node(scc_node)
        for node in scc:
            scc_map[node] = scc_node

    for u, v in call_graph.edges():
        scc_u = scc_map[u]
        scc_v = scc_map[v]
        if scc_u != scc_v:
            scc_graph.add_edge(scc_u, scc_v)

    # Perform a topological sort on the SCC graph
    topological_order = list(nx.topological_sort(scc_graph))

    # Initialize the dependency count dictionary
    dependency_count = {node: 0 for node in scc_graph.nodes()}

    # Traverse the nodes in reverse topological order to count dependencies
    for node in reversed(topological_order):
        for successor in scc_graph.successors(node):
            dependency_count[node] += 1 + dependency_count[successor]

    # Map the dependency counts back to the original nodes
    original_dependency_count = {
        node: dependency_count[scc_map[node]] for node in call_graph.nodes()
    }

    # # Sort the symbols based on the number of dependencies in descending order
    # sorted_symbols = sorted(
    #     original_dependency_count.items(), key=lambda item: item[1], reverse=True
    # )

    # return {node: count for node, count in sorted_symbols}

    return original_dependency_count


def count_symbols_with_most_dependencies(
    symbols: list[Symbol],
    code_files: list[str],
    base_directory: str,
) -> dict[Symbol, int]:
    call_graph = build_call_graph(symbols, code_files, base_directory)
    dependency_counts: dict[str, int] = compute_symbol_dependencies_with_cycles(
        call_graph,
    )

    fullpath_to_symbol = {symbol.full_path: symbol for symbol in symbols}
    symbol_to_dependency_count = {
        fullpath_to_symbol[fullpath]: count
        for fullpath, count in dependency_counts.items()
        if fullpath in fullpath_to_symbol
    }

    return symbol_to_dependency_count


def determine_symbol_degree_from_call_graph(
    symbols: list[Symbol],
    code_files: list[str],
    base_directory: str,
) -> dict[Symbol, SymbolDegree]:
    call_graph = build_call_graph(symbols, code_files, base_directory)
    symbol_to_degrees: dict[Symbol, SymbolDegree] = {}

    # Identify entry points (nodes with no incoming edges)
    # for node in call_graph.nodes:
    #     if call_graph.in_degree(node) == 0:
    #         symbol_to_degrees[node] = True

    for symbol in symbols:
        full_path = symbol.full_path
        if full_path in call_graph:
            in_degree = call_graph.in_degree(full_path)
            out_degree = call_graph.out_degree(full_path)
            symbol_to_degrees[symbol] = {
                "in_degree": in_degree,
                "out_degree": out_degree,
            }
        else:
            symbol_to_degrees[symbol] = {"in_degree": 0, "out_degree": 0}
        # in_degree = call_graph.in_degree(full_path)
        # out_degree = call_graph.out_degree(full_path)
        # symbol_to_degrees[symbol] = {"in_degree": in_degree, "out_degree": out_degree}

    return symbol_to_degrees


def compute_rankable_symbols(directory: str | Path) -> list[RankableSymbol]:
    symbols = extract_all_symbols_under_directory(directory)
    code_files = get_all_pyfiles_under_directory(directory)
    usage_frequency = count_symbol_usage_frequency(symbols, code_files, str(directory))
    connectivity = count_symbol_references(symbols, code_files, str(directory))
    symbol_degrees = determine_symbol_degree_from_call_graph(
        symbols,
        code_files,
        str(directory),
    )
    topological_dependencies = count_symbols_with_most_dependencies(
        symbols,
        code_files,
        str(directory),
    )
    # high_level_api = determine_high_level_api_via_call_graph(symbols, code_files)

    rankable_symbols: list[RankableSymbol] = []
    for symbol in symbols:
        rankable_symbol = RankableSymbol(
            symbol=symbol,
            usage_count=usage_frequency.get(symbol, 0),
            reference_count=connectivity.get(symbol, 0),
            inbound_calls=symbol_degrees.get(symbol, {}).get("in_degree", 0),
            outbound_calls=symbol_degrees.get(symbol, {}).get("out_degree", 0),
            topological_dependencies=topological_dependencies.get(symbol, 0),
        )
        rankable_symbols.append(rankable_symbol)

    # ranked_symbols = sorted(
    #     rankable_symbols,
    #     key=lambda x: (x.outbound_calls, ),
    #     reverse=True,
    # )
    return rankable_symbols


# def rank_symbols_by_importance(directory: str | Path) -> List[Symbol]:
#     symbols = extract_all_symbols_under_directory(directory)
#     code_files = get_all_pyfiles_under_directory(directory)
#     usage_frequency = calculate_usage_frequency(symbols, code_files)
#     connectivity = calculate_connectivity(symbols, code_files)

#     for symbol in symbols:
#         symbol.usage_frequency = usage_frequency.get(symbol.name, 0)
#         symbol.connectivity = connectivity.get(symbol.name, 0)
#         symbol.is_high_level_api = high_level_api.get(symbol.name, False)

#     ranked_symbols = sorted(symbols, key=lambda x: (x.is_high_level_api, x.connectivity, x.usage_frequency), reverse=True)
#     return ranked_symbols


RankingByAttribute = Literal[
    "usage_count",
    "reference_count",
    "inbound_calls",
    "outbound_calls",
    "topological_dependencies",
]


def display_symbols(
    rankable_symbols: list[RankableSymbol],
    rank_by: RankingByAttribute,
    ascending: bool = False,
    format: Literal["pandas", "rich"] = "rich",
    exclude: Sequence[str] | None = (
        "docstring",
        "code",
        "lineno",
        "name",
        "filename",
        "filepath",
    ),
    length: int | None = 10,
):
    exclude = exclude or []
    valid_formats = ["pandas", "rich"]

    # Dynamically get rankable attributes from RankableSymbol fields
    rankable_attributes = [
        field for field in RankableSymbol.model_fields.keys() if field != "symbol"
    ]

    if rank_by not in rankable_attributes:
        raise ValueError(
            f"Invalid rank_by attribute {rank_by}. Must be one of {rankable_attributes}",
        )

    sorted_symbols = sorted(
        rankable_symbols,
        key=lambda x: getattr(x, rank_by),
        reverse=not ascending,
    )

    if length:
        sorted_symbols = sorted_symbols[:length]

    # Dynamically get columns from Symbol fields
    symbol_fields = list(Symbol.model_fields.keys())
    columns = [field for field in symbol_fields if field not in exclude]
    rankable_columns = rankable_attributes

    if format == "pandas":
        data = []
        for rs in sorted_symbols:
            symbol_data = {attr: getattr(rs.symbol, attr) for attr in columns}
            rankable_data = {attr: getattr(rs, attr) for attr in rankable_columns}
            data.append({**symbol_data, **rankable_data})

        df = pd.DataFrame(data)
        return df

    elif format == "rich":
        table = Table(title=f"Symbols Ranked by {rank_by}")
        for col in columns + rankable_columns:
            table.add_column(col)

        for rs in sorted_symbols:
            row = [str(getattr(rs.symbol, col)) for col in columns] + [
                str(getattr(rs, col)) for col in rankable_columns
            ]
            table.add_row(*row)

        console = Console()
        console.print(table)

    else:
        raise ValueError(f"Invalid format. Must be one of {valid_formats}")


def extract_symbol_signature(symbol: Symbol) -> str:
    """Extract a clean signature from a Symbol based on its category."""
    if not symbol.code:
        return f"{symbol.name} (no code available)"

    try:
        tree = ast.parse(symbol.code)
    except SyntaxError:
        return f"{symbol.name} (invalid syntax)"

    if symbol.symbol_type == SymbolCategory.FUNCTION:
        return _extract_function_signature(tree, symbol)
    elif symbol.symbol_type == SymbolCategory.METHOD:
        return _extract_method_signature(tree, symbol)
    elif symbol.symbol_type == SymbolCategory.CLASS:
        return _extract_class_signature(tree, symbol)

    return f"{symbol.name} (unknown symbol type)"


def _extract_function_signature(tree: ast.AST, symbol: Symbol) -> str:
    """Extract signature for a function, including docstring if available."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Get function signature
            args = _format_arguments(node.args)
            returns = _format_returns(node)

            # Get first sentence of docstring if available
            docstring = ast.get_docstring(node)
            doc_summary = f" - {docstring.split('.')[0]}." if docstring else ""

            return f"{node.name}({args}){returns}{doc_summary}"
    return symbol.name


def _extract_method_signature(tree: ast.AST, symbol: Symbol) -> str:
    """Extract signature for a method, including docstring if available."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Get method signature
            args = _format_arguments(node.args)
            returns = _format_returns(node)

            # Get first sentence of docstring if available
            docstring = ast.get_docstring(node)
            doc_summary = f" - {docstring.split('.')[0]}." if docstring else ""

            return f"{node.name}({args}){returns}{doc_summary}"
    return symbol.name


def _extract_class_signature(tree: ast.AST, symbol: Symbol) -> str:
    """Extract a class skeleton with method signatures."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Start with class definition and its bases
            bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
            base_str = f"({', '.join(bases)})" if bases else ""

            # Get first sentence of class docstring if available
            docstring = ast.get_docstring(node)
            doc_summary = f" - {docstring.split('.')[0]}." if docstring else ""

            result = [f"class {node.name}{base_str}{doc_summary}"]

            # Add constructor and method signatures
            methods = []
            for method in node.body:
                if isinstance(method, ast.FunctionDef):
                    args = _format_arguments(method.args)
                    returns = _format_returns(method)
                    methods.append(f"    def {method.name}({args}){returns}: ...")

            if methods:
                result.extend(methods)

            return "\n".join(result)
    return symbol.name


def _format_arguments(args: ast.arguments) -> str:
    """Format function/method arguments with type hints."""
    parts = []

    # Handle positional args
    for arg in args.args:
        arg_str = arg.arg
        if arg.annotation:
            # Use ast.unparse for all annotations to handle complex types
            arg_str += f": {ast.unparse(arg.annotation)}"
        parts.append(arg_str)

    # Handle *args if present
    if args.vararg:
        vararg_str = f"*{args.vararg.arg}"
        if args.vararg.annotation:
            vararg_str += f": {ast.unparse(args.vararg.annotation)}"
        parts.append(vararg_str)

    # Handle keyword-only args
    for arg in args.kwonlyargs:
        arg_str = arg.arg
        if arg.annotation:
            arg_str += f": {ast.unparse(arg.annotation)}"
        parts.append(arg_str)

    # Handle **kwargs if present
    if args.kwarg:
        kwarg_str = f"**{args.kwarg.arg}"
        if args.kwarg.annotation:
            kwarg_str += f": {ast.unparse(args.kwarg.annotation)}"
        parts.append(kwarg_str)

    return ", ".join(parts)


def _format_returns(node: ast.FunctionDef) -> str:
    """Format return type annotation if present."""
    if node.returns:
        return f" -> {ast.unparse(node.returns)}"
    return ""
