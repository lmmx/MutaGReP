from mutagrep.coderec.v3.symbol_mining import (
    Symbol,
    SymbolCategory,
    extract_symbol_signature,
)


def create_symbol(code: str, symbol_type: SymbolCategory, name: str) -> Symbol:
    """Helper function to create a Symbol instance for testing."""
    return Symbol(
        name=name,
        code=code,
        docstring=None,
        filename="test.py",
        filepath="test.py",
        lineno=1,
        symbol_type=symbol_type,
        full_path=f"test.{name}",
    )


def test_function_signature():
    code = """
def calculate_sum(a: int, b: int) -> int:
    return a + b
"""
    symbol = create_symbol(code, SymbolCategory.FUNCTION, "calculate_sum")
    signature = extract_symbol_signature(symbol)
    assert signature == "calculate_sum(a: int, b: int) -> int"


def test_method_signature():
    code = """
def multiply(self, a: int, b: int) -> int:
    return a * b
"""
    symbol = create_symbol(code, SymbolCategory.METHOD, "multiply")
    signature = extract_symbol_signature(symbol)
    assert signature == "multiply(self, a: int, b: int) -> int"


def test_class_signature():
    code = """
class Calculator:
    def __init__(self, model: str, precision: int):
        self.model = model
        self.precision = precision

    def add(self):
        pass

    def subtract(self):
        pass

    def multiply(self):
        pass

    def divide(self):
        pass
"""
    symbol = create_symbol(code, SymbolCategory.CLASS, "Calculator")
    signature = extract_symbol_signature(symbol)
    expected = """class Calculator
    def __init__(self, model: str, precision: int): ...
    def add(self): ...
    def subtract(self): ...
    def multiply(self): ...
    def divide(self): ..."""
    assert signature == expected


def test_function_with_docstring():
    code = '''
def process_data(data: list[str]) -> dict[str, int]:
    """Process the input data and return frequency counts. This is a longer description."""
    return {}
'''
    symbol = create_symbol(code, SymbolCategory.FUNCTION, "process_data")
    signature = extract_symbol_signature(symbol)
    assert (
        signature
        == "process_data(data: list[str]) -> dict[str, int] - Process the input data and return frequency counts."
    )


def test_class_with_inheritance_and_docstring():
    code = '''
class AdvancedCalculator(Calculator):
    """A more sophisticated calculator with advanced operations. Includes scientific functions."""
    def __init__(self, model: str, precision: int, scientific: bool = False):
        super().__init__(model, precision)
        self.scientific = scientific

    def power(self, base: float, exponent: float) -> float:
        """Calculate base raised to exponent."""
        return base ** exponent
'''
    symbol = create_symbol(code, SymbolCategory.CLASS, "AdvancedCalculator")
    signature = extract_symbol_signature(symbol)
    expected = """class AdvancedCalculator(Calculator) - A more sophisticated calculator with advanced operations.
    def __init__(self, model: str, precision: int, scientific: bool): ...
    def power(self, base: float, exponent: float) -> float: ..."""
    assert signature == expected


def test_function_with_complex_types():
    code = """
def process_batch(items: list[dict[str, Any]], callback: Callable[[str], None] | None = None) -> tuple[list[str], int]:
    return [], 0
"""
    symbol = create_symbol(code, SymbolCategory.FUNCTION, "process_batch")
    signature = extract_symbol_signature(symbol)
    assert (
        signature
        == "process_batch(items: list[dict[str, Any]], callback: Callable[[str], None] | None) -> tuple[list[str], int]"
    )


def test_invalid_code():
    code = "def broken_function(x: This is not valid Python"
    symbol = create_symbol(code, SymbolCategory.FUNCTION, "broken_function")
    signature = extract_symbol_signature(symbol)
    assert signature == "broken_function (invalid syntax)"


def test_empty_code():
    symbol = create_symbol("", SymbolCategory.FUNCTION, "empty_function")
    signature = extract_symbol_signature(symbol)
    assert signature == "empty_function (no code available)"
