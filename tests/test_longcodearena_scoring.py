from mutagrep.longcodearena.scoring import Overlap


def test_scoring():
    reference_code = """
from foolib import foo1,foo2,FooClass

foovar = foo1()
foo2()
foo_instance = FooClass()
foo_instance.foo_method()"""

    unique_apis = ["foo1", "foo2", "FooClass", "foo_method"]

    generated_code = """
from foolib import foo1,foo2,FooClass

foovar = foo1()
foo2()
foo_instance = FooClass()
foo_instance.foo_method()"""

    metric = Overlap()
    score = metric.score(generated_code, reference_code, unique_apis)
    assert score == 1.0


def test_scoring_handles_fully_qualified_names():
    reference_code = """
from foolib import foo1,foo2,FooClass

foovar = foo1()
foo2()
foo_instance = FooClass()
foo_instance.foo_method()"""

    unique_apis = ["foo1", "foo2", "FooClass", "foo_method"]

    generated_code = """
import foolib
from foolib import foo

foovar = foolib.foo1()
foo.foo2()
foo_instance = foolib.FooClass()
foo_instance.foo_method()"""

    metric = Overlap()
    score = metric.score(generated_code, reference_code, unique_apis)
    assert score == 1.0
