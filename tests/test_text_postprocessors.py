"""Tests for MathParser (d5p4.text_postprocessors)."""

import unittest

from sympy import Rational

from d5p4.text_postprocessors import MathParser


class TestNormalize(unittest.TestCase):
    """MathParser.normalize — LaTeX cleanup."""

    def test_strips_dollar_signs(self):
        self.assertEqual(MathParser.normalize("$42$"), "42")

    def test_strips_bracket_delimiters(self):
        self.assertEqual(MathParser.normalize("\\[x + 1\\]"), "x+1")

    def test_strips_paren_delimiters(self):
        self.assertEqual(MathParser.normalize("\\(3.14\\)"), "3.14")

    def test_removes_text_command(self):
        self.assertEqual(MathParser.normalize("5\\text{ meters}"), "5")

    def test_removes_trailing_dot(self):
        self.assertEqual(MathParser.normalize("42."), "42")

    def test_unescapes_percent(self):
        self.assertEqual(MathParser.normalize("50\\%"), "50%")

    def test_unescapes_dollar(self):
        self.assertEqual(MathParser.normalize("\\$100"), "$100")

    def test_strips_spaces(self):
        self.assertEqual(MathParser.normalize(" 1 + 2 "), "1+2")


class TestExtractBoxedOrLastNumber(unittest.TestCase):
    """MathParser.extract_boxed_or_last_number — main extraction pipeline."""

    def setUp(self):
        self.mp = MathParser()

    def test_empty_input(self):
        self.assertEqual(self.mp.extract_boxed_or_last_number(""), "")

    def test_boxed_simple(self):
        self.assertEqual(
            self.mp.extract_boxed_or_last_number(r"The answer is \boxed{42}."),
            "42",
        )

    def test_boxed_fraction(self):
        result = self.mp.extract_boxed_or_last_number(r"\boxed{\frac{1}{2}}")
        self.assertEqual(result, "\\frac{1}{2}")

    def test_fallback_last_number(self):
        self.assertEqual(
            self.mp.extract_boxed_or_last_number("There are 7 apples and 12 oranges"),
            "12",
        )

    def test_fallback_decimal(self):
        self.assertEqual(
            self.mp.extract_boxed_or_last_number("The radius is $3.14$ meters."),
            "3.14",
        )

    def test_fallback_fraction_slash(self):
        self.assertEqual(
            self.mp.extract_boxed_or_last_number("The ratio is 1/2 of the total"),
            "1/2",
        )

    def test_no_numbers_returns_input(self):
        self.assertEqual(
            self.mp.extract_boxed_or_last_number("no numbers here"),
            "no numbers here",
        )

    def test_negative_number(self):
        self.assertEqual(
            self.mp.extract_boxed_or_last_number("Temperature is -5 degrees"),
            "-5",
        )


class TestExtractFinalNumber(unittest.TestCase):
    """MathParser.extract_final_number — comma-aware last number."""

    def setUp(self):
        self.mp = MathParser()

    def test_empty_input(self):
        self.assertEqual(self.mp.extract_final_number(""), "")

    def test_comma_separated(self):
        self.assertEqual(
            self.mp.extract_final_number("The cost is 1,234.50 dollars."),
            "1234.50",
        )

    def test_multiple_numbers(self):
        self.assertEqual(
            self.mp.extract_final_number("5 items cost 100"),
            "100",
        )

    def test_negative(self):
        self.assertEqual(self.mp.extract_final_number("Profit: -42"), "-42")

    def test_no_number(self):
        self.assertEqual(self.mp.extract_final_number("no numbers"), "")


class TestExtractMathExpression(unittest.TestCase):
    """MathParser.extract_math_expression — lightweight operator extraction."""

    def setUp(self):
        self.mp = MathParser()

    def test_empty_input(self):
        self.assertEqual(self.mp.extract_math_expression(""), "")

    def test_simple_expression(self):
        self.assertEqual(self.mp.extract_math_expression("result is 3+4"), "3+4")

    def test_decimal(self):
        self.assertEqual(self.mp.extract_math_expression("value: 3.14"), "3.14")

    def test_no_expression(self):
        self.assertEqual(self.mp.extract_math_expression("hello world"), "")


class TestParseLatex(unittest.TestCase):
    """MathParser.parse_latex — symbolic / evaluated SymPy parsing."""

    def setUp(self):
        self.mp = MathParser()

    def test_empty_input(self):
        self.assertIsNone(self.mp.parse_latex(""))

    def test_symbolic_fraction(self):
        expr = self.mp.parse_latex(r"\frac{1}{2} + \frac{1}{4}")
        self.assertIsNotNone(expr)
        self.assertEqual(expr, Rational(3, 4))

    def test_evaluate_mode(self):
        result = self.mp.parse_latex(r"\sqrt{16} * 2", evaluate=True)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(float(result), 8.0)

    def test_invalid_latex_returns_none(self):
        self.assertIsNone(self.mp.parse_latex(r"\pi \approx"))


if __name__ == "__main__":
    unittest.main()
