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


class TestUniversalNumeric(unittest.TestCase):
    """MathParser.extract_universal_numeric — robust extraction pipeline."""

    def setUp(self):
        self.mp = MathParser()

    def test_boxed_extraction(self):
        self.assertEqual(self.mp.extract_universal_numeric(r"Result is \boxed{42}."), "42")

    def test_boxed_fraction(self):
        # This matches what the user was trying to test
        self.assertEqual(self.mp.extract_universal_numeric(r"\boxed{\frac{1}{2}}"), "1/2")

    def test_scientific_notation(self):
        self.assertEqual(self.mp.extract_universal_numeric("Value: 1.23e-4"), "0.000123")

    def test_comma_numbers(self):
        self.assertEqual(self.mp.extract_universal_numeric("The total is 1,234.5"), "1234.5")

    def test_percentage(self):
        self.assertEqual(self.mp.extract_universal_numeric("The rate is 50%"), "1/2")

    def test_mixed_text(self):
        text = "The answer is 42, but wait, the final answer: 7."
        self.assertEqual(self.mp.extract_universal_numeric(text), "7")

    def test_mixed_number(self):
        self.assertEqual(self.mp.extract_universal_numeric("The answer is 3 1/2."), "7/2")

    def test_plain_fraction(self):
        self.assertEqual(self.mp.extract_universal_numeric("It is 3/4 empty"), "3/4")

    def test_degree_symbol(self):
        self.assertEqual(self.mp.extract_universal_numeric(r"Angle is 45^\circ"), "45")

    def test_complex_latex_formatting(self):
        self.assertEqual(self.mp.extract_universal_numeric(r"The answer is \mathrm{42}."), "42")
        self.assertEqual(self.mp.extract_universal_numeric(r"\mathbf{10.5}"), "10.5")

    def test_sqrt_latex(self):
        self.assertEqual(self.mp.extract_universal_numeric(r"\sqrt{16}"), "4")

    def test_scientific_times(self):
        self.assertEqual(self.mp.extract_universal_numeric(r"1.23 \times 10^{-4}"), "0.000123")
        self.assertEqual(self.mp.extract_universal_numeric(r"2 \cdot 10^3"), "2000")

    def test_simple_arithmetic(self):
        self.assertEqual(self.mp.extract_universal_numeric("The answer is 1+2."), "3")
        self.assertEqual(self.mp.extract_universal_numeric("2 * 3"), "6")


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
