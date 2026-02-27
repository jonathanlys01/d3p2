"""Extract number from text string"""

import re

from sympy import simplify
from sympy.parsing.latex import parse_latex


def gsm8k_postprocess(text: str) -> str:
    """opencompass"""
    text = text.split("Question:")[0]
    numbers = re.findall(r"\-?\d+\.\d+|\-?\d+", text)
    if not numbers:
        return "NULL"
    return numbers[-1]


def extract_and_parse_math(generation: str) -> str:
    """
    Extracts and normalizes a math answer from a model's generation.
    """
    if not generation:
        return ""

    # 1. Try to extract the answer from a LaTeX \boxed{} tag
    # This matches \boxed{...} and handles basic nested brackets
    boxed_match = re.search(r"\\boxed{((?:[^{}]|{[^{}]*})*)}", generation)

    if boxed_match:
        answer = boxed_match.group(1)
    else:
        # 2. Fallback: Extract the last standalone number or fraction
        # Matches formats like: 42, -42, 3.14, 1/2, -1/2
        numbers = re.findall(r"-?\d+(?:\.\d+)?(?:/-?\d+)?", generation)
        answer = numbers[-1] if numbers else generation

    return normalize_math(answer)


def normalize_math(answer: str) -> str:
    """
    Cleans up the extracted math string to make it easy to compare.
    """
    # Shelter escaped chars with sentinels so the bare-$ strip doesn't eat them
    answer = answer.replace("\\%", "\x00PCT").replace("\\$", "\x00DLR")

    # Remove standard LaTeX math delimiters
    answer = answer.replace("$", "").replace("\\[", "").replace("\\]", "")
    answer = answer.replace("\\(", "").replace("\\)", "")

    # Remove basic LaTeX text commands
    answer = re.sub(r"\\text{.*?}", "", answer)

    # Restore escaped chars
    answer = answer.replace("\x00PCT", "%").replace("\x00DLR", "$")

    # Remove spaces and trailing punctuation
    answer = answer.replace(" ", "")
    answer = answer.rstrip(".")

    return answer.strip()


def extract_math_expression(text: str) -> str:
    """
    Extracts the last math expression (numbers, fractions, basic operators).
    """
    if not text:
        return ""

    # Matches digits and basic math symbols: +, -, *, /, %, ., (, )
    pattern = r"[-+*/%\.\(\)\d]+"
    matches = re.findall(pattern, text)

    if matches:
        # Grab the last match and strip any accidental leading closing parenthesis
        result = matches[-1].lstrip(")")
        return result.strip()

    return ""


def extract_final_number(text: str) -> str:
    """
    Extracts the final number from a string, stripping out commas.
    """
    if not text:
        return ""

    # Matches optional minus, digits with optional commas, and optional decimals
    matches = re.findall(r"-?[\d,]+(?:.\d+)?", text)

    if matches:
        # Return the last matched number and remove commas for easy parsing
        return matches[-1].replace(",", "")

    return ""


def parse_latex_to_math(latex_str: str):
    """
    Parses a LaTeX math string into a SymPy expression.
    """
    if not latex_str:
        return None

    try:
        # Convert the LaTeX string into a symbolic math object
        math_expr = parse_latex(latex_str, backend="lark")
        return math_expr
    except Exception as e:
        print(f"Failed to parse LaTeX: {e}")
        return None


def evaluate_llm_math(latex_str: str):
    """
    Uses sympy.parsing.latex (lark backend) to parse and evaluate mathematical LaTeX.
    """
    try:
        # parse_latex with lark backend returns a SymPy object
        expr = parse_latex(latex_str, backend="lark")

        if expr is None:
            return None

        # You can return the evaluated float, or keep it symbolic
        if hasattr(expr, "evalf"):
            return expr.evalf()
        return expr
    except Exception:
        # Fallback if the string isn't valid math
        return None


# ---------------------------------------------------------------------------
# Class-based API — consolidates the most robust parsers above
# ---------------------------------------------------------------------------


class MathParser:
    """Robust math / LaTeX answer extractor and evaluator.

    Pre-compiles all regex patterns once and exposes a clean API that
    covers the main extraction strategies:
      • ``extract_boxed_or_last_number`` — best general-purpose pipeline
      • ``extract_final_number`` — comma-aware last-number extraction
      • ``extract_math_expression`` — lightweight operator-level extraction
      • ``parse_latex`` — symbolic parsing via SymPy (optionally evaluated)
    """

    # Pre-compiled regexes ───────────────────────────────────────────────────
    _BOXED_RE = re.compile(r"\\boxed{((?:[^{}]|{[^{}]*})*)}")
    _NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:/-?\d+)?")
    _FINAL_NUMBER_RE = re.compile(r"-?[\d,]+(?:\.\d+)?")
    _MATH_EXPR_RE = re.compile(r"[-+*/%\.\(\)\d]+")
    _LATEX_TEXT_RE = re.compile(r"\\text{.*?}")

    # ── normalisation ──────────────────────────────────────────────────────

    @staticmethod
    def normalize(answer: str) -> str:
        """Strip LaTeX delimiters, ``\\text{}``, and trailing punctuation."""
        # Shelter escaped chars with sentinels so the bare-$ strip doesn't eat them
        answer = answer.replace("\\%", "\x00PCT").replace("\\$", "\x00DLR")
        answer = answer.replace("$", "").replace("\\[", "").replace("\\]", "")
        answer = answer.replace("\\(", "").replace("\\)", "")
        answer = MathParser._LATEX_TEXT_RE.sub("", answer)
        # Restore escaped chars
        answer = answer.replace("\x00PCT", "%").replace("\x00DLR", "$")
        answer = answer.replace(" ", "").rstrip(".")
        return answer.strip()

    # ── text → str extraction ──────────────────────────────────────────────

    def extract_boxed_or_last_number(self, text: str) -> str:
        r"""Extract from ``\boxed{}`` if present, else the last number/fraction.

        The result is fed through :meth:`normalize` before returning.
        """
        if not text:
            return ""

        boxed = self._BOXED_RE.search(text)
        if boxed:
            return self.normalize(boxed.group(1))

        numbers = self._NUMBER_RE.findall(text)
        return self.normalize(numbers[-1]) if numbers else text

    def extract_final_number(self, text: str) -> str:
        """Return the last number in *text*, stripping commas."""
        if not text:
            return ""

        matches = self._FINAL_NUMBER_RE.findall(text)
        return matches[-1].replace(",", "") if matches else ""

    def extract_math_expression(self, text: str) -> str:
        """Return the last math expression (digits + basic operators)."""
        if not text:
            return ""

        matches = self._MATH_EXPR_RE.findall(text)
        if matches:
            return matches[-1].lstrip(")").strip()
        return ""

    # ── symbolic parsing ───────────────────────────────────────────────────

    def parse_latex(self, latex_str: str, *, evaluate: bool = False):
        """Parse a LaTeX string into a SymPy expression.

        Parameters
        ----------
        latex_str:
            Raw LaTeX math string (e.g. ``\\frac{1}{2} + \\frac{1}{4}``).
        evaluate:
            If *True*, call ``.evalf()`` on the result to obtain a float.

        Returns
        -------
        sympy.Expr | None
            The parsed (and optionally evaluated) expression, or *None*
            if parsing fails.
        """
        if not latex_str:
            return None

        try:
            expr = parse_latex(latex_str, backend="lark")
            if expr is None:
                return None
            if evaluate and hasattr(expr, "evalf"):
                return expr.evalf()
            return expr
        except Exception:
            return None


if __name__ == "__main__":
    mp = MathParser()

    print(mp.extract_boxed_or_last_number(r"The answer is \boxed{42}."))
    print(mp.extract_boxed_or_last_number(r"Therefore, the radius is $3.14$ meters."))
    print(mp.extract_boxed_or_last_number(r"I calculate the fraction to be \boxed{\frac{1}{2}}"))
    print(mp.extract_final_number("The cost of the 5 items is 1,234.50 dollars."))

    expr = mp.parse_latex(r"\frac{1}{2} + \frac{1}{4}")
    if expr is not None:
        print(f"Symbolic: {expr}, Simplified: {simplify(expr)}, Float: {expr.evalf()}")

    print(mp.parse_latex(r"\sqrt{16} * 2", evaluate=True))
    print(mp.parse_latex(r"\pi \approx"))  # None
