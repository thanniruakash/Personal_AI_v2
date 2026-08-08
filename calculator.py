import ast
import operator
import re


class Calculator:

    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def _calculate_node(self, node):

        if isinstance(node, ast.Constant):

            if isinstance(
                node.value,
                (int, float)
            ):
                return node.value

            raise ValueError("Invalid number")

        if isinstance(node, ast.UnaryOp):

            operation = self.OPERATORS.get(
                type(node.op)
            )

            if operation is None:
                raise ValueError(
                    "Invalid operator"
                )

            return operation(
                self._calculate_node(
                    node.operand
                )
            )

        if isinstance(node, ast.BinOp):

            operation = self.OPERATORS.get(
                type(node.op)
            )

            if operation is None:
                raise ValueError(
                    "Invalid operator"
                )

            left = self._calculate_node(
                node.left
            )

            right = self._calculate_node(
                node.right
            )

            return operation(
                left,
                right
            )

        raise ValueError(
            "Invalid calculation"
        )

    def calculate(self, expression):

        expression = expression.strip()

        if not expression:
            return None

        # Allow only numbers, decimal points,
        # spaces, brackets and math operators.
        if not re.fullmatch(
            r"[0-9+\-*/%.() \t]+",
            expression
        ):
            return None

        try:

            tree = ast.parse(
                expression,
                mode="eval"
            )

            result = self._calculate_node(
                tree.body
            )

            if isinstance(
                result,
                float
            ) and result.is_integer():

                result = int(result)

            return result

        except Exception:

            return None

    def extract_expression(self, question):

        q = question.lower().strip()

        # Remove common calculation words.
        q = re.sub(
            r"\b(what is|calculate|compute|solve|"
            r"please calculate|answer)\b",
            "",
            q
        )

        # Convert "50% of 800" -> "(50/100)*800"
        match = re.fullmatch(
            r"\s*(\d+(?:\.\d+)?)\s*%\s*of\s*"
            r"(\d+(?:\.\d+)?)\s*",
            q
        )

        if match:

            percentage = match.group(1)
            number = match.group(2)

            return (
                f"({percentage}/100)*{number}"
            )

        # Keep only if it is a mathematical
        # expression.
        expression = q.strip()

        if re.fullmatch(
            r"[0-9+\-*/%.() \t]+",
            expression
        ):

            return expression

        return None

    def ask(self, question):

        expression = (
            self.extract_expression(
                question
            )
        )

        if expression is None:

            return None

        result = self.calculate(
            expression
        )

        if result is None:

            return None

        return (
            f"Answer: {result}"
        )
