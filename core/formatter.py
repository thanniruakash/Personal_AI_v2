class Formatter:
    @staticmethod
    def format_result(result):
        if result is None:
            return "Sorry, I couldn't find an answer."

        if isinstance(result, dict):
            lines = []
            for key, value in result.items():
                lines.append(f"{key.capitalize()}:")
                if isinstance(value, list):
                    for item in value:
                        lines.append(f"  - {item}")
                else:
                    lines.append(f"  {value}")
                lines.append("")
            return "\n".join(lines).strip()

        return str(result)
