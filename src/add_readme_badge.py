from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"

BADGE = "[![tests](https://github.com/dmatharv25-cmd/fraud-detection-mlops/actions/workflows/tests.yml/badge.svg)](https://github.com/dmatharv25-cmd/fraud-detection-mlops/actions/workflows/tests.yml)"
NOTE = "The badge means the 4 unit tests pass (time-split order, PSI, cost arithmetic). They check code logic, not model quality."


def main():
    text = README.read_text(encoding="utf-8")
    if "badge.svg" in text:
        print("Badge already present. Nothing changed.")
        return
    first, sep, rest = text.partition("\n")
    if not sep:
        print("Unexpected README layout. Nothing changed.")
        return
    new_text = first + "\n\n" + BADGE + "\n\n" + NOTE + "\n" + rest
    README.write_text(new_text, encoding="utf-8")
    print("README updated.")


if __name__ == "__main__":
    main()