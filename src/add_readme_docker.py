from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"

LINES = [
    "## Docker",
    "",
    "The prediction API can run in a container. The model file is not in Git, so create it first, then build and run:",
    "",
    "```",
    "python -m src.save_model",
    "docker build -t fraud-api .",
    "docker run --rm -p 8000:8000 fraud-api",
    "```",
    "",
    "The same benchmark (200 test-set transactions, one request at a time) was run against the local server and against the container:",
    "",
    "| | Local server | Docker container |",
    "|---|---|---|",
    "| Median latency | 6.8 ms | 10.2 ms |",
    "| p95 latency | 8.6 ms | 11.7 ms |",
    "| p99 latency | 9.2 ms | 13.0 ms |",
    "| Max score difference vs direct scoring | 0 | 2.6e-26 |",
    "",
    "- Scores match to floating-point rounding. I did not verify the cause of the tiny difference.",
    "- The container is about 3.5 ms slower at the median. I did not measure where the extra time goes.",
    "- The model is copied in from the local `models/` folder at build time, so the image only works with that exact model. A real system would load it from a registry or storage at startup.",
    "- The image also contains `candidate.pkl`, which the API never loads.",
    "- Latency is one request at a time on one laptop. This is not a load test.",
    "",
]


def main():
    text = README.read_text(encoding="utf-8")
    if "## Docker" in text:
        print("Section already present. Nothing changed.")
        return
    head, sep, tail = text.partition("## Next")
    if not sep:
        print("Could not find the '## Next' heading. Nothing changed.")
        return
    tail = tail.replace("Docker, tests and CI.", "Load testing, a fresh-window evaluation of the retrained candidate.")
    new_text = head + "\n".join(LINES) + "\n" + sep + tail
    README.write_text(new_text, encoding="utf-8")
    print("README updated.")


if __name__ == "__main__":
    main()