from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"

LINES = [
    "## Retraining trigger",
    "",
    "`python -m src.retrain_trigger` decides whether to retrain. It retrains only when both conditions fire:",
    "",
    "- Drift: more than 5 of 29 features have PSI above 0.25 against the training period.",
    "- Performance: the share of frauds caught at the 0.11 cutoff falls by more than 10 percentage points compared with validation.",
    "",
    "Result on this data, with the test period as the new window:",
    "",
    "| Check | Value | Fired |",
    "|---|---|---|",
    "| Drifted features | 7 of 29 | Yes |",
    "| Catch rate, validation vs new window | 75.4% vs 74.7% | No |",
    "",
    "- Decision: do not retrain. Drift alone is not enough, because earlier the drift alerts fired while the catch rate barely moved.",
    "- The limits (PSI 0.25, more than 5 features, a 10 point drop) are my choices, not standards.",
    "- The catch-rate comparison rests on 57 validation frauds and 75 test frauds, so a 10 point margin is a loose guard.",
    "- `--force` trains a candidate on all data and saves it to `models/candidate.pkl`. It never overwrites the current model. The candidate cannot be evaluated here because no unseen data is left. Promoting it would need a fresh later time window.",
    "",
]


def main():
    text = README.read_text(encoding="utf-8")
    if "## Retraining trigger" in text:
        print("Section already present. Nothing changed.")
        return
    head, sep, tail = text.partition("## Next")
    if not sep:
        print("Could not find the '## Next' heading. Nothing changed.")
        return
    tail = tail.replace("Docker, retraining trigger.", "Docker, tests and CI.")
    new_text = head + "\n".join(LINES) + "\n" + sep + tail
    README.write_text(new_text, encoding="utf-8")
    print("README updated.")


if __name__ == "__main__":
    main()