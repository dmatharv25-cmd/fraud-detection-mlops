from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"

LINES = [
    "## Drift monitoring",
    "",
    "Drift is measured with PSI (Population Stability Index) and the KS test, written with numpy and scipy. The reference period is the training split. The commonly used PSI cutoffs (0.1 and 0.25) are conventions, not laws.",
    "",
    "| Period | Fraud rate | Mean model score | Flagged at 0.11 |",
    "|---|---|---|---|",
    "| Train | 0.211% | 0.00211 | 0.211% |",
    "| Validation | 0.100% | 0.00078 | 0.084% |",
    "| Test | 0.132% | 0.00107 | 0.111% |",
    "",
    "- Against the training period, 8 of 29 features have PSI above 0.25 in validation and 7 of 29 in test.",
    "- V1, V3, V28, V11 and V25 lead both comparisons. V1 and V3 are above 1.0 in both. PSI and KS agree on these features.",
    "- The mean score and flag rate fall and rise together with the fraud rate. When labels arrive late, this is the early signal available in production.",
    "",
    "Does the drift hurt the model? The test period was split into two halves by time and scored with the same saved model:",
    "",
    "| | First half | Second half |",
    "|---|---|---|",
    "| Frauds | 53 | 22 |",
    "| PR-AUC | 0.848 (95% interval 0.748 to 0.934) | 0.730 (95% interval 0.545 to 0.888) |",
    "| Frauds caught at 0.11 | 40 of 53 (75%) | 16 of 22 (73%) |",
    "| False alarms | 1 | 6 |",
    "",
    "- Drift alerts fired, but the catch rate stayed about the same. An alert means look closer, not the model is broken.",
    "- The PR-AUC intervals overlap, and I did not test the difference directly, so I cannot call the drop a real decline. Part of it may come from the fall in fraud volume, but I did not separate that from model decay.",
    "- The data covers only about 48 hours. This demonstrates the monitoring method. It is not evidence of long-term drift, and some of the shift may be time-of-day effects.",
    "",
]


def main():
    text = README.read_text(encoding="utf-8")
    if "## Drift monitoring" in text:
        print("Section already present. Nothing changed.")
        return
    head, sep, tail = text.partition("## Next")
    if not sep:
        print("Could not find the '## Next' heading. Nothing changed.")
        return
    tail = tail.replace("Docker, drift monitoring.", "Docker, retraining trigger.")
    new_text = head + "\n".join(LINES) + "\n" + sep + tail
    README.write_text(new_text, encoding="utf-8")
    print("README updated.")


if __name__ == "__main__":
    main()