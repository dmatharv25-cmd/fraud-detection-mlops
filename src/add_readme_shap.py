from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"

LINES = [
    "## Explainability (SHAP)",
    "",
    "SHAP values were computed for the final LightGBM on a test-set sample: all 75 frauds plus 2,000 random normal transactions. Values are in log-odds units, where positive pushes the score toward fraud.",
    "",
    "![SHAP summary](shap_summary.png)",
    "",
    "| Rank | Feature | Mean abs SHAP |",
    "|---|---|---|",
    "| 1 | V14 | 0.860 |",
    "| 2 | V4 | 0.677 |",
    "| 3 | V8 | 0.578 |",
    "| 4 | V12 | 0.531 |",
    "| 5 | V11 | 0.398 |",
    "",
    "- The highest-scored fraud (score 1.000) had several extreme feature values agreeing, led by V14 (+7.3), V12, V17, V4 and V10.",
    "- The lowest-scored fraud (score 0.000, missed) had no strong signal on the features the model relies on. Its largest contribution was only +1.2 from V14, and some features pushed the other way.",
    "- V1 to V28 are anonymized PCA components, so SHAP shows which components drive the model but not what they mean in real life.",
    "- The sample is fraud-enriched, so the ranking reflects what drives fraud calls, not importance across normal traffic.",
    "- I looked at only two individual transactions. I have not checked whether the missed frauds share a pattern.",
    "",
]


def main():
    text = README.read_text(encoding="utf-8")
    if "## Explainability (SHAP)" in text:
        print("Section already present. Nothing changed.")
        return
    head, sep, tail = text.partition("## Next")
    if not sep:
        print("Could not find the '## Next' heading. Nothing changed.")
        return
    tail = tail.replace("SHAP explanations, ", "")
    new_text = head + "\n".join(LINES) + "\n" + sep + tail
    README.write_text(new_text, encoding="utf-8")
    print("README updated.")


if __name__ == "__main__":
    main()