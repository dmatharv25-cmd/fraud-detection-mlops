from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"

LINES = [
    "## Cost-sensitive threshold results",
    "",
    "Costs are assumptions, not real bank data. A false alarm costs 5, and a missed fraud costs the value in the first column. Thresholds were chosen on validation only, then applied once to the test set.",
    "",
    "| Missed-fraud cost | LogReg test cost | LightGBM test cost | Flag nothing |",
    "|---|---|---|---|",
    "| 10 | 435 | 230 | 750 |",
    "| 50 | 1075 | 985 | 3750 |",
    "| 100 | 1745 | 1935 | 7500 |",
    "| 500 | 6680 | 9535 | 37500 |",
    "",
    "- Both models beat flagging nothing at every cost ratio.",
    "- The lower-cost model flipped between a missed-fraud cost of 50 and 100 (LightGBM lower at 10 and 50, Logistic Regression lower at 100 and 500).",
    "- The flip is not established. The gaps come from a handful of transactions, thresholds were picked on only 57 validation frauds, and I did not compute bootstrap intervals for these costs.",
    "- Logistic Regression's best threshold hit the edge of my search grid (0.99) at low costs, so the true best cutoff may be higher.",
    "",
]


def main():
    text = README.read_text(encoding="utf-8")
    if "## Cost-sensitive threshold results" in text:
        print("Section already present. Nothing changed.")
        return
    head, sep, tail = text.partition("## Next")
    if not sep:
        print("Could not find the '## Next' heading. Nothing changed.")
        return
    tail = tail.replace("Cost-sensitive threshold, SHAP", "SHAP")
    new_text = head + "\n".join(LINES) + "\n" + sep + tail
    README.write_text(new_text, encoding="utf-8")
    print("README updated.")


if __name__ == "__main__":
    main()