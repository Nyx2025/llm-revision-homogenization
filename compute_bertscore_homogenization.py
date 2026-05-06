# compute_bertscore_homogenization.py
# Purpose:
#   Compute real BERTScore + Jaccard Similarity for LLM revision homogenization.
# Input:
#   revisions_output.csv
# Required columns:
#   unit_id, error_type, topic_type,
#   gpt4omini_revision, deepseekv3_revision, geminiflash_revision
#
# Install first:
#   pip install pandas numpy bert-score torch openpyxl
#
# Recommended run:
#   python compute_bertscore_homogenization.py

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from bert_score import score


INPUT_CSV = "revisions_output.csv"
OUTPUT_DETAILED_CSV = "quantitative_results_bertscore_detailed.csv"
OUTPUT_EXCEL = "quantitative_results_bertscore_analysis.xlsx"

MODEL_COLUMNS = {
    "GPT-4o-mini": "gpt4omini_revision",
    "DeepSeek-V3": "deepseekv3_revision",
    "Gemini Flash": "geminiflash_revision",
}

# For English revision outputs, roberta-large is a common and defensible BERTScore setting.
# If your computer is slow, change this to "distilbert-base-uncased" for a faster pilot,
# but keep "roberta-large" for the final paper if possible.
BERTSCORE_MODEL = "roberta-large"
RESCALE_WITH_BASELINE = True
BATCH_SIZE = 16


def tokenize_for_jaccard(text: str):
    """Lowercase, remove punctuation, and keep words/numbers as lexical tokens."""
    return re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?", str(text).lower())


def jaccard_similarity(text_a: str, text_b: str) -> float:
    a = set(tokenize_for_jaccard(text_a))
    b = set(tokenize_for_jaccard(text_b))
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def compute_bertscore_f1(candidates, references, model_type=BERTSCORE_MODEL):
    """Return BERTScore F1 values for aligned candidate/reference lists."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running BERTScore on device: {device}")
    print(f"BERTScore model: {model_type}")
    P, R, F1 = score(
        candidates,
        references,
        lang="en",
        model_type=model_type,
        rescale_with_baseline=RESCALE_WITH_BASELINE,
        batch_size=BATCH_SIZE,
        device=device,
        verbose=True,
    )
    return F1.detach().cpu().numpy()


def main():
    input_path = Path(INPUT_CSV)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_CSV}. Put this script in the same folder as revisions_output.csv."
        )

    df = pd.read_csv(input_path)

    required_cols = ["unit_id", "error_type", "topic_type"] + list(MODEL_COLUMNS.values())
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Clean revision text columns.
    for col in MODEL_COLUMNS.values():
        df[col] = df[col].fillna("").astype(str).str.strip()

    pair_specs = [
        ("gpt_deepseek", "GPT-4o-mini", "DeepSeek-V3"),
        ("gpt_gemini", "GPT-4o-mini", "Gemini Flash"),
        ("deepseek_gemini", "DeepSeek-V3", "Gemini Flash"),
    ]

    # 1. BERTScore F1 for semantic homogenization.
    for pair_key, model_a, model_b in pair_specs:
        col_a = MODEL_COLUMNS[model_a]
        col_b = MODEL_COLUMNS[model_b]

        print(f"\nComputing BERTScore F1: {model_a} vs {model_b}")
        df[f"bertscore_{pair_key}"] = compute_bertscore_f1(
            df[col_a].tolist(),
            df[col_b].tolist(),
        )

    df["semantic_homogenization_bertscore"] = df[
        ["bertscore_gpt_deepseek", "bertscore_gpt_gemini", "bertscore_deepseek_gemini"]
    ].mean(axis=1)

    # 2. Jaccard Similarity for lexical homogenization.
    for pair_key, model_a, model_b in pair_specs:
        col_a = MODEL_COLUMNS[model_a]
        col_b = MODEL_COLUMNS[model_b]
        df[f"jaccard_{pair_key}"] = [
            jaccard_similarity(a, b) for a, b in zip(df[col_a], df[col_b])
        ]

    df["lexical_homogenization_jaccard"] = df[
        ["jaccard_gpt_deepseek", "jaccard_gpt_gemini", "jaccard_deepseek_gemini"]
    ].mean(axis=1)

    # Exact match indicator.
    df["exact_same_all_three"] = (
        (df[MODEL_COLUMNS["GPT-4o-mini"]] == df[MODEL_COLUMNS["DeepSeek-V3"]])
        & (df[MODEL_COLUMNS["GPT-4o-mini"]] == df[MODEL_COLUMNS["Gemini Flash"]])
    )

    # Summary tables.
    overall_summary = pd.DataFrame({
        "Metric": [
            "Semantic Homogenization: BERTScore F1",
            "Lexical Homogenization: Jaccard Similarity",
        ],
        "Mean": [
            df["semantic_homogenization_bertscore"].mean(),
            df["lexical_homogenization_jaccard"].mean(),
        ],
        "SD": [
            df["semantic_homogenization_bertscore"].std(),
            df["lexical_homogenization_jaccard"].std(),
        ],
        "Min": [
            df["semantic_homogenization_bertscore"].min(),
            df["lexical_homogenization_jaccard"].min(),
        ],
        "Max": [
            df["semantic_homogenization_bertscore"].max(),
            df["lexical_homogenization_jaccard"].max(),
        ],
    })

    pairwise_summary = pd.DataFrame([
        {
            "Model Pair": "GPT-4o-mini vs DeepSeek-V3",
            "Mean BERTScore F1": df["bertscore_gpt_deepseek"].mean(),
            "SD BERTScore F1": df["bertscore_gpt_deepseek"].std(),
            "Mean Jaccard": df["jaccard_gpt_deepseek"].mean(),
            "SD Jaccard": df["jaccard_gpt_deepseek"].std(),
        },
        {
            "Model Pair": "GPT-4o-mini vs Gemini Flash",
            "Mean BERTScore F1": df["bertscore_gpt_gemini"].mean(),
            "SD BERTScore F1": df["bertscore_gpt_gemini"].std(),
            "Mean Jaccard": df["jaccard_gpt_gemini"].mean(),
            "SD Jaccard": df["jaccard_gpt_gemini"].std(),
        },
        {
            "Model Pair": "DeepSeek-V3 vs Gemini Flash",
            "Mean BERTScore F1": df["bertscore_deepseek_gemini"].mean(),
            "SD BERTScore F1": df["bertscore_deepseek_gemini"].std(),
            "Mean Jaccard": df["jaccard_deepseek_gemini"].mean(),
            "SD Jaccard": df["jaccard_deepseek_gemini"].std(),
        },
    ])

    by_error_type = (
        df.groupby("error_type")
        .agg(
            N=("unit_id", "count"),
            Mean_BERTScore=("semantic_homogenization_bertscore", "mean"),
            SD_BERTScore=("semantic_homogenization_bertscore", "std"),
            Mean_Jaccard=("lexical_homogenization_jaccard", "mean"),
            SD_Jaccard=("lexical_homogenization_jaccard", "std"),
        )
        .reset_index()
        .sort_values("Mean_BERTScore", ascending=False)
    )

    by_topic_type = (
        df.groupby("topic_type")
        .agg(
            N=("unit_id", "count"),
            Mean_BERTScore=("semantic_homogenization_bertscore", "mean"),
            SD_BERTScore=("semantic_homogenization_bertscore", "std"),
            Mean_Jaccard=("lexical_homogenization_jaccard", "mean"),
            SD_Jaccard=("lexical_homogenization_jaccard", "std"),
        )
        .reset_index()
        .sort_values("Mean_BERTScore", ascending=False)
    )

    exact_match_summary = pd.DataFrame({
        "Item": ["Exact same revision across all three LLMs", "Total error units"],
        "Count": [int(df["exact_same_all_three"].sum()), len(df)],
        "Percentage": [df["exact_same_all_three"].mean(), 1.0],
    })

    # Save outputs.
    df.to_csv(OUTPUT_DETAILED_CSV, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        overall_summary.to_excel(writer, sheet_name="Overall Summary", index=False)
        pairwise_summary.to_excel(writer, sheet_name="Pairwise Summary", index=False)
        by_error_type.to_excel(writer, sheet_name="By Error Type", index=False)
        by_topic_type.to_excel(writer, sheet_name="By Topic Type", index=False)
        exact_match_summary.to_excel(writer, sheet_name="Exact Match", index=False)
        df.to_excel(writer, sheet_name="Detailed Results", index=False)

    print("\nDone.")
    print(f"Saved detailed CSV: {OUTPUT_DETAILED_CSV}")
    print(f"Saved Excel workbook: {OUTPUT_EXCEL}")
    print("\nOverall summary:")
    print(overall_summary.to_string(index=False))
    print("\nPairwise summary:")
    print(pairwise_summary.to_string(index=False))


if __name__ == "__main__":
    main()
