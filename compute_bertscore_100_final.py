# compute_bertscore_100_final.py

#   计算 100 条 learner-generated error units 的最终量化结果：
#   1) BERTScore F1：semantic homogenization
#   2) Jaccard Similarity：lexical homogenization
#

#   pip install pandas numpy bert-score torch openpyxl
#
# 运行：
#   python /.../compute_bertscore_100_final.py

import re
from pathlib import Path

import pandas as pd
import torch
from bert_score import score


# =========================
# 1. File paths
# =========================

INPUT_CSV = Path("/.../revisions_output.csv")

N_SAMPLES = 100

OUTPUT_DIR = INPUT_CSV.parent
OUTPUT_DETAILED_CSV = OUTPUT_DIR / "quantitative_results_bertscore_100_detailed.csv"
OUTPUT_EXCEL = OUTPUT_DIR / "quantitative_results_bertscore_100_analysis.xlsx"


# =========================
# 2. Model output columns
# =========================
# 如果你的 revisions_output.csv 列名不同，只需要改这里。

MODEL_COLUMNS = {
    "GPT-4o-mini": "gpt4omini_revision",
    "DeepSeek-V3": "deepseekv3_revision",
    "Gemini Flash": "geminiflash_revision",
}


# =========================
# 3. BERTScore settings


BERTSCORE_MODEL = "roberta-large"
RESCALE_WITH_BASELINE = True
BATCH_SIZE = 16


# =========================
# 4. Helper functions
# =========================

def tokenize_for_jaccard(text: str):
    """
    Tokenization for Jaccard Similarity.
    Lowercase the text and keep English words and numbers.
    """
    return re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?", str(text).lower())


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """
    Jaccard Similarity = shared lexical tokens / all unique lexical tokens.
    This is used as lexical homogenization.
    """
    a = set(tokenize_for_jaccard(text_a))
    b = set(tokenize_for_jaccard(text_b))
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def compute_bertscore_f1(candidates, references, model_type=BERTSCORE_MODEL):
    """
    Compute BERTScore F1 for aligned candidate-reference pairs.
    This is used as semantic homogenization.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\nRunning BERTScore")
    print(f"Device: {device}")
    print(f"Model: {model_type}")
    print(f"Number of pairs: {len(candidates)}")

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


def safe_group_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Create summary table by error_type or topic_type.
    """
    return (
        df.groupby(group_col)
        .agg(
            N=("unit_id", "count"),
            Mean_BERTScore=("semantic_homogenization_bertscore", "mean"),
            SD_BERTScore=("semantic_homogenization_bertscore", "std"),
            Min_BERTScore=("semantic_homogenization_bertscore", "min"),
            Max_BERTScore=("semantic_homogenization_bertscore", "max"),
            Mean_Jaccard=("lexical_homogenization_jaccard", "mean"),
            SD_Jaccard=("lexical_homogenization_jaccard", "std"),
            Min_Jaccard=("lexical_homogenization_jaccard", "min"),
            Max_Jaccard=("lexical_homogenization_jaccard", "max"),
        )
        .reset_index()
        .sort_values("Mean_BERTScore", ascending=False)
    )


# =========================
# 5. Main procedure
# =========================

def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Cannot find input file: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    required_cols = ["unit_id", "error_type", "topic_type"] + list(MODEL_COLUMNS.values())
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + str(missing)
            + "\nAvailable columns are: "
            + str(df.columns.tolist())
        )

    if len(df) < N_SAMPLES:
        raise ValueError(
            f"The input file contains only {len(df)} rows, but N_SAMPLES is set to {N_SAMPLES}."
        )

    # Use the first 100 rows as the final selected sample.
    df = df.head(N_SAMPLES).copy()

    print(f"Loaded {len(df)} rows from: {INPUT_CSV}")

    # Clean revision columns.
    for col in MODEL_COLUMNS.values():
        df[col] = df[col].fillna("").astype(str).str.strip()

    pair_specs = [
        ("gpt_deepseek", "GPT-4o-mini", "DeepSeek-V3"),
        ("gpt_gemini", "GPT-4o-mini", "Gemini Flash"),
        ("deepseek_gemini", "DeepSeek-V3", "Gemini Flash"),
    ]

    # -------------------------
    # 5.1 BERTScore F1
    # -------------------------
    for pair_key, model_a, model_b in pair_specs:
        col_a = MODEL_COLUMNS[model_a]
        col_b = MODEL_COLUMNS[model_b]

        print(f"\nComputing BERTScore F1: {model_a} vs {model_b}")
        df[f"bertscore_{pair_key}"] = compute_bertscore_f1(
            df[col_a].tolist(),
            df[col_b].tolist(),
        )

    df["semantic_homogenization_bertscore"] = df[
        [
            "bertscore_gpt_deepseek",
            "bertscore_gpt_gemini",
            "bertscore_deepseek_gemini",
        ]
    ].mean(axis=1)

    # -------------------------
    # 5.2 Jaccard Similarity
    # -------------------------
    for pair_key, model_a, model_b in pair_specs:
        col_a = MODEL_COLUMNS[model_a]
        col_b = MODEL_COLUMNS[model_b]

        df[f"jaccard_{pair_key}"] = [
            jaccard_similarity(text_a, text_b)
            for text_a, text_b in zip(df[col_a], df[col_b])
        ]

    df["lexical_homogenization_jaccard"] = df[
        [
            "jaccard_gpt_deepseek",
            "jaccard_gpt_gemini",
            "jaccard_deepseek_gemini",
        ]
    ].mean(axis=1)

    # -------------------------
    # 5.3 Exact match
    # -------------------------
    df["exact_same_all_three"] = (
        (df[MODEL_COLUMNS["GPT-4o-mini"]] == df[MODEL_COLUMNS["DeepSeek-V3"]])
        & (df[MODEL_COLUMNS["GPT-4o-mini"]] == df[MODEL_COLUMNS["Gemini Flash"]])
    )

    # -------------------------
    # 5.4 Summary tables
    # -------------------------
    overall_summary = pd.DataFrame(
        {
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
        }
    )

    pairwise_summary = pd.DataFrame(
        [
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
        ]
    )

    by_error_type = safe_group_summary(df, "error_type")
    by_topic_type = safe_group_summary(df, "topic_type")

    exact_match_summary = pd.DataFrame(
        {
            "Item": [
                "Exact same revision across all three LLMs",
                "Total error units",
            ],
            "Count": [
                int(df["exact_same_all_three"].sum()),
                len(df),
            ],
            "Percentage": [
                df["exact_same_all_three"].mean(),
                1.0,
            ],
        }
    )

    # -------------------------
    # 5.5 Save outputs
    # -------------------------
    df.to_csv(OUTPUT_DETAILED_CSV, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        overall_summary.to_excel(writer, sheet_name="Overall Summary", index=False)
        pairwise_summary.to_excel(writer, sheet_name="Pairwise Summary", index=False)
        by_error_type.to_excel(writer, sheet_name="By Error Type", index=False)
        by_topic_type.to_excel(writer, sheet_name="By Topic Type", index=False)
        exact_match_summary.to_excel(writer, sheet_name="Exact Match", index=False)
        df.to_excel(writer, sheet_name="Detailed Results", index=False)

    # -------------------------
    # 5.6 Print key results
    # -------------------------
    print("\nFinal 100-sample quantitative analysis completed successfully.")
    print(f"Saved detailed CSV: {OUTPUT_DETAILED_CSV}")
    print(f"Saved Excel workbook: {OUTPUT_EXCEL}")

    print("\nOverall summary:")
    print(overall_summary.to_string(index=False))

    print("\nPairwise summary:")
    print(pairwise_summary.to_string(index=False))

    print("\nExact match summary:")
    print(exact_match_summary.to_string(index=False))

    print("\nBy error type:")
    print(by_error_type.to_string(index=False))


if __name__ == "__main__":
    main()
