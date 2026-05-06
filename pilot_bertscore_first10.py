# pilot_bertscore_first10.py
# 目的：先计算 revisions_output.csv 前 10 条的 BERTScore + Jaccard Similarity
# 运行前安装：
#   pip install pandas numpy bert-score torch openpyxl
#
# 运行：
#   python pilot_bertscore_first10.py

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from bert_score import score


# 你的 revision_output 文件绝对路径
INPUT_CSV = Path("/.../revisions_output.csv")

# 只跑前 10 条做 pilot
N_PILOT = 10

# 输出文件会自动保存在 Downloads 文件夹
OUTPUT_DIR = INPUT_CSV.parent
OUTPUT_PILOT_CSV = OUTPUT_DIR / "pilot_bertscore_first10_results.csv"
OUTPUT_PILOT_EXCEL = OUTPUT_DIR / "pilot_bertscore_first10_results.xlsx"

# 三个模型 revision 所在列名
MODEL_COLUMNS = {
    "GPT-4o-mini": "gpt4omini_revision",
    "DeepSeek-V3": "deepseekv3_revision",
    "Gemini Flash": "geminiflash_revision",
}

# 为了和正式论文方法一致，pilot 也使用 roberta-large
BERTSCORE_MODEL = "roberta-large"
RESCALE_WITH_BASELINE = True
BATCH_SIZE = 8


def tokenize_for_jaccard(text: str):
    """用于 Jaccard Similarity 的简单分词：小写化，保留英文单词和数字。"""
    return re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?", str(text).lower())


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Jaccard = 两个文本共享词汇数量 / 两个文本全部不同词汇数量。"""
    a = set(tokenize_for_jaccard(text_a))
    b = set(tokenize_for_jaccard(text_b))
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def compute_bertscore_f1(candidates, references, model_type=BERTSCORE_MODEL):
    """计算 aligned candidate/reference list 的 BERTScore F1。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nRunning BERTScore on device: {device}")
    print(f"BERTScore model: {model_type}")
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


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Cannot find file: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    required_cols = ["unit_id", "error_type", "topic_type"] + list(MODEL_COLUMNS.values())
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: "
            + str(missing)
            + "\nAvailable columns are: "
            + str(df.columns.tolist())
        )

    # 只取前 10 条做 pilot
    df = df.head(N_PILOT).copy()

    # 清理 revision 文本
    for col in MODEL_COLUMNS.values():
        df[col] = df[col].fillna("").astype(str).str.strip()

    pair_specs = [
        ("gpt_deepseek", "GPT-4o-mini", "DeepSeek-V3"),
        ("gpt_gemini", "GPT-4o-mini", "Gemini Flash"),
        ("deepseek_gemini", "DeepSeek-V3", "Gemini Flash"),
    ]

    # 1. BERTScore F1：语义层面的 homogenization
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

    # 2. Jaccard Similarity：词汇层面的 homogenization
    for pair_key, model_a, model_b in pair_specs:
        col_a = MODEL_COLUMNS[model_a]
        col_b = MODEL_COLUMNS[model_b]
        df[f"jaccard_{pair_key}"] = [
            jaccard_similarity(a, b) for a, b in zip(df[col_a], df[col_b])
        ]

    df["lexical_homogenization_jaccard"] = df[
        ["jaccard_gpt_deepseek", "jaccard_gpt_gemini", "jaccard_deepseek_gemini"]
    ].mean(axis=1)

    # 3. 完全一致检查
    df["exact_same_all_three"] = (
        (df[MODEL_COLUMNS["GPT-4o-mini"]] == df[MODEL_COLUMNS["DeepSeek-V3"]])
        & (df[MODEL_COLUMNS["GPT-4o-mini"]] == df[MODEL_COLUMNS["Gemini Flash"]])
    )

    # 4. 汇总结果
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
            "Mean Jaccard": df["jaccard_gpt_deepseek"].mean(),
        },
        {
            "Model Pair": "GPT-4o-mini vs Gemini Flash",
            "Mean BERTScore F1": df["bertscore_gpt_gemini"].mean(),
            "Mean Jaccard": df["jaccard_gpt_gemini"].mean(),
        },
        {
            "Model Pair": "DeepSeek-V3 vs Gemini Flash",
            "Mean BERTScore F1": df["bertscore_deepseek_gemini"].mean(),
            "Mean Jaccard": df["jaccard_deepseek_gemini"].mean(),
        },
    ])

    # 5. 保存结果
    df.to_csv(OUTPUT_PILOT_CSV, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(OUTPUT_PILOT_EXCEL, engine="openpyxl") as writer:
        overall_summary.to_excel(writer, sheet_name="Overall Summary", index=False)
        pairwise_summary.to_excel(writer, sheet_name="Pairwise Summary", index=False)
        df.to_excel(writer, sheet_name="Pilot Detailed Results", index=False)

    print("\nPilot completed successfully.")
    print(f"Saved pilot CSV: {OUTPUT_PILOT_CSV}")
    print(f"Saved pilot Excel: {OUTPUT_PILOT_EXCEL}")

    print("\nOverall summary:")
    print(overall_summary.to_string(index=False))

    print("\nPairwise summary:")
    print(pairwise_summary.to_string(index=False))

    print("\nFirst 10 unit-level scores:")
    print(
        df[
            [
                "unit_id",
                "error_type",
                "semantic_homogenization_bertscore",
                "lexical_homogenization_jaccard",
                "exact_same_all_three",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
