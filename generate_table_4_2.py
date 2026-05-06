#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_table_4_2.py

Purpose
-------
Generate Table 4.2: Overall Homogenization Scores Across 100 Error Units.

Table 4.2 reports the overall semantic and lexical homogenization scores
across the 100 valid learner-generated error units.

Calculation logic
-----------------
For each error unit, BERTScore F1 and Jaccard Similarity were first averaged
across the three pairwise model comparisons:

    1. GPT-4o-mini vs DeepSeek-V3
    2. GPT-4o-mini vs Gemini Flash
    3. DeepSeek-V3 vs Gemini Flash

The table then reports the mean, standard deviation, minimum, and maximum
values across the 100 error units.

In the paper, the table note can be written as:

    Note. For each error unit, BERTScore F1 and Jaccard Similarity were first
    averaged across the three pairwise model comparisons. The table then reports
    the mean, standard deviation, minimum, and maximum values across the 100
    error units. Values are rounded to three decimal places.

Expected input
--------------
A detailed quantitative results CSV file, normally produced by the full
BERTScore + Jaccard analysis script.

The input file should contain at least the following pairwise score columns:

    bertscore_gpt_deepseek
    bertscore_gpt_gemini
    bertscore_deepseek_gemini
    jaccard_gpt_deepseek
    jaccard_gpt_gemini
    jaccard_deepseek_gemini

If the file already contains these two columns, the script will reuse them:

    semantic_homogenization_bertscore
    lexical_homogenization_jaccard

Default input path
------------------
    /Users/renwei/Downloads/quantitative_results_bertscore_100_detailed.csv

Default outputs
---------------
    /Users/renwei/Downloads/table_4_2_overall_homogenization_scores.csv
    /Users/renwei/Downloads/table_4_2_overall_homogenization_scores.xlsx

Usage
-----
Option 1: Use default paths

    python generate_table_4_2.py

Option 2: Specify your own input file

    python generate_table_4_2.py --input /path/to/quantitative_results_bertscore_100_detailed.csv

Option 3: Specify input and output directory

    python generate_table_4_2.py \
        --input /path/to/quantitative_results_bertscore_100_detailed.csv \
        --output-dir /path/to/output_folder
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_INPUT = Path("/Users/renwei/Downloads/quantitative_results_bertscore_100_detailed.csv")

BERTSCORE_PAIR_COLUMNS = [
    "bertscore_gpt_deepseek",
    "bertscore_gpt_gemini",
    "bertscore_deepseek_gemini",
]

JACCARD_PAIR_COLUMNS = [
    "jaccard_gpt_deepseek",
    "jaccard_gpt_gemini",
    "jaccard_deepseek_gemini",
]

SEMANTIC_SCORE_COLUMN = "semantic_homogenization_bertscore"
LEXICAL_SCORE_COLUMN = "lexical_homogenization_jaccard"


def check_required_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    """Raise a clear error message if required columns are missing."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            "The input file is missing required columns:\n"
            f"{missing}\n\n"
            "Available columns are:\n"
            f"{list(df.columns)}"
        )


def add_homogenization_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add semantic and lexical homogenization columns if they are not already present.

    Semantic homogenization:
        average BERTScore F1 across the three pairwise model comparisons.

    Lexical homogenization:
        average Jaccard Similarity across the same three pairwise model comparisons.
    """
    df = df.copy()

    if SEMANTIC_SCORE_COLUMN not in df.columns:
        check_required_columns(df, BERTSCORE_PAIR_COLUMNS)
        df[SEMANTIC_SCORE_COLUMN] = df[BERTSCORE_PAIR_COLUMNS].mean(axis=1)

    if LEXICAL_SCORE_COLUMN not in df.columns:
        check_required_columns(df, JACCARD_PAIR_COLUMNS)
        df[LEXICAL_SCORE_COLUMN] = df[JACCARD_PAIR_COLUMNS].mean(axis=1)

    return df


def generate_table_4_2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate Table 4.2.

    The table reports descriptive statistics across all error units:
        Mean, SD, Min, Max

    By default, pandas uses sample standard deviation (ddof=1), which is
    appropriate for reporting descriptive statistics for the sample.
    """
    df = add_homogenization_scores(df)

    table = pd.DataFrame(
        [
            {
                "Metric": "BERTScore F1 (semantic homogenization)",
                "Mean": df[SEMANTIC_SCORE_COLUMN].mean(),
                "SD": df[SEMANTIC_SCORE_COLUMN].std(),
                "Min": df[SEMANTIC_SCORE_COLUMN].min(),
                "Max": df[SEMANTIC_SCORE_COLUMN].max(),
            },
            {
                "Metric": "Jaccard Similarity (lexical homogenization)",
                "Mean": df[LEXICAL_SCORE_COLUMN].mean(),
                "SD": df[LEXICAL_SCORE_COLUMN].std(),
                "Min": df[LEXICAL_SCORE_COLUMN].min(),
                "Max": df[LEXICAL_SCORE_COLUMN].max(),
            },
        ]
    )

    numeric_columns = ["Mean", "SD", "Min", "Max"]
    table[numeric_columns] = table[numeric_columns].round(3)

    return table


def save_table(table: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    """Save Table 4.2 as CSV and Excel."""
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / "table_4_2_overall_homogenization_scores.csv"
    output_excel = output_dir / "table_4_2_overall_homogenization_scores.xlsx"

    table.to_csv(output_csv, index=False, encoding="utf-8-sig")
    table.to_excel(output_excel, index=False)

    return output_csv, output_excel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Table 4.2 from detailed BERTScore and Jaccard results."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to detailed quantitative results CSV. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for output CSV and Excel files. "
            "Default: the same folder as the input CSV."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Cannot find input file: {input_path}")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else input_path.parent
    )

    df = pd.read_csv(input_path)
    table_4_2 = generate_table_4_2(df)
    output_csv, output_excel = save_table(table_4_2, output_dir)

    print("\nTable 4.2 generated successfully.\n")
    print(table_4_2.to_string(index=False))
    print(f"\nSaved CSV file to:   {output_csv}")
    print(f"Saved Excel file to: {output_excel}")

    print(
        "\nNote. For each error unit, BERTScore F1 and Jaccard Similarity "
        "were first averaged across the three pairwise model comparisons. "
        "The table then reports the mean, standard deviation, minimum, and "
        "maximum values across the error units. Values are rounded to three "
        "decimal places."
    )


if __name__ == "__main__":
    main()
