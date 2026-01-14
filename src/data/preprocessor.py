"""Data preprocessing utilities for CSV/TSV data."""

import logging
import re
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocessor for CSV/TSV data files."""

    def __init__(self, cfg: DictConfig):
        """Initialize preprocessor with configuration.

        Args:
            cfg: Data configuration from Hydra.
        """
        self.cfg = cfg
        self.format_cfg = cfg.format
        self.preprocess_cfg = cfg.preprocessing
        self.columns_cfg = cfg.columns

    def load_file(self, file_path: str | Path) -> pd.DataFrame:
        """Load a CSV/TSV file into a DataFrame.

        Args:
            file_path: Path to the data file.

        Returns:
            Loaded DataFrame.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        delimiter = self.format_cfg.delimiter
        if delimiter == "\\t":
            delimiter = "\t"

        df = pd.read_csv(
            file_path,
            delimiter=delimiter,
            encoding=self.format_cfg.encoding,
            header=0 if self.format_cfg.header else None,
        )

        logger.info(f"Loaded {len(df)} rows from {file_path}")
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing steps to the DataFrame.

        Args:
            df: Input DataFrame.

        Returns:
            Preprocessed DataFrame.
        """
        original_len = len(df)

        if self.preprocess_cfg.strip_whitespace:
            df = self._strip_whitespace(df)

        if self.preprocess_cfg.lowercase:
            df = self._lowercase_text(df)

        if self.preprocess_cfg.remove_duplicates:
            df = self._remove_duplicates(df)

        df = self._filter_by_length(df)

        logger.info(f"Preprocessing: {original_len} -> {len(df)} rows")
        return df

    def _strip_whitespace(self, df: pd.DataFrame) -> pd.DataFrame:
        """Strip whitespace from string columns."""
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str).str.strip()
        return df

    def _lowercase_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert text columns to lowercase."""
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str).str.lower()
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        return df.drop_duplicates()

    def _filter_by_length(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter rows by text length."""
        min_len = self.preprocess_cfg.min_length
        max_len = self.preprocess_cfg.max_length

        required_cols = self._get_required_text_columns()
        for col in required_cols:
            if col in df.columns:
                df = df[df[col].astype(str).str.len() >= min_len]
                df = df[df[col].astype(str).str.len() <= max_len]

        return df

    def _get_required_text_columns(self) -> list[str]:
        """Get list of required text columns for length filtering.

        Only filters on instruction/output (for SFT) or prompt/chosen/rejected (for DPO).
        Does not filter on optional 'input' column.
        """
        cols = []
        if self.columns_cfg.instruction:
            cols.append(self.columns_cfg.instruction)
        if self.columns_cfg.output:
            cols.append(self.columns_cfg.output)
        if self.columns_cfg.chosen:
            cols.append(self.columns_cfg.chosen)
        if self.columns_cfg.rejected:
            cols.append(self.columns_cfg.rejected)
        return cols

    def _get_text_columns(self) -> list[str]:
        """Get list of all text columns from configuration."""
        cols = []
        for key in ["instruction", "input", "output", "chosen", "rejected"]:
            col_name = getattr(self.columns_cfg, key, None)
            if col_name:
                cols.append(col_name)
        return cols

    def validate_columns(self, df: pd.DataFrame) -> None:
        """Validate that required columns exist in DataFrame.

        Args:
            df: DataFrame to validate.

        Raises:
            ValueError: If required columns are missing.
        """
        required = []

        if self.columns_cfg.instruction:
            required.append(self.columns_cfg.instruction)

        if self.columns_cfg.output:
            required.append(self.columns_cfg.output)
        elif self.columns_cfg.chosen and self.columns_cfg.rejected:
            required.extend([self.columns_cfg.chosen, self.columns_cfg.rejected])

        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

    def to_instruction_format(self, df: pd.DataFrame) -> list[dict[str, str]]:
        """Convert DataFrame to instruction tuning format.

        Args:
            df: Preprocessed DataFrame.

        Returns:
            List of dictionaries with instruction format.
        """
        records = []
        instruction_col = self.columns_cfg.instruction
        input_col = self.columns_cfg.input
        output_col = self.columns_cfg.output

        for _, row in df.iterrows():
            record = {
                "instruction": str(row.get(instruction_col, "")),
                "output": str(row.get(output_col, "")),
            }
            if input_col and input_col in df.columns:
                record["input"] = str(row.get(input_col, ""))
            else:
                record["input"] = ""
            records.append(record)

        return records

    def to_preference_format(self, df: pd.DataFrame) -> list[dict[str, str]]:
        """Convert DataFrame to preference (DPO) format.

        Args:
            df: Preprocessed DataFrame.

        Returns:
            List of dictionaries with preference format.
        """
        records = []
        prompt_col = self.columns_cfg.instruction
        chosen_col = self.columns_cfg.chosen
        rejected_col = self.columns_cfg.rejected

        for _, row in df.iterrows():
            records.append({
                "prompt": str(row.get(prompt_col, "")),
                "chosen": str(row.get(chosen_col, "")),
                "rejected": str(row.get(rejected_col, "")),
            })

        return records

    def to_chat_format(
        self,
        df: pd.DataFrame,
        system_message: str | None = None,
    ) -> list[dict[str, Any]]:
        """Convert DataFrame to chat format.

        Args:
            df: Preprocessed DataFrame.
            system_message: Optional system message to prepend.

        Returns:
            List of chat conversations.
        """
        records = []
        instruction_col = self.columns_cfg.instruction
        input_col = self.columns_cfg.input
        output_col = self.columns_cfg.output

        for _, row in df.iterrows():
            messages = []

            if system_message:
                messages.append({"role": "system", "content": system_message})

            user_content = str(row.get(instruction_col, ""))
            if input_col and input_col in df.columns and row.get(input_col):
                user_content += f"\n\n{row.get(input_col)}"

            messages.append({"role": "user", "content": user_content})
            messages.append({
                "role": "assistant",
                "content": str(row.get(output_col, "")),
            })

            records.append({"messages": messages})

        return records


def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and special characters.

    Args:
        text: Input text.

    Returns:
        Cleaned text.
    """
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def apply_template(
    record: dict[str, str],
    template: str,
    template_no_input: str | None = None,
) -> str:
    """Apply a prompt template to a record.

    Args:
        record: Dictionary with instruction, input, output keys.
        template: Template string with placeholders.
        template_no_input: Template to use when input is empty.

    Returns:
        Formatted prompt string.
    """
    if template_no_input and not record.get("input"):
        return template_no_input.format(**record)
    return template.format(**record)
