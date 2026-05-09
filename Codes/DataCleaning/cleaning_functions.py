"""
Reusable data cleaning functions for the Predictive Maintenance ML project.

This module contains only data loading, validation, cleaning, basic feature engineering,
categorical encoding, and target distribution summary functions.

It intentionally does NOT contain:
- train-test split logic,
- resampling / SMOTE / imbalance handling,
- model training,
- model evaluation.

Author: MLBeadando project team
"""

from pathlib import Path
from typing import Union

import pandas as pd


PathLike = Union[str, Path]


def load_raw_data(data_path: PathLike) -> pd.DataFrame:
    """
    Load the raw Predictive Maintenance dataset from a CSV file.

    Parameters
    ----------
    data_path : str or pathlib.Path
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataframe loaded from the CSV file.
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    return pd.read_csv(data_path)


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename raw dataset columns to consistent Python-friendly column names.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe with original column names.

    Returns
    -------
    pd.DataFrame
        Dataframe with renamed columns.
    """
    df = df.copy()

    column_mapping = {
        "UDI": "udi",
        "Product ID": "product_id",
        "Type": "type",
        "Air temperature [K]": "air_temperature_k",
        "Process temperature [K]": "process_temperature_k",
        "Rotational speed [rpm]": "rotational_speed_rpm",
        "Torque [Nm]": "torque_nm",
        "Tool wear [min]": "tool_wear_min",
        "Target": "target",
        "Failure Type": "failure_type",
    }

    return df.rename(columns=column_mapping)


def validate_cleaned_data(df: pd.DataFrame) -> None:
    """
    Run basic validation checks on the dataframe after column renaming.

    The function checks:
    - required columns are present,
    - target contains only 0 and 1,
    - type contains only expected categories: L, M, H,
    - there are no missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe after column renaming.

    Raises
    ------
    ValueError
        If required columns are missing, unexpected values are found,
        or missing values exist.
    """
    required_columns = [
        "udi",
        "product_id",
        "type",
        "air_temperature_k",
        "process_temperature_k",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
        "target",
        "failure_type",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    allowed_target_values = {0, 1}
    actual_target_values = set(df["target"].dropna().unique())

    if not actual_target_values.issubset(allowed_target_values):
        raise ValueError(
            f"Unexpected target values found: {actual_target_values}. "
            "Expected only 0 and 1."
        )

    allowed_type_values = {"L", "M", "H"}
    actual_type_values = set(df["type"].dropna().unique())

    if not actual_type_values.issubset(allowed_type_values):
        raise ValueError(
            f"Unexpected type categories found: {actual_type_values}. "
            "Expected only L, M and H."
        )

    missing_value_count = df.isna().sum().sum()

    if missing_value_count > 0:
        raise ValueError(
            f"The dataset contains missing values. Total missing values: {missing_value_count}"
        )


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple engineered features based on existing numerical columns.

    Currently added feature:
    - temperature_difference_k:
      process_temperature_k - air_temperature_k

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe with renamed columns.

    Returns
    -------
    pd.DataFrame
        Dataframe with engineered features.
    """
    df = df.copy()

    df["temperature_difference_k"] = (
        df["process_temperature_k"] - df["air_temperature_k"]
    )

    return df


def prepare_modeling_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the dataframe for binary classification modeling.

    Steps:
    - remove identifier columns,
    - remove failure_type because it is directly related to the target
      and would cause data leakage in binary target prediction,
    - one-hot encode the type variable using drop_first=True.

    Notes
    -----
    The type variable has three categories: L, M and H.
    With drop_first=True, only two dummy columns are created.
    The dropped category is used as the reference category.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe with engineered features.

    Returns
    -------
    pd.DataFrame
        Modeling-ready dataframe.
    """
    df = df.copy()

    columns_to_drop = [
        "udi",
        "product_id",
        "failure_type",
    ]

    df = df.drop(columns=columns_to_drop)

    df = pd.get_dummies(
        df,
        columns=["type"],
        drop_first=True,
        dtype=int,
    )

    return df


def clean_predictive_maintenance_data(
    data_path: PathLike,
    return_cleaned: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full cleaning pipeline for the Predictive Maintenance dataset.

    Pipeline steps:
    1. Load raw CSV file.
    2. Rename columns.
    3. Validate required structure and values.
    4. Add engineered features.
    5. Prepare modeling-ready dataframe.

    Parameters
    ----------
    data_path : str or pathlib.Path
        Path to the raw CSV file.

    return_cleaned : bool, default=False
        If True, returns both:
        - df_model: modeling-ready dataframe,
        - df_cleaned: cleaned dataframe before dropping identifiers,
          dropping failure_type and one-hot encoding.

    Returns
    -------
    pd.DataFrame
        Modeling-ready dataframe.

    or

    tuple[pd.DataFrame, pd.DataFrame]
        If return_cleaned=True, returns (df_model, df_cleaned).
    """
    df_raw = load_raw_data(data_path)

    df_cleaned = rename_columns(df_raw)

    validate_cleaned_data(df_cleaned)

    df_cleaned = add_engineered_features(df_cleaned)

    df_model = prepare_modeling_dataset(df_cleaned)

    if return_cleaned:
        return df_model, df_cleaned

    return df_model


def create_target_summary(
    df: pd.DataFrame,
    target_column: str = "target",
) -> pd.DataFrame:
    """
    Create a summary table for the target variable distribution.

    This function is useful for identifying class imbalance.
    It does NOT perform imbalance handling.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the target column.

    target_column : str, default="target"
        Name of the target column.

    Returns
    -------
    pd.DataFrame
        Summary dataframe with count and ratio_percent columns.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column not found in dataframe: {target_column}")

    target_counts = df[target_column].value_counts().sort_index()
    target_ratios = df[target_column].value_counts(normalize=True).sort_index() * 100

    target_summary = pd.DataFrame(
        {
            "count": target_counts,
            "ratio_percent": target_ratios.round(2),
        }
    )

    return target_summary


def check_basic_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a simple data quality summary for each column.

    The summary contains:
    - dtype,
    - missing_count,
    - missing_ratio_percent,
    - unique_values.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Data quality summary.
    """
    summary = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing_count": df.isna().sum(),
            "missing_ratio_percent": (df.isna().mean() * 100).round(2),
            "unique_values": df.nunique(),
        }
    )

    return summary