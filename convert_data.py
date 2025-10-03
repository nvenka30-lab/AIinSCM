"""Utilities for converting the sales history workbook to CSV format.

This script reads the provided Excel workbook containing the historical
sales records and persists an equivalent comma-separated values file.
The CSV representation is faster to load for repeated experimentation,
which keeps the exploratory analysis and forecasting workflows
lightweight.
"""
from __future__ import annotations

import pathlib

import pandas as pd


EXCEL_PATH = pathlib.Path("Sales_History_Dataset.xlsx")
"""pathlib.Path: Location of the source Excel workbook."""

CSV_PATH = pathlib.Path("Sales_History_Dataset.csv")
"""pathlib.Path: Destination path for the generated CSV file."""


def convert_excel_to_csv(
    excel_path: pathlib.Path = EXCEL_PATH,
    csv_path: pathlib.Path = CSV_PATH,
    *,
    engine: str = "openpyxl",
) -> pathlib.Path:
    """Load the Excel workbook and save its first sheet to CSV format.

    Parameters
    ----------
    excel_path:
        Filesystem path to the source ``.xlsx`` file that should be converted.
    csv_path:
        Location where the resulting CSV file will be written.
    engine:
        Excel reader engine passed to :func:`pandas.read_excel`. The default
        ``"openpyxl"`` works for standard ``.xlsx`` files.

    Returns
    -------
    pathlib.Path
        Path to the newly created CSV file.

    Raises
    ------
    FileNotFoundError
        If the ``excel_path`` does not exist on disk.
    """

    if not excel_path.exists():
        raise FileNotFoundError(
            f"Source Excel workbook not found at {excel_path.resolve()}"
        )

    dataframe = pd.read_excel(excel_path, engine=engine)
    dataframe.to_csv(csv_path, index=False)
    return csv_path


def main() -> None:
    """Command-line entry point for converting the Excel dataset to CSV."""

    output_path = convert_excel_to_csv()
    print(f"Saved CSV dataset to {output_path.resolve()}")


if __name__ == "__main__":
    main()
