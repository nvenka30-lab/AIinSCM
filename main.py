"""Main entry point for sales forecasting exploratory analysis.

This script loads the sales history dataset, performs initial exploratory
analysis, and creates baseline visualizations that will support downstream
forecasting work.
"""
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATA_PATH = pathlib.Path("Sales_History_Dataset.csv")
OUTPUT_DIR = pathlib.Path("outputs")


COLUMN_RENAMES = {
    "Invoice Date (Shipment Delivered)": "Invoice Date",
}

DATE_COLUMNS = [
    "Order_Date",
    "Confirmed_Delivery_Date",
    "Invoice Date",
]


def load_dataset(path: pathlib.Path) -> pd.DataFrame:
    """Load the CSV dataset and standardize date columns.

    Parameters
    ----------
    path:
        Filesystem path to the CSV dataset containing the sales history.

    Returns
    -------
    pandas.DataFrame
        Raw dataset with normalized column names and parsed datetime columns.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path.resolve()}")

    df = pd.read_csv(path)

    # Standardize column names that may differ slightly from the specification.
    df = df.rename(columns=COLUMN_RENAMES)

    for column in DATE_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")

    return df


def describe_dataset(df: pd.DataFrame) -> None:
    """Print key exploratory summaries for the provided DataFrame.

    Parameters
    ----------
    df:
        DataFrame loaded from the sales history workbook.
    """
    print("=== Dataset Preview ===")
    print(df.head())

    print("\n=== Dataset Info ===")
    df.info()

    print("\n=== Numerical Summary ===")
    print(df.describe(include="all"))


def ensure_output_dir(directory: pathlib.Path) -> None:
    """Create the output directory tree if it is missing.

    Parameters
    ----------
    directory:
        Target folder where generated figures should be written.
    """
    directory.mkdir(parents=True, exist_ok=True)


def plot_daily_sales_trend(df: pd.DataFrame, directory: pathlib.Path) -> None:
    """Plot the total cost of sales by order date.

    Parameters
    ----------
    df:
        Sales history DataFrame containing ``Order_Date`` and ``Total_Cost`` columns.
    directory:
        Folder where the generated line chart will be saved.
    """
    if "Order_Date" not in df.columns or "Total_Cost" not in df.columns:
        print("Skipping daily sales trend plot due to missing columns.")
        return

    daily_sales = (
        df.dropna(subset=["Order_Date", "Total_Cost"])
        .groupby("Order_Date", as_index=False)["Total_Cost"]
        .sum()
        .sort_values("Order_Date")
    )

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=daily_sales, x="Order_Date", y="Total_Cost", marker="o")
    plt.title("Daily Sales Trend (Total Cost)")
    plt.xlabel("Order Date")
    plt.ylabel("Total Cost")
    plt.tight_layout()

    output_file = directory / "daily_sales_trend.png"
    plt.savefig(output_file)
    plt.close()
    print(f"Saved daily sales trend plot to {output_file}")


def plot_sales_distribution_by_product(df: pd.DataFrame, directory: pathlib.Path) -> None:
    """Plot the distribution of total cost by product name.

    Parameters
    ----------
    df:
        Sales history DataFrame containing ``Product_Name`` and ``Total_Cost`` columns.
    directory:
        Folder where the horizontal bar chart will be saved.
    """
    if "Product_Name" not in df.columns or "Total_Cost" not in df.columns:
        print("Skipping product distribution plot due to missing columns.")
        return

    product_sales = (
        df.dropna(subset=["Product_Name", "Total_Cost"])
        .groupby("Product_Name", as_index=False)["Total_Cost"]
        .sum()
        .sort_values("Total_Cost", ascending=False)
    )

    plt.figure(figsize=(14, 8))
    sns.barplot(data=product_sales, y="Product_Name", x="Total_Cost", color="#2a9d8f")
    plt.title("Sales Distribution by Product (Total Cost)")
    plt.xlabel("Total Cost")
    plt.ylabel("Product Name")
    plt.tight_layout()

    output_file = directory / "sales_distribution_by_product.png"
    plt.savefig(output_file)
    plt.close()
    print(f"Saved sales distribution plot to {output_file}")


def plot_quantity_time_series(df: pd.DataFrame, directory: pathlib.Path) -> None:
    """Plot the time series of quantity sold aggregated by order date.

    Parameters
    ----------
    df:
        Sales history DataFrame containing ``Order_Date`` and ``Quantity_Sold`` columns.
    directory:
        Folder where the quantity trend chart will be written.
    """
    if "Order_Date" not in df.columns or "Quantity_Sold" not in df.columns:
        print("Skipping quantity time series plot due to missing columns.")
        return

    daily_quantity = (
        df.dropna(subset=["Order_Date", "Quantity_Sold"])
        .groupby("Order_Date", as_index=False)["Quantity_Sold"]
        .sum()
        .sort_values("Order_Date")
    )

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=daily_quantity, x="Order_Date", y="Quantity_Sold", marker="o")
    plt.title("Quantity Sold Over Time")
    plt.xlabel("Order Date")
    plt.ylabel("Quantity Sold")
    plt.tight_layout()

    output_file = directory / "quantity_sold_time_series.png"
    plt.savefig(output_file)
    plt.close()
    print(f"Saved quantity sold time series plot to {output_file}")


def main() -> None:
    """Execute the exploratory data analysis workflow."""

    df = load_dataset(DATA_PATH)
    describe_dataset(df)

    ensure_output_dir(OUTPUT_DIR)

    plot_daily_sales_trend(df, OUTPUT_DIR)
    plot_sales_distribution_by_product(df, OUTPUT_DIR)
    plot_quantity_time_series(df, OUTPUT_DIR)


if __name__ == "__main__":
    main()
