from pathlib import Path
import os
import re
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets"

INPUT_WORKORDER_FILE = DATASET_DIR / "valid-WKO_and_component_times.csv"
INPUT_CLUSTERED_FILE = DATASET_DIR / "dataset.csv"
OUTPUT_FOLDER = BASE_DIR / "cluster_outputs"
OUTPUT_JOINED_FILE = DATASET_DIR / "final_joined_output.csv"


def clean_workorder_column(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def load_data(
    workorder_file: Path = INPUT_WORKORDER_FILE,
    clustered_file: Path = INPUT_CLUSTERED_FILE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(workorder_file)
    dataset = pd.read_csv(clustered_file)
    return df, dataset


def build_joined_dataset(df: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dataset = dataset.copy()

    # Clean workorder columns
    df["WKO#"] = clean_workorder_column(df["WKO#"])
    dataset["WKO#"] = clean_workorder_column(dataset["WKO#"])

    # Convert dates
    df["Date_Opened"] = pd.to_datetime(df["Date_Opened"], errors="coerce")
    df["Date_Closed"] = pd.to_datetime(df["Date_Closed"], errors="coerce")

    # Drop unwanted columns
    df = df.drop(columns=["Part#", "TSO"], errors="ignore")

    # Aggregate per workorder
    agg_df = (
        df.groupby("WKO#", as_index=False)
        .agg(
            {
                "Total_Time": "sum",
                "Date_Opened": "min",
                "Date_Closed": "max",
                "Registration#": "first",
            }
        )
        .rename(
            columns={
                "WKO#": "workorder",
                "Total_Time": "total_time",
                "Date_Opened": "date_time_opened",
                "Date_Closed": "date_time_closed",
                "Registration#": "registration",
            }
        )
    )

    dataset = dataset.rename(columns={"WKO#": "workorder"})

    # Clean renamed join columns
    agg_df["workorder"] = agg_df["workorder"].astype(str).str.strip()
    dataset["workorder"] = dataset["workorder"].astype(str).str.strip()

    # Print overlap stats
    agg_wko = set(agg_df["workorder"])
    dataset_wko = set(dataset["workorder"])

    print("Unique workorders in agg_df:", len(agg_wko))
    print("Unique workorders in dataset:", len(dataset_wko))
    print("Matching unique workorders:", len(agg_wko & dataset_wko))
    print("Only in agg_df:", len(agg_wko - dataset_wko))
    print("Only in dataset:", len(dataset_wko - agg_wko))

    # Join
    final_df = pd.merge(agg_df, dataset, on="workorder", how="inner")

    final_df = final_df.rename(
        columns={
            "ATA_Code": "ata_code",
            "PROBLEM": "problem",
            "DATE": "date",
            "ACTION": "action",
        }
    )

    final_df = final_df[
        [
            "workorder",
            "date_time_opened",
            "date_time_closed",
            "registration",
            "total_time",
            "ata_code",
            "problem",
            "date",
            "action",
            "cluster",
            "cluster_name",
        ]
    ]

    print("df shape:", df.shape)
    print("dataset shape:", dataset.shape)
    print("agg_df shape:", agg_df.shape)
    print("final_df shape:", final_df.shape)

    return final_df


def save_cluster_csvs(
    final_df: pd.DataFrame,
    output_folder: Path = OUTPUT_FOLDER,
) -> None:
    os.makedirs(output_folder, exist_ok=True)

    for cluster_name, group in final_df.groupby("cluster"):
        safe_name = re.sub(r"[^\w]+", "_", str(cluster_name))
        file_name = output_folder / f"cluster_{safe_name}.csv"
        group.to_csv(file_name, index=False)
        print(f"{cluster_name}: {len(group)} rows saved to {file_name}")

    print("Cluster-wise CSV files generated successfully.")


def save_full_output(
    final_df: pd.DataFrame,
    output_file: Path = OUTPUT_JOINED_FILE,
) -> None:
    os.makedirs(output_file.parent, exist_ok=True)
    final_df.to_csv(output_file, index=False)
    print(f"Full joined output saved to {output_file}")


def main() -> None:
    os.makedirs(DATASET_DIR, exist_ok=True)

    df, dataset = load_data()
    final_df = build_joined_dataset(df, dataset)

    save_full_output(final_df)
    save_cluster_csvs(final_df)


if __name__ == "__main__":
    main()