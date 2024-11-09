import gc
import os
import libpysal
import pandas as pd
from esda.moran import Moran
from tqdm import tqdm
import warnings


def clean_dataframe(
    df: pd.DataFrame, drop_rate: float = 0.5, verbose: bool = True
) -> pd.DataFrame:
    if drop_rate < 0 or drop_rate > 1:
        raise ValueError("drop_rate must be between 0 and 1")
    shape = df.shape
    df = df[df.columns[df.isna().sum() / df.shape[0] <= drop_rate]]
    df = df[df.columns[df.nunique() != 1]]
    df.drop_duplicates(inplace=True)
    if verbose:
        print(f"Cleaned DataFrame shape: {df.shape}")
        print(f"{shape[0] - df.shape[0]} rows were dropped")
        print(f"{shape[1] - df.shape[1]} columns were dropped")
    return df


df_kenya_birds1_raw = pd.read_csv("./data/simple/simple.csv", sep="\t")
df_kenya_birds1 = clean_dataframe(df_kenya_birds1_raw)
print(df_kenya_birds1.columns)

df = df_kenya_birds1[
    ["order", "decimalLongitude", "decimalLatitude", "individualCount"]
].copy()
print(f"DataFrame shape: {df.shape}")
# df = df.loc[df["individualCount"] <= 5]
print("max count:", df["individualCount"].max())
print(df.isna().sum())
df.dropna(inplace=True)
print(f"DataFrame shape: {df.shape}")


morrans = {}
thresholds = {}
lonely_point_counts = {}

if "morans.csv" in os.listdir():
    morans_df = pd.read_csv("morans.csv", index_col=0)
    calculated_orders = morans_df.index
    morrans = morans_df[["morans_I", "morans_p_sim"]].T.to_dict(orient="list")
    thresholds = morans_df["threshold"].to_dict()
    lonely_point_counts = morans_df["lonely_points"].to_dict()
    print(
        f"Loaded previous results,{calculated_orders=}\n{morrans=}\n{thresholds=}\n{lonely_point_counts=}"
    )

for order, data in tqdm(df.groupby("order")):
    if order in calculated_orders:
        print(f"Skipping {order}")
        continue
    n_lonely_points: dict[float, int] = {}
    # maps threshold to number of lonely points
    for i in tqdm(range(1, 40), desc=f"Order: {order: <20}"):
        threshold = (i / 3) ** 0.3
        print(data.shape, threshold)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            w = libpysal.weights.DistanceBand.from_array(
                data[["decimalLongitude", "decimalLatitude"]].values,
                threshold=threshold,
            )
        n_lonely_points[threshold] = (
            nlp := sum([1 for v in w.neighbors.values() if len(v) == 0])
        )
        if nlp == 0:
            lonely_point_counts[order] = 0
            print(f"Threshold for {order}: {threshold}")
            break
        else:
            n_lonely_points[threshold] = sum(
                [1 for v in w.neighbors.values() if len(v) == 0]
            )
    else:
        threshold = max(n_lonely_points, key=n_lonely_points.get)  # type: ignore
        print(
            f"Threshold for {order} not found, using {threshold}, which has {n_lonely_points[threshold]} lonely points (smallest number of lonely points)"
        )
        lonely_point_counts[order] = n_lonely_points[threshold]

    # assert len(w.neighbors) != 1, "No neighbors found"
    thresholds[order] = threshold
    moran = Moran(data["individualCount"].values, w)
    morrans[order] = [moran.I, moran.p_sim]

    results_df = pd.DataFrame(
        [morrans, thresholds, lonely_point_counts],
        index=["morans", "threshold", "lonely_points"],
    ).T

    results_df[["morans_I", "morans_p_sim"]] = pd.DataFrame(
        results_df["morans"].tolist(), index=results_df.index
    )
    results_df.drop(columns=["morans"], inplace=True)
    print(results_df)
    results_df.to_csv("morans.csv")

    del w
    del moran
    gc.collect()
