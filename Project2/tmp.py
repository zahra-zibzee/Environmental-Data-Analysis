import gc
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
for order, data in tqdm(df.groupby("order")):
    for i in tqdm(range(1, 10), desc=order):
        threshold = (i / 3) ** 0.3
        print(data.shape, threshold)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            w = libpysal.weights.DistanceBand.from_array(
                data[["decimalLongitude", "decimalLatitude"]].values,
                threshold=threshold,
            )
        if len(w.neighbors) == 1:
            break
    else:
        print(
            f"Threshold for {order} not found, using {threshold} instead and biggest neighbor"
        )
        data = data.copy()
        lonely_points = [i for i in range(data.shape[0]) if len(w.neighbors[i]) <= 1]
        # remove lonely points
        data.drop(lonely_points, inplace=True)
        w = libpysal.weights.DistanceBand.from_array(
            data[["decimalLongitude", "decimalLatitude"]].values, threshold=threshold
        )
    assert len(w.neighbors) != 1, "No neighbors found"
    thresholds[order] = threshold
    print(f"Threshold for {order}: {threshold}")
    moran = Moran(data["individualCount"].values, w)
    print(moran.I, moran.p_sim)
    morrans[order] = [moran.I, moran.p_sim]

print(morrans, thresholds)
