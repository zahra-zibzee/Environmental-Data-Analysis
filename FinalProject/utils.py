from functools import lru_cache
import glob
import json
import os
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import laspy
import cv2
from tqdm import tqdm
from scipy.ndimage import label, center_of_mass


def lidar_to_point_cloud(lidar_file: str | os.PathLike) -> np.ndarray:
    """Read a LiDAR file and return a NumPy array of points."""
    las = laspy.read(lidar_file)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    return points


@lru_cache
def read_geojson(
    geojson_address: str | os.PathLike = "data/field_survey.geojson",
) -> pd.DataFrame:
    """Read a geojson file into a DataFrame."""
    with open(geojson_address) as f:
        data = json.load(f)
    data = pd.DataFrame([i["properties"] | i["geometry"] for i in data["features"]])
    data["x"] = data["coordinates"].apply(lambda x: x[0])
    data["y"] = data["coordinates"].apply(lambda x: x[1])
    data = data.drop(columns=["coordinates"])
    return data


def tree_scope_definition(
    depth_img_df: pd.DataFrame,
    plot_num: int,
    geojson_address: str | os.PathLike = "data/field_survey.geojson",
) -> pd.DataFrame:
    data = read_geojson(geojson_address)
    data = data[data["plot"] == plot_num]

    depth_img_df = depth_img_df.drop_duplicates().copy()
    depth_img_df["label"] = 0

    # label data points from depth_img_df that are in the circle, with the center of x, y and radius of dbh from data
    not_founed_trees = 0
    not_founed_trees_r = []

    for index, row in data.iterrows():
        x, y, r = row["x"], row["y"], row["dbh"]
        r /= 100  # converting dbh to meters and dividing by 2 to get the radius
        # tmp = depth_img_df[
        #     (depth_img_df["x"] >= x - r)
        #     & (depth_img_df["x"] <= x + r)
        #     & (depth_img_df["y"] >= y - r)
        #     & (depth_img_df["y"] <= y + r)
        # ]
        distances = np.sqrt(
            (depth_img_df["x"] - x) ** 2 + (depth_img_df["y"] - y) ** 2
        ).sort_values()
        # tmp = distances.iloc[:40]
        tmp = distances[distances <= r]
        if tmp.empty:
            # print(f"No points in the circle with center {x, y} and radius {r}")
            not_founed_trees += 1
            not_founed_trees_r.append(r)
            continue
        # adding label to the points in the circle
        depth_img_df.loc[tmp.index, "label"] = index
        # assert (
        #     depth_img_df.groupby(["x", "y"])["label"].count().max() == 1
        # ), f"There are multiple points in the same x, y, {tmp}, {index}"
        # print("dede")
    # print(f"{not_founed_trees} trees were not found in the depth image")
    # print("The average radius of the not found trees is:", np.mean(not_founed_trees_r))
    return depth_img_df


def take_photo_from_top(points, step=0.1, remove_outliers=True):
    """Take a photo from the top of the 3D points."""
    points = points.copy()
    assert "x" in points.columns and "y" in points.columns and "z" in points.columns
    assert step > 0
    # assert (
    #     a := points.groupby(["x", "y"])["label"].nunique()
    # ).max() == 1, f"There are multiple points in the same x, y, but {a[a > 1]} has multiple points "
    # print(points)
    points[["x", "y"]] = points[["x", "y"]] // step * step
    depth_map = points.groupby(["x", "y"])[["z", "label"]].max().reset_index()
    if remove_outliers:
        Q1 = np.percentile(depth_map["z"], 25)
        Q3 = np.percentile(depth_map["z"], 75)
        IQR = Q3 - Q1
        depth_map = depth_map[
            (depth_map["z"] >= Q1 - 1.5 * IQR) & (depth_map["z"] <= Q3 + 1.5 * IQR)
        ]
    # add back the label column using join
    # depth_map["label"] = 0
    # depth_map = points[["x", "y", "z"]].merge(depth_map, on=["x", "y"], how="left")
    return depth_map


def create_depth_img(depth_map: pd.DataFrame) -> np.ndarray:
    """Create a depth map from 3D points."""

    for col in ["x", "y", "z"]:
        depth_map[col] -= depth_map[col].min()
        # we can not ues depth_map -= depth_map.min() because we need to preserve the columns dtypes
        # and also, we dnot want to do it for all columns
    # the xmapper and y_mapper are used to map the x and y values to the image
    x_mapper = {v: i for i, v in enumerate(sorted(depth_map["x"].unique()))}
    y_mapper = {v: i for i, v in enumerate(sorted(depth_map["y"].unique()))}
    depth_map["x_index"] = depth_map["x"].map(x_mapper)
    depth_map["y_index"] = depth_map["y"].map(y_mapper)
    img = np.zeros((len(x_mapper), len(y_mapper)))
    img[depth_map["x_index"], depth_map["y_index"]] = depth_map["z"]
    img = (img / img.max() * 255).astype(np.uint8)
    img = cv2.equalizeHist(img)

    mask = np.zeros((len(x_mapper), len(y_mapper)))
    tree_depth_map = depth_map[depth_map["label"] != 0]
    mask[tree_depth_map["x_index"], tree_depth_map["y_index"]] = 1
    # print(img.shape)

    missing_pixels = np.ones_like(img)
    missing_pixels[depth_map["x_index"], depth_map["y_index"]] = 0

    return img, mask, missing_pixels


def calculate_rotation_angle(points, orientation="landscape"):
    """
    Calculate the angle of rotation for a quadrilateral based on its four corner points.

    Parameters:
    points (list or np.ndarray): A list or array of four points, each represented as [x, y].
                                 The points should be in any order.
    orientation (str): The orientation of the quadrilateral. Can be either "landscape" or "vertical".

    Returns:
    float: The rotation angle (in degrees) of the top edge with respect to the horizontal axis.
    """
    # Ensure points are a NumPy array
    points = np.array(points)

    # Sort points by their y-coordinates (to distinguish top and bottom)
    points = points[np.argsort(points[:, 1])]

    # Extract top and bottom points
    top_points = points[:2]  # First two are the top points
    bottom_points = points[2:]  # Last two are the bottom points

    # Further sort top and bottom points by their x-coordinates
    top_left, top_right = top_points[np.argsort(top_points[:, 0])]
    bottom_left, bottom_right = bottom_points[np.argsort(bottom_points[:, 0])]

    # Calculate the angle of the top edge
    dx = top_right[0] - top_left[0]
    dy = top_right[1] - top_left[1]
    angle = np.degrees(np.arctan2(dy, dx))

    # Adjust the angle based on the orientation
    if orientation == "landscape" and abs(dx) < abs(dy):  # If height > width
        angle += 90 if dy > 0 else -90
    elif orientation == "portrait" and abs(dx) > abs(dy):  # If width > height
        angle += 90 if dx > 0 else -90

    return angle


def turn_points(points, angle, orientation="landscape"):
    """
    Rotate a set of points by a given angle around the origin.

    Parameters:
    points (np.ndarray): An array of points, each represented as [x, y].
    angle (float): The angle of rotation (in degrees).
    orientation (str): The orientation of the quadrilateral. Can be either "landscape" or "vertical".

    Returns:
    np.ndarray: An array of rotated points.
    """
    if isinstance(points, pd.DataFrame):
        columns = points.columns
        points = points.values
    else:
        columns = None

    angle = np.radians(angle)

    # Define the rotation matrix
    rotation_matrix = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )

    # Rotate the points
    rotated_points = points @ rotation_matrix

    width = rotated_points[:, 0].max() - rotated_points[:, 0].min()
    height = rotated_points[:, 1].max() - rotated_points[:, 1].min()

    if (orientation == "landscape" and height < width) or (
        orientation == "portrait" and width < height
    ):
        rotated_points = rotated_points[:, ::-1]

    if columns is not None:
        rotated_points = pd.DataFrame(rotated_points, columns=columns)

    return rotated_points


def create_dataset(
    data_folder: str | os.PathLike,
    step: float = 0.1,
    remove_outliers: bool = True,
    orientation: str = "landscape",
    max_file: int = -1,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Create a dataset from the LiDAR data."""
    data_folder = Path(data_folder)
    depth_imgs, masks, missing_pixels, angles = [], [], [], []
    for index, las_file in enumerate(
        tqdm(list(data_folder.glob("*.las")), desc="Reading LiDAR files")
    ):
        if index == max_file:
            print(f"Reached the maximum number of files ({max_file})")
            break
        plot_num = int(os.path.splitext(os.path.basename(las_file))[0].split("_")[-1])
        raw_depth_map = pd.DataFrame(
            lidar_to_point_cloud(las_file), columns=["x", "y", "z"]
        )
        gt_depth_map = tree_scope_definition(raw_depth_map, plot_num)

        most_left_point = gt_depth_map.loc[gt_depth_map["x"].idxmin()]
        most_right_point = gt_depth_map.loc[gt_depth_map["x"].idxmax()]

        most_bottom_point = gt_depth_map.loc[gt_depth_map["y"].idxmin()]
        most_top_point = gt_depth_map.loc[gt_depth_map["y"].idxmax()]

        angle = calculate_rotation_angle(
            [
                [most_left_point["x"], most_left_point["y"]],
                [most_right_point["x"], most_right_point["y"]],
                [most_bottom_point["x"], most_bottom_point["y"]],
                [most_top_point["x"], most_top_point["y"]],
            ],
            orientation=orientation,
        )
        if orientation:
            gt_depth_map[["x", "y"]] = turn_points(
                gt_depth_map[["x", "y"]], angle, orientation
            )

        depth_map = take_photo_from_top(
            gt_depth_map, step=step, remove_outliers=remove_outliers
        )
        depth_img, mask, missing_pixel = create_depth_img(depth_map)
        depth_imgs.append(depth_img)
        masks.append(mask)
        missing_pixels.append(missing_pixel)
        angles.append(angle)
    return depth_imgs, masks, missing_pixels, angles


def split_image_into_patches(
    img: np.ndarray, patch_size: tuple[int, int], stride: int = None
) -> list[np.ndarray]:
    """Split an image into patches."""
    if stride is None:
        stride = patch_size[0] // 2

    patches = []
    for i in range(0, img.shape[0] - patch_size[0] + 1, stride):
        for j in range(0, img.shape[1] - patch_size[1] + 1, stride):
            patch = img[i : i + patch_size[0], j : j + patch_size[1]]
            assert (
                patch.shape == patch_size
            ), f"Patch shape is {patch.shape}, expected {patch_size}"
            patches.append(patch)
    return patches


def find_tree_locations(
    segmentation_map: np.ndarray, kernel_size: int = 3
) -> np.ndarray:
    """
    Extract tree locations from a binary segmentation map using morphological transforms
    and connected components analysis.

    Parameters:
        segmentation_map (np.ndarray): Binary segmentation map of shape (H, W),
                                        where 1 represents tree regions, and 0 represents the background.
        kernel_size (int): Size of the structuring element for morphological operations.

    Returns:
        np.ndarray: Array of tree locations as (x, y) coordinates.
    """
    assert (
        segmentation_map.ndim == 2
    ), f"Segmentation map must be 2D, but got shape {segmentation_map.shape}"
    assert np.unique(segmentation_map).tolist() == [
        0,
        1,
    ], f"Segmentation map must be binary, but contains values {np.unique(segmentation_map)}"

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned_map = cv2.morphologyEx(segmentation_map, cv2.MORPH_OPEN, kernel)

    labeled_map, num_features = label(cleaned_map)
    tree_locations = np.array(
        center_of_mass(cleaned_map, labeled_map, range(1, num_features + 1))
    )
    return tree_locations


if __name__ == "__main__":
    args = {
        "data_folder": "./data/als",
        "step": 0.1,
        "remove_outliers": True,
        "orientation": True,
        "max_file": -1,
    }
    patch_size = (128, 128)
    overlap = 50
    outfolder = Path("data/patches")

    depth_imgs, masks, missing_pixels, angles = create_dataset(**args)
    all_depth_patches = []
    all_mask_patches = []
    for mask, img in tqdm(
        zip(masks, depth_imgs),
        desc="Splitting images into patches",
        total=len(depth_imgs),
    ):
        mask_patches = split_image_into_patches(
            mask, patch_size=patch_size, stride=overlap
        )
        img_patches = split_image_into_patches(
            img, patch_size=patch_size, stride=overlap
        )
        all_mask_patches.extend(mask_patches)
        all_depth_patches.extend(img_patches)

    if not outfolder.exists():
        outfolder.mkdir(parents=True)

    for i, (mask, img) in tqdm(
        enumerate(zip(all_mask_patches, all_depth_patches)),
        desc="Saving patches",
        total=len(all_mask_patches),
    ):
        cv2.imwrite(str(outfolder / f"depth_{i}.png"), img)
        np.save(str(outfolder / f"mask_{i}.npy"), mask)
    print(f"Saved {len(all_mask_patches)} patches to {outfolder}")
