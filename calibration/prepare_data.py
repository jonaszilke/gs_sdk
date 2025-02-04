import argparse
import json
import os
import sys

import cv2
import numpy as np
import yaml

from utils import load_csv_as_dict

folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if folder not in sys.path:
    sys.path.insert(1, folder)

from gs_sdk.gs_reconstruct import image2bgrxys

"""
This script prepares dataset for the tactile sensor calibration.
It is based on the collected and labeled data.

Prerequisite:
    - Tactile images collected using rectangular indenters with known dimensions are collected.
    - Collected tactile images are labeled.

Usage:
    python prepare_data.py --calib_dir CALIB_DIR [--config_path CONFIG_PATH] [--margin_reduction MARGIN_REDUCTION]

Arguments:
    --calib_dir: Path to the directory where the collected data will be saved
    --config_path: (Optional) Path to the configuration file about the sensor dimensions.
                   If not provided, GelSight Mini is assumed.
    --margin_reduction: (Optional) Reduce the size of the labeled rectangle. This helps guarantee all labeled pixels are indented.
                        If not provided, 4 pixels will be reduced.
"""

config_dir = os.path.join(os.path.dirname(__file__), "../examples/configs")


def create_rotated_rectangle_mask(shape, center, length, width1, width2, angle):
    """
    Creates a binary mask for a rotated, asymmetric rectangle without using OpenCV.

    Args:
        shape (tuple): Shape of the image (height, width).
        center (tuple): (x, y) center of the dividing line.
        length (float): Length of the dividing line.
        width1 (float): Width on one side of the line.
        width2 (float): Width on the other side of the line.
        angle (float): Angle of the dividing line in degrees.

    Returns:
        np.ndarray: Boolean mask with the same shape as the image.
    """
    mask = np.zeros(shape, dtype=bool)

    # Compute endpoints of the dividing line
    dx = (length / 2) * np.cos(np.radians(angle))
    dy = (length / 2) * np.sin(np.radians(angle))

    x1, y1 = center[0] - dx, center[1] - dy
    x2, y2 = center[0] + dx, center[1] + dy

    # Compute rectangle corners
    def rotate_point(x, y, cx, cy, theta):
        """Rotate (x, y) around (cx, cy) by theta degrees."""
        cos_t, sin_t = np.cos(np.radians(theta)), np.sin(np.radians(theta))
        x_new = cos_t * (x - cx) - sin_t * (y - cy) + cx
        y_new = sin_t * (x - cx) + cos_t * (y - cy) + cy
        return x_new, y_new

    corners = np.array([
        rotate_point(x1, y1 - width1, center[0], center[1], angle),
        rotate_point(x1, y1 + width1, center[0], center[1], angle),
        rotate_point(x2, y2 - width2, center[0], center[1], angle),
        rotate_point(x2, y2 + width2, center[0], center[1], angle),
    ])

    # Convert to integer pixel coordinates
    corners = np.round(corners).astype(int)

    # Find bounding box of the polygon
    x_min, y_min = np.clip(np.min(corners, axis=0), 0, [shape[1] - 1, shape[0] - 1])
    x_max, y_max = np.clip(np.max(corners, axis=0), 0, [shape[1] - 1, shape[0] - 1])

    # Scanline polygon fill algorithm
    def is_inside_polygon(x, y, corners):
        """Ray-casting algorithm to check if a point is inside a polygon."""
        num_vertices = len(corners)
        inside = False
        j = num_vertices - 1
        for i in range(num_vertices):
            xi, yi = corners[i]
            xj, yj = corners[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    # Fill the mask by checking each point within the bounding box
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            if is_inside_polygon(x, y, corners):
                mask[y, x] = True

    return mask



def prepare_data():
    # Argument Parsers
    parser = argparse.ArgumentParser(
        description="Use the labeled collected data to prepare the dataset files (npz)."
    )
    parser.add_argument(
        "-b",
        "--calib_dir",
        type=str,
        help="Path of the calibration data",
    )
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        help="Path of configuring GelSight",
        default=os.path.join(config_dir, "gsmini.yaml"),
    )
    parser.add_argument(
        "-m",
        "--margin_reduction",
        type=float,
        help="Reduce the size of the labeled rectangle. This helps guarantee all labeled pixels are indented.",
        default=4.0,
    )
    args = parser.parse_args()

    # Load the data_dict
    calib_dir = args.calib_dir
    catalog_path = os.path.join(calib_dir, "catalog.csv")
    data_dict = load_csv_as_dict(catalog_path)

    widths = np.array(data_dict["indenter_width(mm)"], dtype=float)
    lengths = np.array(data_dict["indenter_height(mm)"], dtype=float)
    experiment_reldirs = np.array(data_dict["experiment_reldir"])

    # Split data into train and test and save the split information
    perm = np.random.permutation(len(experiment_reldirs))
    n_train = 4 * len(experiment_reldirs) // 5
    data_path = os.path.join(calib_dir, "train_test_split.json")
    dict_to_save = {
        "train": experiment_reldirs[perm[:n_train]].tolist(),
        "test": experiment_reldirs[perm[n_train:]].tolist(),
    }
    with open(data_path, "w") as f:
        json.dump(dict_to_save, f, indent=4)

    # Read the configuration
    config_path = args.config_path
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        ppmm = config["ppmm"]

    # Process each experiment
    for experiment_reldir, length, widths in zip(
        experiment_reldirs, lengths, widths
    ):
        experiment_dir = os.path.join(calib_dir, experiment_reldir)
        image_path = os.path.join(experiment_dir, "gelsight.png")
        image = cv2.imread(image_path)

        label_path = os.path.join(experiment_dir, "label.npz")
        label_data = np.load(label_path)
        line_center = label_data["line_center"]
        line_angle = label_data["line_angle"]
        line_length = label_data["line_length"]
        rect1 = label_data["rect1"]
        rect2 = label_data["rect2"]
        width1 = label_data["width1"]
        width2 = label_data["width2"]

        # Create the rotated rectangle mask
        mask = create_rotated_rectangle_mask(image.shape[:2], line_center, line_length, width1, width2, line_angle)

        # Compute the gradient map
        dx = np.cos(np.radians(line_angle))
        dy = np.sin(np.radians(line_angle))
        depth_map = np.zeros(image.shape[:2])
        depth_map[mask] = np.sqrt((width1 + width2) / (2 * ppmm))

        gxangles = np.gradient(depth_map, axis=1)  # Gradient in x-direction
        gyangles = np.gradient(depth_map, axis=0)  # Gradient in y-direction
        gxyangles = np.stack([gxangles, gyangles], axis=-1)
        gxyangles[np.logical_not(mask)] = np.array([0.0, 0.0])

        # Convert image to BGR and XYS format
        bgrxys = image2bgrxys(image)
        save_path = os.path.join(experiment_dir, "data.npz")
        np.savez(save_path, bgrxys=bgrxys, gxyangles=gxyangles, mask=mask)

    # Save the background data
    bg_path = os.path.join(calib_dir, "background.png")
    bg_image = cv2.imread(bg_path)
    bgrxys = image2bgrxys(bg_image)
    gxyangles = np.zeros((bg_image.shape[0], bg_image.shape[1], 2))
    mask = np.ones((bg_image.shape[0], bg_image.shape[1]), dtype=np.bool_)
    save_path = os.path.join(calib_dir, "background_data.npz")
    np.savez(save_path, bgrxys=bgrxys, gxyangles=gxyangles, mask=mask)


if __name__ == "__main__":
    prepare_data()
