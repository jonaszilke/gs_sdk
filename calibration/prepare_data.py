import argparse
import json
import os

import cv2
import numpy as np
import yaml

from calibration.utils import load_csv_as_dict
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


def compute_corners(center, length, width1, width2, angle):
    """
    Computes the corners of the rectangle.

    Args:
        center (tuple): (x, y) center of the dividing line.
        length (float): Length of the dividing line.
        width1 (float): Width on one side of the middle line (distance from the middle line to the rectangle).
        width2 (float): Width on the other side of the middle line (distance from the middle line to the rectangle).
        angle (float): Angle of the middle line in degrees.

    The dividing line is a line inside the rectangle parallel to two sides.
    The middle line is defined by the center, the angle, and the length.
    The middle line defines one dimension and the rotation of the rectangle.
    The second dimension is defined by width1 and width2, both describing the distance from the middle line to the rectangle.

    Returns:
        np.ndarray: Corners of the rectangle
    """

    theta = np.radians(angle)
    p0,p1 = get_line_endpoints(center, length, angle)

    # Compute perpendicular unit vector
    perp = np.array([np.sin(theta), -np.cos(theta)])

    # calculate corners
    rectangle1_0 = p0 + (perp * width1)
    rectangle1_1 = p1 + (perp * width1)
    rectangle2_0 = p0 + (-perp * width2)
    rectangle2_1 = p1 + (-perp * width2)

    # Define the four corners of the rotated rectangle.
    corners = np.array([
        rectangle1_0, rectangle1_1, rectangle2_1, rectangle2_0
    ])

    return corners

def get_line_endpoints(center, length, angle):
    theta = np.radians(angle)
    dx = (length / 2) * np.cos(theta)
    dy = (length / 2) * np.sin(theta)
    p0 = np.array([center[0] - dx, center[1] - dy])
    p1 = np.array([center[0] + dx, center[1] + dy])

    return p0, p1

def inside_rectangle(p, corners):
    """
    Returns True if point p is inside the rectangle defined by corners.
    The method uses the cross-product test.
    """
    num_corners = len(corners)
    prev_sign = 0
    for i in range(num_corners):
        a = corners[i]
        b = corners[(i + 1) % num_corners]
        edge = b - a
        vec = p - a
        cross = edge[0] * vec[1] - edge[1] * vec[0]
        if cross != 0:
            current_sign = np.sign(cross)
            if prev_sign == 0:
                prev_sign = current_sign
            elif current_sign != prev_sign:
                return False
    return True

def create_rotated_rectangle_mask(image_shape, corners):
    """
    Creates a binary mask for a rotated rectangle in an image.

    Args:
        image_shape (tuple): Shape of the image (height, width).
        corners (np.array): corners of the rectangle

    Returns:
        np.ndarray: Boolean mask with the same shape as the image.
    """
    height, width = image_shape

    # Generate grid of pixel coordinates.
    # Note: X corresponds to column indices and Y to row indices.
    Y, X = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    points = np.stack((X, Y), axis=-1).reshape(-1, 2)


    # Create the mask by testing each pixel.
    mask = np.array([inside_rectangle(p, corners) for p in points]).reshape(height, width)

    return mask

def get_depth_map(image_shape, middle_line, line_1, line_2, depth):
    height, width = image_shape
    # Generate grid of pixel coordinates.
    Y, X = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    points = np.stack((X, Y), axis=-1).reshape(-1, 2)

    # Vectorized depth calculation
    rect_1 = np.array([middle_line[0], middle_line[1], line_1[1], line_1[0]])
    rect_2 = np.array([middle_line[0], middle_line[1], line_2[0], line_2[1]])

    # Vectorized inside rectangle checks
    mask_1 = np.array([inside_rectangle(p, rect_1) for p in points])
    mask_2 = np.array([inside_rectangle(p, rect_2) for p in points])

    value_map = np.where(mask_1, linear_interpolation_two_lines(line_1, middle_line, depth, points), 0)
    value_map = np.where(mask_2, linear_interpolation_two_lines(line_2, middle_line, depth, points), value_map)

    return value_map.reshape(height, width)

def get_value(p, edge, line_1, line_2, depth):
    result = 0.0

    rect_1 = np.array([edge[0], edge[1], line_1[1], line_1[0]])
    rect_2 = np.array([edge[0], edge[1], line_2[0], line_2[1]])
    if inside_rectangle(p, rect_1):
        result = linear_interpolation_two_lines(line_1, edge, depth, p)
    elif inside_rectangle(p, rect_2):
        result = linear_interpolation_two_lines(line_2, edge, depth, p)
    return result


def linear_interpolation_two_lines(p1, p2, d, query_points):
    """
    Compute the interpolated values at the query points given two parallel lines with values 0 and d.
    """
    direction = p1[1] - p1[0]
    direction = direction / np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]])
    normal = normal / np.linalg.norm(normal)

    d1 = np.dot(p1[0], normal)
    d2 = np.dot(p2[0], normal)
    dq = np.dot(query_points, normal)

    t = (dq - d1) / (d2 - d1)
    t = np.clip(t, 0, 1)

    return d * t



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

    lengths = np.array(data_dict["indenter_width(mm)"], dtype=float)
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
    for experiment_reldir, length in zip(
            experiment_reldirs, lengths
    ):
        experiment_dir = os.path.join(calib_dir, experiment_reldir)
        image_path = os.path.join(experiment_dir, "gelsight.png")
        image = cv2.imread(image_path)

        label_path = os.path.join(experiment_dir, "label.npz")
        label_data = np.load(label_path)
        line_center = label_data["line_center"]
        line_angle = label_data["line_angle"]
        line_length = label_data["line_length"]
        width1 = label_data["width1"]
        width2 = label_data["width2"]
        depth = np.sqrt(width1 * width2) / ppmm

        # Create the rotated rectangle mask
        corners = compute_corners(line_center, line_length, width1, width2, line_angle)
        edge = get_line_endpoints(line_center, line_length, line_angle)
        line_1 = corners[:2]
        line_2 = corners[2:]
        mask = create_rotated_rectangle_mask(image.shape[:2], corners)
        depth_map = get_depth_map(image.shape[:2], edge, line_1, line_2, depth)

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
