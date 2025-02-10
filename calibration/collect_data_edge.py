import argparse
import os
import sys

import cv2
import numpy as np
import yaml


from gs_sdk.gs_device import Camera
from calibration.utils import load_csv_as_dict

"""
This script collects tactile data using indenters for sensor calibration.

Instruction: 
    1. Connect the sensor to the computer.
    2. Prepare an indenter with known width and height.
    3. Run this script, press 'b' to collect a background image.
    4. Press the sensor with the indenter at multiple locations (~50 locations preferred),
       press 'w' to save the tactile image. When done, press 'q' to quit.
Note:
    If you have prepared multiple indenters with different dimensions, you can run this script multiple
    times, assign the same calib_dir but different indenter dimensions, and the system will treat it as 
    one single dataset.

Usage:
    python collect_data_edge.py --calib_dir CALIB_DIR --indenter_width WIDTH [--config_path CONFIG_PATH]

Arguments:
    --calib_dir: Path to the directory where the collected data will be saved
    --indenter_width: Width of the indenter in mm
    --config_path: (Optional) Path to the configuration file about the sensor dimensions.
                   If not provided, GelSight Mini is assumed.
"""

config_dir = os.path.join(os.path.dirname(__file__), "../examples/configs")


def collect_data():
    # Argument Parsers
    parser = argparse.ArgumentParser(
        description="Collect calibration data with indenters to calibrate the sensor."
    )
    parser.add_argument(
        "-b",
        "--calib_dir",
        type=str,
        help="path to save calibration data",
    )
    parser.add_argument(
        "--indenter_width", type=float, help="width of the indenter in mm", required=True
    )
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        help="path of the sensor information",
        default=os.path.join(config_dir, "gsmini.yaml"),
    )
    args = parser.parse_args()

    # Create the data saving directories
    calib_dir = args.calib_dir
    indenter_width = args.indenter_width
    # Create a subdirectory name using both dimensions, e.g., "10.000x20.000mm"
    indenter_subdir = "{:.3f}mm".format(indenter_width)
    indenter_dir = os.path.join(calib_dir, indenter_subdir)
    if not os.path.isdir(indenter_dir):
        os.makedirs(indenter_dir)

    # Read the configuration
    config_path = args.config_path
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        device_name = config["device_name"]
        imgh = config["imgh"]
        imgw = config["imgw"]

    # Create the data saving catalog with headers for width and height
    catalog_path = os.path.join(calib_dir, "catalog.csv")
    if not os.path.isfile(catalog_path):
        with open(catalog_path, "w") as f:
            f.write("experiment_reldir,indenter_width(mm),indenter_height(mm)\n")

    # Find last data_count collected with these indenter dimensions
    data_dict = load_csv_as_dict(catalog_path)

    widths = np.array([float(x) for x in data_dict["indenter_width(mm)"]])
    data_idxs = np.where(np.abs(widths - indenter_width) < 1e-3)[0]
    data_counts = np.array(
        [int(os.path.basename(reldir)) for reldir in data_dict["experiment_reldir"]]
    )
    if len(data_idxs) == 0:
        data_count = 0
    else:
        data_count = max(data_counts[data_idxs]) + 1


    # Connect to the device and collect data until quit
    device = Camera(device_name, imgh, imgw)
    device.connect()
    print("Press key to collect data, collect background, or quit (w/b/q)")
    while True:
        image = device.get_image()

        # Display the image and decide record or quit
        cv2.imshow("frame", image)
        key = cv2.waitKey(100)
        if key == ord("w"):
            # Save the image
            experiment_reldir = os.path.join(indenter_subdir, str(data_count))
            experiment_dir = os.path.join(calib_dir, experiment_reldir)
            if not os.path.isdir(experiment_dir):
                os.makedirs(experiment_dir)
            save_path = os.path.join(experiment_dir, "gelsight.png")
            cv2.imwrite(save_path, image)
            print("Saved data to new path: %s" % save_path)

            # Save to catalog with both dimensions
            with open(catalog_path, "a") as f:
                f.write("{},{:.3f}\n".format(experiment_reldir, indenter_width))
            data_count += 1
        elif key == ord("b"):
            print("Collecting 10 background images, please wait ...")
            images = []
            for _ in range(10):
                image = device.get_image()
                images.append(image)
                cv2.imshow("frame", image)
                cv2.waitKey(1)
            image = np.mean(images, axis=0).astype(np.uint8)
            # Save the background image
            save_path = os.path.join(calib_dir, "background.png")
            cv2.imwrite(save_path, image)
            print("Saved background image to %s" % save_path)
        elif key == ord("q"):
            # Quit
            break
        elif key == -1:
            # No key pressed
            continue
        else:
            print("Unrecognized key.\nPlease use 'w' to save the image, 'b' to save the background or 'q' to quit.")

    device.release()
    cv2.destroyAllWindows()
    print("%d images collected in total." % data_count)


if __name__ == "__main__":
    collect_data()
