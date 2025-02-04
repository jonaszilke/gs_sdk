import gc
import os
import argparse

import cv2
import numpy as np
import nanogui as ng
from nanogui import Texture
from nanogui import glfw
import yaml

from utils import load_csv_as_dict

"""
Modified Calibration Script:
This version calibrates using a sharp edge defined by a line.
Stage 0: Adjust a line (position, length, and rotation) to match the edge.
Stage 1: Adjust the width for rectangle 1 (using the line as one side).
Stage 2: Adjust the width for rectangle 2 (on the opposite side).
When finished, the tool saves the following information in an NPZ file:
    - The line parameters: center, length, and angle.
    - The two widths (one for each rectangle).
    - The computed corner coordinates for both rectangles.
Usage and key instructions are printed in the console.
"""

config_dir = os.path.join(os.path.dirname(__file__), "../examples/configs")


class Line:
    """A line representing an edge.
    The line is defined by a center point, a length, and an angle (in degrees).
    Additional parameters control how much the line moves, grows, or rotates per key press.
    """
    color_line = (128, 0, 0)
    opacity = 0.5

    def __init__(self, cx, cy, length=100, angle=0.0, move_incr=2, length_incr=2, angle_incr=2):
        self.center = [cx, cy]
        self.length = length
        self.angle = angle  # in degrees
        self.move_incr = move_incr
        self.length_incr = length_incr
        self.angle_incr = angle_incr


class CalibrateApp(ng.Screen):
    fnames = list()
    read_all = False  # flag to indicate if all images have been read
    load_img = True
    change = False

    def __init__(
        self, calib_data, imgw, imgh, display_difference=False, detect_circle=False
    ):
        super(CalibrateApp, self).__init__((1024, 768), "Edge Calibration App")
        self.imgw = imgw
        self.imgh = imgh
        self.display_difference = display_difference
        self.detect_circle = detect_circle
        # Load background image (if needed)
        self.bg_img = cv2.imread(os.path.join(calib_data, "background.png"))
        # Initialize calibration state:
        # Stage 0: Calibrate the line.
        # Stage 1: Calibrate width for rectangle on one side.
        # Stage 2: Calibrate width for rectangle on the other side.
        self.stage = 0
        self.line = Line(self.imgw / 2, self.imgh / 2, length=100, angle=0.0)
        self.width1 = 20  # default extrusion (width) for rectangle 1
        self.width2 = 20  # default extrusion (width) for rectangle 2

        window = ng.Window(self, "IO Window")
        window.set_position((15, 15))
        window.set_layout(ng.GroupLayout())

        ng.Label(window, "Folder dialog", "sans-bold")
        tools = ng.Widget(window)
        tools.set_layout(
            ng.BoxLayout(ng.Orientation.Horizontal, ng.Alignment.Middle, 0, 6)
        )

        # Initialize the file directory and list of filenames
        b = ng.Button(tools, "Open")

        def open_cb():
            self.parent_dir = calib_data
            # Read the catalog and create a list of all filenames
            catalog_dict = load_csv_as_dict(
                os.path.join(self.parent_dir, "catalog.csv")
            )
            self.fnames = [
                os.path.join(self.parent_dir, fname)
                for fname in catalog_dict["experiment_reldir"]
            ]
            print(
                f"Selected directory = {self.parent_dir}, total {len(self.fnames)} images"
            )
            self.img_idx = 0

        b.set_callback(open_cb)

        # Initialize the image window
        self.img_window = ng.Window(self, "Current image")
        self.img_window.set_position((200, 15))
        self.img_window.set_layout(ng.GroupLayout())

        # Initialize the calibrate button. Its callback will serve to advance through the stages.
        b = ng.Button(self.img_window, "Calibrate")

        def calibrate_cb():
            if self.stage == 0:
                self.stage = 1
                print("Stage 0 complete (line calibrated). Now adjust width for rectangle 1 (use M/P keys).")
                self.change = True
            elif self.stage == 1:
                self.stage = 2
                print("Rectangle 1 width set. Now adjust width for rectangle 2 (use M/P keys).")
                self.change = True
            elif self.stage == 2:
                # Final stage: compute rectangle coordinates and save calibration.
                frame = self.orig_img
                # Compute the line endpoints:
                half = self.line.length / 2.0
                angle_rad = np.deg2rad(self.line.angle)
                dx = half * np.cos(angle_rad)
                dy = half * np.sin(angle_rad)
                p1 = (int(self.line.center[0] - dx), int(self.line.center[1] - dy))
                p2 = (int(self.line.center[0] + dx), int(self.line.center[1] + dy))
                # Compute perpendicular vector:
                perp = (-np.sin(angle_rad), np.cos(angle_rad))
                # Define rectangle 1 (on one side of the line):
                rect1 = np.array([
                    p1,
                    p2,
                    (p2[0] + int(self.width1 * perp[0]), p2[1] + int(self.width1 * perp[1])),
                    (p1[0] + int(self.width1 * perp[0]), p1[1] + int(self.width1 * perp[1]))
                ], dtype=np.int32)
                # Define rectangle 2 (on the opposite side of the line):
                rect2 = np.array([
                    p1,
                    p2,
                    (p2[0] - int(self.width2 * perp[0]), p2[1] - int(self.width2 * perp[1])),
                    (p1[0] - int(self.width2 * perp[0]), p1[1] - int(self.width2 * perp[1]))
                ], dtype=np.int32)
                print(f"Frame {self.img_idx}: line center = {self.line.center}, length = {self.line.length}, angle = {self.line.angle}, width1 = {self.width1}, width2 = {self.width2}")
                save_dir = os.path.join(self.fnames[self.img_idx], "label.npz")
                np.savez(save_dir,
                         line_center=self.line.center,
                         line_length=self.line.length,
                         line_angle=self.line.angle,
                         width1=self.width1,
                         width2=self.width2,
                         rect1=rect1,
                         rect2=rect2)
                # Save the labeled image with the overlay:
                labeled_img = self.overlay_calibration(frame)
                labeled_img_path = os.path.join(self.fnames[self.img_idx], "labeled.png")
                cv2.imwrite(labeled_img_path, labeled_img)
                # Prepare for the next image:
                self.load_img = True
                self.update_img_idx()
                self.stage = 0  # reset stage for next image
                # (Optionally, reset line and widths to defaults for the next image.)
                self.line = Line(self.imgw / 2, self.imgh / 2, length=100, angle=0.0)
                self.width1 = 20
                self.width2 = 20

        b.set_callback(calibrate_cb)

        self.img_view = ng.ImageView(self.img_window)
        self.img_tex = ng.Texture(
            pixel_format=Texture.PixelFormat.RGB,
            component_format=Texture.ComponentFormat.UInt8,
            size=[imgw, imgh],
            min_interpolation_mode=Texture.InterpolationMode.Trilinear,
            mag_interpolation_mode=Texture.InterpolationMode.Nearest,
            flags=Texture.TextureFlags.ShaderRead | Texture.TextureFlags.RenderTarget,
        )
        self.perform_layout()

    def update_img_idx(self):
        self.img_idx += 1
        if self.img_idx == len(self.fnames) - 1:
            self.read_all = True

    def overlay_calibration(self, orig_img):
        """Overlay the calibrated line and rectangles on the image."""
        overlay = orig_img.copy()
        half = self.line.length / 2.0
        angle_rad = np.deg2rad(self.line.angle)
        dx = half * np.cos(angle_rad)
        dy = half * np.sin(angle_rad)
        p1 = (int(self.line.center[0] - dx), int(self.line.center[1] - dy))
        p2 = (int(self.line.center[0] + dx), int(self.line.center[1] + dy))
        # Draw the line
        cv2.line(overlay, p1, p2, self.line.color_line, 2)

        # Compute perpendicular vector (as float):
        perp = (-np.sin(angle_rad), np.cos(angle_rad))

        if self.stage >= 1:
            # Compute rectangle 1 (using width1 on one side):
            r1 = np.array([
                p1,
                p2,
                (p2[0] + int(self.width1 * perp[0]), p2[1] + int(self.width1 * perp[1])),
                (p1[0] + int(self.width1 * perp[0]), p1[1] + int(self.width1 * perp[1]))
            ], dtype=np.int32)
            cv2.polylines(overlay, [r1], isClosed=True, color=(0, 255, 0), thickness=2)
        if self.stage >= 2:
            # Compute rectangle 2 (using width2 on the opposite side):
            r2 = np.array([
                p1,
                p2,
                (p2[0] - int(self.width2 * perp[0]), p2[1] - int(self.width2 * perp[1])),
                (p1[0] - int(self.width2 * perp[0]), p1[1] - int(self.width2 * perp[1]))
            ], dtype=np.int32)
            cv2.polylines(overlay, [r2], isClosed=True, color=(0, 0, 255), thickness=2)

        cv2.addWeighted(overlay, self.line.opacity, orig_img, 1 - self.line.opacity, 0, overlay)
        return overlay

    def draw(self, ctx):
        self.img_window.set_size((2000, 2600))
        self.img_view.set_size((self.imgw, self.imgh))

        # Load a new image when flagged:
        if self.load_img and len(self.fnames) > 0 and not self.read_all:
            print("Loading %s" % self.fnames[self.img_idx])
            self.orig_img = cv2.imread(
                os.path.join(self.fnames[self.img_idx], "gelsight.png")
            )
            # (Optional) If you want to do any auto-detection based on the background image, you could add that here.
        if (self.load_img and len(self.fnames) > 0) or self.change:
            self.load_img = False
            self.change = False
            # Either show the raw image or the difference image if the flag is set.
            if self.display_difference:
                diff_img = (self.orig_img.astype(np.float32) - self.bg_img.astype(np.float32)) * 3
                diff_img = np.clip(diff_img, -127, 128) + np.ones_like(diff_img) * 127
                display_img = cv2.cvtColor(diff_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
            else:
                display_img = cv2.cvtColor(self.orig_img, cv2.COLOR_BGR2RGB)
            img = self.overlay_calibration(display_img)
            # Ensure the image has an alpha channel if required by the texture.
            if self.img_tex.channels() > 3:
                height, width = img.shape[:2]
                alpha = 255 * np.ones((height, width, 1), dtype=img.dtype)
                img = np.concatenate((img, alpha), axis=2)
            self.img_tex.upload(img)
            self.img_view.set_image(self.img_tex)
        super(CalibrateApp, self).draw(ctx)

    def keyboard_event(self, key, scancode, action, modifiers):
        if super(CalibrateApp, self).keyboard_event(key, scancode, action, modifiers):
            return True
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            self.set_visible(False)
            return True

        # Behavior depends on the current calibration stage.
        if self.stage == 0:
            # Stage 0: Adjust the line.
            if key == glfw.KEY_LEFT:
                self.line.center[0] -= self.line.move_incr
            elif key == glfw.KEY_RIGHT:
                self.line.center[0] += self.line.move_incr
            elif key == glfw.KEY_UP:
                self.line.center[1] -= self.line.move_incr
            elif key == glfw.KEY_DOWN:
                self.line.center[1] += self.line.move_incr
            elif key == glfw.KEY_M:
                self.line.length = max(self.line.length - self.line.length_incr, 10)  # minimum length
            elif key == glfw.KEY_P:
                self.line.length += self.line.length_incr
            elif key == glfw.KEY_Q:
                self.line.angle -= self.line.angle_incr
            elif key == glfw.KEY_E:
                self.line.angle += self.line.angle_incr
            elif key == glfw.KEY_C:
                self.line.move_incr *= 2
            elif key == glfw.KEY_F:
                self.line.move_incr = max(self.line.move_incr / 2, 1)
            self.change = True
        elif self.stage == 1:
            # Stage 1: Adjust width for rectangle 1.
            if key == glfw.KEY_M:
                self.width1 = max(self.width1 - 1, 1)
            elif key == glfw.KEY_P:
                self.width1 += 1
            self.change = True
        elif self.stage == 2:
            # Stage 2: Adjust width for rectangle 2.
            if key == glfw.KEY_M:
                self.width2 = max(self.width2 - 1, 1)
            elif key == glfw.KEY_P:
                self.width2 += 1
            self.change = True

        return False


def label_data():
    # Argument parser
    parser = argparse.ArgumentParser(
        description="Label the tactile edge data using Nanogui."
    )
    parser.add_argument(
        "-b",
        "--calib_dir",
        type=str,
        help="path to save calibration data",
    )
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        help="path of configuring gelsight",
        default=os.path.join(config_dir, "gsmini.yaml"),
    )
    parser.add_argument(
        "-d",
        "--display_difference",
        action="store_true",
        help="Display the difference between the background image",
    )
    parser.add_argument(
        "-r",
        "--detect_circle",
        action="store_true",
        help="(Unused in this version) Automatically detect the edge in the image.",
    )
    args = parser.parse_args()

    # Read the configuration
    config_path = args.config_path
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        imgh = config["imgh"]
        imgw = config["imgw"]

    # Start the label process
    ng.init()
    app = CalibrateApp(
        args.calib_dir,
        imgw,
        imgh,
        display_difference=args.display_difference,
        detect_circle=args.detect_circle,
    )
    app.draw_all()
    app.set_visible(True)
    ng.mainloop(refresh=1 / 60.0 * 1000)
    del app
    gc.collect()
    ng.shutdown()


if __name__ == "__main__":
    label_data()
