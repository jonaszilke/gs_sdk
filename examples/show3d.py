import argparse
from argparse import Namespace

import cv2
import os

import yaml

from gs_sdk import gs_device
from gs_sdk import gs_reconstruct

config_dir = os.path.join(os.path.dirname(__file__), "./configs")
model_dir = os.path.join(os.path.dirname(__file__), "./models")


def main():
    # Set flags
    SAVE_VIDEO_FLAG = False
    FIND_ROI = False
    GPU = False

    # Path to 3d model
    args: Namespace = parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
        # Get the camera resolution
        ppmm = config["ppmm"]
        imgw = config["imgw"]
        imgh = config["imgh"]

    # the device ID can change after unplugging and changing the usb ports.
    # on linux run, v4l2-ctl --list-devices, in the terminal to get the device ID for camera
    dev = gs_device.Camera("GelSight Mini", imgh, imgw)
    dev.connect()

    if GPU:
        gpuorcpu = "cuda"
    else:
        gpuorcpu = "cpu"

    recon = gs_reconstruct.Reconstructor(args.model_path, device=gpuorcpu)
    bg_image = cv2.imread(args.background_path)
    recon.load_bg(bg_image)

    f0 = dev.get_image()
    roi = (0, 0, f0.shape[1], f0.shape[0])

    if SAVE_VIDEO_FLAG:
        #### Below VideoWriter object will create a frame of above defined The output is stored in 'filename.avi' file.
        file_path = './3dnnlive.mov'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(file_path, fourcc, 60, (f0.shape[1], f0.shape[0]), isColor=True)
        print(f'Saving video to {file_path}')

    if FIND_ROI:
        roi = cv2.selectROI(f0)
        roi_cropped = f0[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
        cv2.imshow('ROI', roi_cropped)
        print('Press q in ROI image to continue')
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print('roi = ', roi)
    print('press q on image to exit')

    ''' use this to plot just the 3d '''
    vis3d = gs_reconstruct.Visualize3D(dev.imgh, dev.imgw, '', ppmm)

    try:
        while True:

            # get the roi image
            f1 = dev.get_image()
            bigframe = cv2.resize(f1, (f1.shape[1] * 2, f1.shape[0] * 2))
            cv2.imshow('Image', bigframe)

            # compute the depth map
            G, H, C = recon.get_surface_info(f1, ppmm)

            ''' Display the results '''
            vis3d.update(H)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if SAVE_VIDEO_FLAG:
                out.write(f1)

    except KeyboardInterrupt:
        print('Interrupted!')
        dev.release()
        cv2.destroyAllWindows()


def parse_args() -> Namespace:
    # Argument Parser
    parser = argparse.ArgumentParser(description="Show 3D-Reconstruction")
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cpu",
        help="The device to load and run the neural network model.",
    )
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        help="path of the sensor information",
        default=os.path.join(config_dir, "gsmini.yaml"),
    )
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        help="path of the model",
        default=os.path.join(model_dir, "gsmini_old.pth"),
    )
    parser.add_argument(
        "-bg",
        "--background_path",
        type=str,
        help="path of the background image",
        default=os.path.join(model_dir, "background.png"),
    )
    args: Namespace = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
