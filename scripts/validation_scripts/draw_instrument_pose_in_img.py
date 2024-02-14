from pathlib import Path
import cv2
import numpy as np
import pandas as pd

# fmt: off
mtx =   [1767.7722 ,    0.     ,  529.11477,
            0.     , 1774.33579,  510.58841,
            0.     ,    0.     ,    1.       ]
dist = [-0.337317, 0.500592, 0.001082, 0.002775, 0.000000]

Y = [[-0.722624032680848,    0.5730602138325281, -0.3865443036888354,  -0.06336454384831497], 
     [ 0.6870235345618597,   0.533741788848803,  -0.49307034568569336, -0.15304205999332426], 
     [-0.07624414961292797, -0.621869515379792,  -0.7794004974922103,   0.0664797333995826], 
     [0.0, 0.0, 0.0, 1.0]]

mtx = np.array(mtx).reshape(3, 3)
dist = np.array(dist)
Y = np.array(Y)
Y_inv = np.linalg.inv(Y)

# mtx = np.array( [ [1747.03274843401, 0.0, 521.0492603841079], [0.0, 1747.8694061586648, 494.32395180615987], [0.0, 0.0, 1.0], ])
# dist = np.array( [ -0.33847195608374453, 0.16968704500434714, 0.0007293228134352138, 0.005422675750927001, 0.9537762252401928, ])
# marker_size = 0.01

# Y = np.array([[ 0.15755188, 0.51678588, 0.84149258, 0.26527036],
#               [-0.89806276,-0.27940152, 0.33973234,-0.02360361],
#               [ 0.41068319,-0.80923862, 0.42008592,-0.08475506],
#               [ 0.        , 0.        , 0.        , 1.        ]])
# Y_inv = np.linalg.inv(Y)
# fmt: on


def load_images_data(root_path: Path, idx: int):
    img_path = root_path / "imgs" / "left" / f"camera_l_{idx:05d}.jpeg"

    left_img = cv2.imread(str(img_path))
    return left_img


def load_poses_data(root_path: Path):
    file_path = root_path / "pose_data.csv"
    pose_data = pd.read_csv(file_path).values
    pose_data = pose_data[:, 1:]  # remove index column
    pose_data = pose_data.reshape(-1, 4, 4)

    return pose_data


def main():

    root_path = Path("temp/20240213_212744_raw_dataset_handeye_rect_img_local")
    idx = 30
    img = load_images_data(root_path, idx)

    pose_data = load_poses_data(root_path)

    pose = Y_inv @ pose_data[idx]
    tvec = pose[:3, 3]
    rvec = cv2.Rodrigues(pose[:3, :3])[0]

    points_3d = np.array([[[0, 0, 0]]], np.float32)
    points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, mtx, dist)

    print(points_2d)
    print(pose_data[idx])

    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
