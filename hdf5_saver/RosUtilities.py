from __future__ import annotations
from typing import Any, Callable, Dict, Tuple
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped
from enum import Enum
import tf_conversions.posemath as pm
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import PyKDL
import numpy as np
import cv2


##############################
# ROS utility functions
##############################


def convert_units(frame: PyKDL.Frame):
    scaled_frame = PyKDL.Frame(frame.M, frame.p / 1.0)
    return scaled_frame


def processing_pose_data(msg: PoseStamped) -> np.ndarray:
    return pm.toMatrix(convert_units(pm.fromMsg(msg.pose)))


def processing_joint_state_data(msg: JointState) -> np.ndarray:
    return np.array(msg.position)


def get_image_processor() -> Callable[[Image], np.ndarray]:
    bridge = CvBridge()

    def process_img(msg: Image) -> np.ndarray:
        return bridge.imgmsg_to_cv2(msg, "bgr8")

    return process_img


def get_image_processor_with_resize(
    new_resolution: Tuple = (640, 480)
) -> Callable[[Image], np.ndarray]:
    bridge = CvBridge()

    def process_img(msg: Image) -> np.ndarray:
        raw = bridge.imgmsg_to_cv2(msg, "bgr8")
        resized_img = cv2.resize(raw, (640, 480))
        return resized_img

    return process_img
