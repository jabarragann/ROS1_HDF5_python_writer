from __future__ import annotations
from typing import Any, Callable, Dict
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped
from enum import Enum
import tf_conversions.posemath as pm
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import PyKDL
import numpy as np
from hdf5_saver.custom_configs.hand_eye_dvrk_config import (
    HandEyeDVRKConfig,
)


##############################
# ROS client configuration
##############################

# fmt: off
class RosTopics(Enum):
    """
    Topics to record in sync. Each enum value is a tuple with the following elements:
    (<topic_name>, <message_type>, <attribute_name>)

    attribute_name: corresponds to the attribute name in the DatasetSample class
    """

    CAMERA_L_IMAGE = ( "/ambf/env/cameras/cameraL/ImageData", Image, "left_rgb_img")
    CAMERA_R_IMAGE = ( "/ambf/env/cameras/cameraL2/ImageData", Image, "right_rgb_img")
    MEASURED_CP = ("/CRTK/psm1/measured_cp", PoseStamped, "measured_cp")
    MEASURED_JP = ("/CRTK/psm1/measured_js", JointState, "measured_jp")
# fmt: on

# Association between rostopics and the corresponding key in DataContainer
topic_to_key_in_container = {
    RosTopics.CAMERA_L_IMAGE: HandEyeDVRKConfig.camera_l.value[0],
    RosTopics.CAMERA_R_IMAGE: HandEyeDVRKConfig.camera_r.value[0],
    RosTopics.MEASURED_CP: HandEyeDVRKConfig.psm1_measured_cp.value[0],
    RosTopics.MEASURED_JP: HandEyeDVRKConfig.psm1_measured_jp.value[0],
}

selected_topics = [
    RosTopics.CAMERA_L_IMAGE,
    RosTopics.CAMERA_R_IMAGE,
    RosTopics.MEASURED_CP,
    RosTopics.MEASURED_JP,
]


def get_topics_processing_cb() -> Dict[RosTopics, Callable[[Any]]]:
    image_processor = get_image_processor()

    TopicsProcessingCb = {
        RosTopics.CAMERA_L_IMAGE: image_processor,
        RosTopics.CAMERA_R_IMAGE: image_processor,
        RosTopics.MEASURED_CP: processing_pose_data,
        RosTopics.MEASURED_JP: processing_joint_state_data,
    }

    return TopicsProcessingCb


##############################
# Utility functions
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
