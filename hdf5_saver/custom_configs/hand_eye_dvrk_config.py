from __future__ import annotations
from typing import Any, Callable, Dict
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped
from enum import Enum
from sensor_msgs.msg import Image
import numpy as np
from enum import Enum
import numpy as np
from hdf5_saver.RosUtilities import *

##############################
# Handeye calibration config
##############################

# fmt: off
_chunk = 100 
class HandEyeHdf5Config(Enum):
    camera_l = ("camera_l", (_chunk, 480, 640, 3), (None, 480, 640, 3), "gzip", np.uint8)
    camera_r = ("camera_r", (_chunk, 480, 640, 3), (None, 480, 640, 3), "gzip", np.uint8)
    psm1_measured_cp = ("psm1_measured_cp", (_chunk, 4,4), (None, 4,4), "gzip", np.float64)
    psm1_measured_jp = ("psm1_measured_jp", (_chunk, 6), (None, 6), "gzip", np.float64)

class HandEyeRostopicsConfig(Enum):
    """
    Topics to record in sync. Each enum value is a tuple with the following elements:
    (<topic_name>, <message_type>, <attribute_name>)

    attribute_name: corresponds to the attribute name in the DatasetSample class
    """

    CAMERA_L_IMAGE = ( "/ambf/env/cameras/cameraL/ImageData", Image, "left_rgb_img")
    CAMERA_R_IMAGE = ( "/ambf/env/cameras/cameraL2/ImageData", Image, "right_rgb_img")
    MEASURED_CP = ("/CRTK/psm1/measured_cp", PoseStamped, "measured_cp")
    MEASURED_JP = ("/CRTK/psm1/measured_js", JointState, "measured_jp")

# Association between rostopics and the corresponding key in DataContainer
topic_to_key_in_container = {
    HandEyeRostopicsConfig.CAMERA_L_IMAGE: HandEyeHdf5Config.camera_l.value[0],
    HandEyeRostopicsConfig.CAMERA_R_IMAGE: HandEyeHdf5Config.camera_r.value[0],
    HandEyeRostopicsConfig.MEASURED_CP: HandEyeHdf5Config.psm1_measured_cp.value[0],
    HandEyeRostopicsConfig.MEASURED_JP: HandEyeHdf5Config.psm1_measured_jp.value[0],
}

selected_topics = [
    HandEyeRostopicsConfig.CAMERA_L_IMAGE,
    HandEyeRostopicsConfig.CAMERA_R_IMAGE,
    HandEyeRostopicsConfig.MEASURED_CP,
    HandEyeRostopicsConfig.MEASURED_JP,
]


def get_topics_processing_cb() -> Dict[HandEyeRostopicsConfig, Callable[[Any]]]:
    image_processor = get_image_processor()

    TopicsProcessingCb = {
        HandEyeRostopicsConfig.CAMERA_L_IMAGE: image_processor,
        HandEyeRostopicsConfig.CAMERA_R_IMAGE: image_processor,
        HandEyeRostopicsConfig.MEASURED_CP: processing_pose_data,
        HandEyeRostopicsConfig.MEASURED_JP: processing_joint_state_data,
    }

    return TopicsProcessingCb
# fmt: on


if __name__ == "__main__":
    from hdf5_saver.Hdf5Writer import Hdf5EntryConfig, Hdf5FullDatasetConfig

    selected_configs = [HandEyeHdf5Config.camera_l, HandEyeHdf5Config.camera_r]
    dataset_config = Hdf5FullDatasetConfig.create_from_enum_list(selected_configs)

    for config in dataset_config:
        print(config)
