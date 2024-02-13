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

"""
Config element:

camera_l = (
    "camera_l",                              ## hdf5 dataset name 
    (_chunk, 1024, 1280, 3),                 ## Chunk shape - Must match shape of data coming from Rostopic 
    (None, 1024, 1280, 3),                   ## Max shape - similar to chunk shape 
    "gzip",                                  ## compression for hdf5 dataset 
    np.uint8,                                ## datatype for hdf5 dataset 
    "/jhu_daVinci/decklink/left/image_raw",  ## rostopic where data is published 
    Image,                                   ## message type 
    processing callback                      ## processing callback that transforms message to numpy array
)
"""
# Original dvrk resolution: 1024, 1280
_chunk = 100
resize_image_processor = get_image_processor_with_resize((640, 480))
image_processor = get_image_processor()


class HandEyeRostopicsConfig(Enum):
    """
    Topics to record in sync. Each enum value is a tuple with the following elements:
    (<topic_name>, <message_type>, <attribute_name>)

    attribute_name: corresponds to the attribute name in the DatasetSample class
    """

    CAMERA_L_IMAGE = ("/jhu_daVinci/decklink/left/image_raw", Image, image_processor)
    CAMERA_R_IMAGE = ("/jhu_daVinci/decklink/right/image_raw", Image, image_processor)
    MEASURED_CP = ("/PSM2/measured_cp", PoseStamped, processing_pose_data)
    MEASURED_JP = ("/PSM2/measured_js", JointState, processing_joint_state_data)


class HandEyeHdf5Config(Enum):
    camera_l_resized = (
        "camera_l",
        (_chunk, 480, 640, 3),
        (None, 480, 640, 3),
        "gzip",
        np.uint8,
        HandEyeRostopicsConfig.CAMERA_L_IMAGE.value[0],
        HandEyeRostopicsConfig.CAMERA_L_IMAGE.value[1],
        resize_image_processor,
    )
    camera_l = (
        "camera_l",
        (_chunk, 1024, 1280, 3),
        (None, 1024, 1280, 3),
        "gzip",
        np.uint8,
        HandEyeRostopicsConfig.CAMERA_L_IMAGE.value[0],
        HandEyeRostopicsConfig.CAMERA_L_IMAGE.value[1],
        image_processor,
    )
    camera_r = (
        "camera_r",
        (_chunk, 480, 640, 3),
        (None, 480, 640, 3),
        "gzip",
        np.uint8,
        HandEyeRostopicsConfig.CAMERA_R_IMAGE.value[0],
        HandEyeRostopicsConfig.CAMERA_R_IMAGE.value[1],
        HandEyeRostopicsConfig.CAMERA_R_IMAGE.value[2],
    )
    psm1_measured_cp = (
        "psm1_measured_cp",
        (_chunk, 4, 4),
        (None, 4, 4),
        "gzip",
        np.float64,
        HandEyeRostopicsConfig.MEASURED_CP.value[0],
        HandEyeRostopicsConfig.MEASURED_CP.value[1],
        HandEyeRostopicsConfig.MEASURED_CP.value[2],
    )
    psm1_measured_jp = (
        "psm1_measured_jp",
        (_chunk, 6),
        (None, 6),
        "gzip",
        np.float64,
        HandEyeRostopicsConfig.MEASURED_JP.value[0],
        HandEyeRostopicsConfig.MEASURED_JP.value[1],
        HandEyeRostopicsConfig.MEASURED_JP.value[2],
    )


data_to_record = [
    HandEyeRostopicsConfig.CAMERA_L_IMAGE,
    # HandEyeRostopicsConfig.CAMERA_R_IMAGE,
    HandEyeRostopicsConfig.MEASURED_CP,
    HandEyeRostopicsConfig.MEASURED_JP,
]


if __name__ == "__main__":
    from hdf5_saver.Hdf5Writer import Hdf5EntryConfig, Hdf5FullDatasetConfig

    selected_configs = [HandEyeHdf5Config.camera_l, HandEyeHdf5Config.camera_r]
    dataset_config = Hdf5FullDatasetConfig.create_from_enum_list(selected_configs)

    for config in dataset_config:
        print(config)
