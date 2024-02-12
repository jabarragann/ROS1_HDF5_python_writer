from enum import Enum

import numpy as np

# fmt: off
_chunk = 100 
class HandEyeDVRKConfig(Enum):
    camera_l = ("camera_l", (_chunk, 480, 640, 3), (None, 480, 640, 3), "gzip", np.uint8)
    camera_r = ("camera_r", (_chunk, 480, 640, 3), (None, 480, 640, 3), "gzip", np.uint8)
    psm1_measured_cp = ("psm1_measured_cp", (_chunk, 4,4), (None, 4,4), "gzip", np.float64)
    psm1_measured_jp = ("psm1_measured_jp", (_chunk, 6), (None, 6), "gzip", np.float64)

# fmt: on


if __name__ == "__main__":
    from hdf5_saver.Hdf5Writer import Hdf5EntryConfig, Hdf5FullDatasetConfig

    selected_configs = [HandEyeDVRKConfig.camera_l, HandEyeDVRKConfig.camera_r]
    dataset_config = Hdf5FullDatasetConfig.create_from_enum_list(selected_configs)

    for config in dataset_config:
        print(config)
