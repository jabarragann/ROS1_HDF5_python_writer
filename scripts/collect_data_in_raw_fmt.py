from pathlib import Path
import pandas as pd
import time
import cv2
import numpy as np
import rospy
from hdf5_saver.Hdf5Writer import (
    DataContainer,
    Hdf5FullDatasetConfig,
    HDF5Writer,
)
from hdf5_saver.SyncRosClient import DatasetSample, SyncRosClient
from hdf5_saver.custom_configs.hand_eye_dvrk_config import (
    HandEyeHdf5Config,
)
from queue import Empty, Queue
from threading import Thread, Lock
import click
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped

QUEUE_MAX_SIZE = 1000
dataset_config = Hdf5FullDatasetConfig.create_from_enum_list(
    [
        HandEyeHdf5Config.camera_l,
        # HandEyeHdf5Config.camera_r,
        HandEyeHdf5Config.psm1_measured_cp,
        HandEyeHdf5Config.psm1_measured_jp,
    ]
)


class TimerCb(Thread):

    def __init__(self, dataset_root: Path, data_queue: Queue, hdf5_writer: HDF5Writer):
        super(TimerCb, self).__init__()
        self.dataset_root = dataset_root
        self.img_path = self.dataset_root / "imgs"
        self.img_path.mkdir(parents=True, exist_ok=True)
        self.data_queue: Queue[DatasetSample] = data_queue
        self.hdf5_writer = hdf5_writer
        self.data_container = DataContainer(self.hdf5_writer.dataset_config)

        self.terminate_recording = False
        self.saved_points = 0

        self.joint_data = []
        self.pose_data = []
        # fmt: off
        self.joint_headers = ["j1","j2","j3","j4","j5","j6"]
        self.pose_headers = ["p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15","p16"] 
        # fmt: on

    def run(self):
        log_time = time.time()

        try:
            self.record_data_loop()
        finally:
            self.joint_data = np.vstack(self.joint_data)
            self.pose_data = np.vstack(self.pose_data)
            print(self.joint_data.shape)
            print(self.pose_data.shape)
            pd.DataFrame(self.joint_data, columns=self.joint_headers).to_csv(
                self.dataset_root / "joint_data.csv"
            )
            pd.DataFrame(self.pose_data, columns=self.pose_headers).to_csv(
                self.dataset_root / "pose_data.csv"
            )

        print(f"Collected {self.saved_points} samples")

    def record_data_loop(self):
        while not self.terminate_recording:
            data = None
            try:
                data = self.data_queue.get_nowait()
            except Empty:
                pass

            if data is not None:
                print(data.keys())
                for config in self.hdf5_writer.dataset_config:
                    if config.dataset_name == "psm1_measured_jp":
                        print("jp")
                        self.joint_data.append(data[config.dataset_name].reshape(1, -1))
                    elif config.dataset_name == "psm1_measured_cp":
                        print("cp")
                        self.pose_data.append(
                            data[config.dataset_name].flatten().reshape(1, -1)
                        )
                    elif config.dataset_name == "camera_l":
                        print("img")
                        img_pth = str(
                            self.img_path / f"camera_l_{self.saved_points:05d}.jpeg"
                        )
                        cv2.imwrite(img_pth, data[config.dataset_name])
                    else:
                        raise ValueError(f"Unknown dataset name: {config.dataset_name}")

                self.saved_points += 1
            time.sleep(0.005)

    def write_data_and_empty_container(self):
        self.hdf5_writer.write_chunk(self.data_container)
        self.data_container = DataContainer(self.hdf5_writer.dataset_config)

    def convert_raw_sample_to_dict(self, raw_sample: DatasetSample):
        data_dict = {}
        for config in self.hdf5_writer.dataset_config:
            data_dict[config.dataset_name] = raw_sample._internal_data_dict[
                config.dataset_name
            ]

        return data_dict

    def finish_recording(self):
        self.terminate_recording = True


@click.command()
@click.option("--output_dir", type=click.Path(file_okay=False), default="./temp")
def main(output_dir: Path):

    ts = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir) / (ts + "_raw_dataset")
    data_queue = Queue(maxsize=QUEUE_MAX_SIZE)
    ros_client = SyncRosClient(
        data_queue=data_queue, dataset_config=dataset_config, collection_freq=5
    )
    hdf5_writer = HDF5Writer(output_dir, dataset_config)
    timer_cb = TimerCb(output_dir, data_queue, hdf5_writer)

    ros_client.wait_for_data()
    timer_cb.start()

    rospy.spin()
    timer_cb.finish_recording()
    print("\nmain thread finished")


if __name__ == "__main__":
    main()
