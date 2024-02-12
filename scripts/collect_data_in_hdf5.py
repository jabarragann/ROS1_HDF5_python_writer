from pathlib import Path
import time

import rospy
from hdf5_saver.Hdf5Writer import (
    DataContainer,
    Hdf5FullDatasetConfig,
    HDF5Writer,
)
from hdf5_saver.SyncRosClient import DatasetSample, SyncRosClient
from hdf5_saver.custom_configs.hand_eye_dvrk_config import (
    HandEyeDVRKConfig,
)
from queue import Empty, Queue
from threading import Thread, Lock
import click

QUEUE_MAX_SIZE = 1000
dataset_config = Hdf5FullDatasetConfig.create_from_enum_list(
    [
        HandEyeDVRKConfig.camera_l,
        HandEyeDVRKConfig.camera_r,
        HandEyeDVRKConfig.psm1_measured_cp,
        HandEyeDVRKConfig.psm1_measured_jp,
    ]
)


class TimerCb(Thread):

    def __init__(self, data_queue: Queue, hdf5_writer: HDF5Writer):
        super(TimerCb, self).__init__()
        self.data_queue: Queue[DatasetSample] = data_queue
        self.hdf5_writer = hdf5_writer
        self.data_container = DataContainer(self.hdf5_writer.dataset_config)

        self.terminate_recording = False

    def run(self):
        log_time = time.time()
        total_data = 0

        with self.hdf5_writer:
            while not self.terminate_recording:
                try:
                    total_data += 1
                    data = self.data_queue.get_nowait()
                except Empty:
                    pass

                if data is not None:
                    if self.data_container.is_full():
                        self.write_data_and_empty_container()

                    self.data_container.add_data(data.to_dict())

                time.sleep(0.005)

                if (time.time() - log_time) > 4:
                    log_time = time.time()
                    # print(f"len of container {len(self.data_container)}")
                    print(f"queue size {self.data_queue.qsize()}")

            if len(self.data_container) > 0:  # write any remaining data
                self.hdf5_writer.write_chunk(self.data_container)

        print(f"Collected {total_data} samples")
        print("timer finished")

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
@click.option(
    "--output_dir", type=click.Path(file_okay=False, path_type=Path), default="temp"
)
def main(output_dir):

    data_queue = Queue()
    ros_client = SyncRosClient(data_queue=data_queue)
    hdf5_writer = HDF5Writer(output_dir, dataset_config)
    timer_cb = TimerCb(data_queue, hdf5_writer)

    ros_client.wait_for_data()
    timer_cb.start()

    rospy.spin()
    timer_cb.finish_recording()
    print("\nmain thread finished")


if __name__ == "__main__":
    main()
