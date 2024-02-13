from __future__ import annotations
from queue import Queue
import time
from typing import Dict
import numpy as np
import rospy
from dataclasses import dataclass, field
from abc import ABC
import message_filters
from hdf5_saver.Hdf5Writer import Hdf5EntryConfig, Hdf5FullDatasetConfig
from queue import Empty, Queue


@dataclass
class DatasetSample(dict):
    dataset_config: Hdf5FullDatasetConfig
    data_dict: Dict[str, np.ndarray] = field(default=dict, init=False)

    def copy_from_dict(self, data: dict[str, np.ndarray]):
        self.data_dict = {}
        try:
            for config in self.dataset_config:
                self.add_data(config.dataset_name, data[config.dataset_name])
        except KeyError as e:
            raise KeyError(f"Data for {config.dataset_name} is missing")

    def is_complete(self) -> bool:
        bool_checks = []
        for config in self.dataset_config:
            bool_checks.append(config.dataset_name in self.data_dict)
        return all(bool_checks)

    def to_dict(self) -> Dict[str, np.ndarray]:
        return self.data_dict


@dataclass
class AbstractSimulationClient(ABC):
    """
    Abstract ros client for collecting data from the simulation.

    * Derived classes from this abstract class will need default values for the
    attributes in python version less than 3.10.
    * Derived classes need to call super().__post_init__() in its __post_init__()

    https://medium.com/@aniscampos/python-dataclass-inheritance-finally-686eaf60fbb5
    """

    raw_data: DatasetSample = field(default=None, init=False)
    client_name: str = "ambf_collection_client"

    def __post_init__(self):
        if "/unnamed" == rospy.get_name():
            rospy.init_node(self.client_name)
            time.sleep(0.2)
        else:
            self._client_name = rospy.get_name()

    def get_data(self) -> DatasetSample:
        if self.raw_data is None:
            raise ValueError("No data has been received")

        data = self.raw_data
        self.raw_data = None
        return data

    def has_data(self) -> bool:
        return self.raw_data is not None

    def wait_for_data(self, timeout=10) -> None:
        init_time = last_time = time.time()
        while not self.has_data() and not rospy.is_shutdown():
            time.sleep(0.1)
            last_time = time.time()
            if last_time - init_time > timeout:
                raise TimeoutError(
                    f"Timeout waiting for data. No data received for {timeout}s"
                )


@dataclass
class SyncRosClient(AbstractSimulationClient):
    """
    SyncRosClient requires keyword arguments in the constructor.
    """

    data_queue: Queue = None
    dataset_config: Hdf5FullDatasetConfig = None
    collection_freq: float = None
    print_cb_freq: bool = False

    def __post_init__(self):
        self.cb_timer = time.time()
        self.collection_timer = time.time()

        assert (
            self.data_queue is not None
        ), "data_queue should be provided in constructor as a keyword arg"
        assert (
            self.dataset_config is not None
        ), "dataset_config should be provided in constructor as a keyword arg"
        assert self.collection_freq is not None, "collection_freq should be provided"

        super().__post_init__()
        self.subscribers = []

        for config in self.dataset_config:
            self.subscribers.append(
                message_filters.Subscriber(config.rostopic, config.msg_type)
            )

        # WARNING: TimeSynchronizer did not work. Use ApproximateTimeSynchronizer instead.
        # self.time_sync = message_filters.TimeSynchronizer(self.subscribers, 10)
        self.time_sync = message_filters.ApproximateTimeSynchronizer(
            self.subscribers, queue_size=10, slop=0.05
        )
        self.time_sync.registerCallback(self.cb)

        time.sleep(0.25)

    def cb(self, *inputs):
        raw_data = DatasetSample(self.dataset_config)

        config: Hdf5EntryConfig
        for input_msg, config in zip(inputs, self.dataset_config):
            raw_data[config.dataset_name] = config.processing_cb(input_msg)

        self.raw_data = raw_data

        if time.time() - self.collection_timer > 1 / self.collection_freq:
            # print(f"Collection freq: {1/(time.time() - self.collection_timer)}")
            # print(f"Data queue size: {self.data_queue.qsize()}")
            self.data_queue.put(self.raw_data)
            self.collection_timer = time.time()

        if self.print_cb_freq:
            print(f"Callback freq: {1/(time.time() - self.cb_timer)}")
            self.cb_timer = time.time()


def main():

    from hdf5_saver.custom_configs.hand_eye_dvrk_config import HandEyeHdf5Config

    config_list = [
        HandEyeHdf5Config.camera_l,
        HandEyeHdf5Config.psm1_measured_cp,
        HandEyeHdf5Config.psm1_measured_jp,
    ]
    dataset_config = Hdf5FullDatasetConfig.create_from_enum_list(config_list)

    data_queue = Queue()
    sync_client = SyncRosClient(
        dataset_config=dataset_config,
        collection_freq=10,
        data_queue=data_queue,
        print_cb_freq=False,
    )
    sync_client.wait_for_data()
    data = sync_client.get_data()
    print("data received!")
    print(data[HandEyeHdf5Config.camera_l.value[0]].shape)
    print(data[HandEyeHdf5Config.psm1_measured_cp.value[0]])
    print(data[HandEyeHdf5Config.psm1_measured_jp.value[0]])
    print(f"queue size {data_queue.qsize()}")

    rospy.spin()

    print(f"\nqueue size {data_queue.qsize()}")


if __name__ == "__main__":
    main()
