from __future__ import annotations
from queue import Queue
import time
from typing import Dict
import numpy as np
import rospy
from dataclasses import dataclass, field
from abc import ABC
import message_filters
from hdf5_saver.SaverConfig import (
    RosTopics,
    get_topics_processing_cb,
    selected_topics,
    topic_to_key_in_container,
)


@dataclass
class DatasetSample:
    left_rgb_img: np.ndarray
    right_rgb_img: np.ndarray
    measured_cp: np.ndarray = None
    measured_jp: np.ndarray = None

    @classmethod
    def from_dict(cls: DatasetSample, data: dict[RosTopics, np.ndarray]):
        # Map the keys to the class attributes
        dict_variables = {}
        for ros_topic_config, value in data.items():
            dict_variables[ros_topic_config.value[2]] = value
        return cls(**dict_variables)

    def to_dict(self) -> Dict[str, np.ndarray]:
        """
        Map a `DatasetSample` to a dictionary that can be consumed by the
        `DataContainer`.  Keys in the output dictionary are defined by the
        `topic_to_key_in_container` config dict
        """
        new_dict = {}

        for topic, new_key in topic_to_key_in_container.items():
            new_dict[new_key] = getattr(self, topic.value[2])

        return new_dict


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
    data_queue: Queue = None

    def __post_init__(self):
        if self.data_queue is None:
            raise ValueError("data_queue should be provided in constructor")

        super().__post_init__()
        self.subscribers = []
        self.callback_dict = get_topics_processing_cb()

        for topic in selected_topics:
            self.subscribers.append(
                message_filters.Subscriber(topic.value[0], topic.value[1])
            )

        # WARNING: TimeSynchronizer did not work. Use ApproximateTimeSynchronizer instead.
        # self.time_sync = message_filters.TimeSynchronizer(self.subscribers, 10)
        self.time_sync = message_filters.ApproximateTimeSynchronizer(
            self.subscribers, queue_size=10, slop=0.05
        )
        self.time_sync.registerCallback(self.cb)

        time.sleep(0.25)

    def cb(self, *inputs):
        raw_data_dict = {}
        for input_msg, topic in zip(inputs, selected_topics):
            raw_data_dict[topic] = self.callback_dict[topic](input_msg)

        self.raw_data = DatasetSample.from_dict(raw_data_dict)

        self.data_queue.put(self.raw_data)


def main():
    sync_client = SyncRosClient()
    sync_client.wait_for_data()
    data = sync_client.get_data()
    print("data received!")
    print(data.left_rgb_img.shape)
    print(data.measured_jp)


if __name__ == "__main__":
    main()
