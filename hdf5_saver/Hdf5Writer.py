from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np
import os
import h5py
from enum import Enum
from enum import Enum
from typing import Dict, List, Tuple


@dataclass
class Hdf5EntryConfig:
    dataset_name: str
    chunk_shape: Tuple[int]
    max_shape: Tuple[int]
    compression: str
    dtype: np.dtype

    def __post_init__(self):
        self.chunk_size = self.chunk_shape[0]


@dataclass
class Hdf5FullDatasetConfig(list):
    data_config: List[Hdf5EntryConfig]

    def __post_init__(self):
        for config in self.data_config:
            self.append(config)

        self._idx = 0

    @classmethod
    def create_from_enum_list(cls, enum_list: List[Enum]):
        init_list = []
        for enum in enum_list:
            init_list.append(Hdf5EntryConfig(*enum.value))

        return cls(init_list)

    def __getitem__(self, idx) -> Hdf5EntryConfig:
        return self.data_config[idx]

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self) -> Hdf5EntryConfig:
        if self._idx < len(self.data_config):
            config = self.data_config[self._idx]
            self._idx += 1
            return config
        else:
            raise StopIteration


@dataclass
class DataContainer:
    """
    Container capable of storing fixed defined number of data points
    """

    dataset_config: Hdf5FullDatasetConfig

    def __post_init__(self):
        self._internal_data_dict: Dict[Hdf5EntryConfig, np.ndarray] = dict()
        self.internal_idx = 0
        self.max_idx = self.dataset_config[0].chunk_size

        for config in self.dataset_config:
            self._internal_data_dict[config.dataset_name] = np.zeros(
                config.chunk_shape, dtype=config.dtype
            )

    def __len__(self):
        return self.internal_idx

    def add_data(self, data_dict: dict):
        """
        Data is meant to be feed one element at a time.
        """

        if self.is_full():
            raise ValueError("Chunk is full")

        for key, value in data_dict.items():
            if key not in self._internal_data_dict:
                raise ValueError(f"Dataset {key} not found in the container config")

            try:
                self._internal_data_dict[key][self.internal_idx] = value
            except ValueError as e:
                raise ValueError(f"Error adding data to {key}. {e}")

        self.internal_idx += 1

    def is_full(self):
        return self.internal_idx >= self.max_idx

    def get_data(self, key):
        return self._internal_data_dict[key][: self.internal_idx][: self.internal_idx]


@dataclass
class HDF5Writer:
    output_dir: Path
    dataset_config: Hdf5FullDatasetConfig
    file_name: str = None

    def __post_init__(self):
        # Member attributes
        self.file_path = self._create_path()
        self.current_dataset_size = None
        self._internal_idx = 0

    def _create_path(self) -> Path:
        if not os.path.exists(self.output_dir):
            self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.file_name is not None:
            file_path = self.output_dir / self.file_name
        else:
            file_path = self.output_dir / (time.strftime("%Y%m%d_%H%M%S") + ".hdf5")

        return file_path

    def _query_permission_to_overwrite(self, file_path) -> bool:
        if self.file_path.exists():
            ans = input(
                f"HDF5 file {file_path}.already exists. Do you want to overwrite it? (y/n) "
            )

            if ans != "y":
                raise ValueError(
                    f"File {file_path} already exists. Please choose another name."
                )

    def _init_hdf5_file(self):

        self._query_permission_to_overwrite(self.file_path)
        self.hdf5_file_handler = h5py.File(self.file_path, "w")

        metadata = self.hdf5_file_handler.create_group("metadata")
        metadata.create_dataset(
            "README",
            data="All position information is in meters unless specified otherwise. \n"
            "Quaternion is a list in the order of [qx, qy, qz, qw]. \n"
            "Poses are defined to be T_world_obj. \n"
            "Depth in CV convention (corrected by extrinsic, T_cv_ambf). \n",
        )

        self.data_group = self.hdf5_file_handler.create_group("data")
        self.create_datasets()

    def __len__(self):
        return self._internal_idx

    def create_datasets(self):
        self.datasets_dict: Dict[str, h5py.Dataset] = dict()
        self.current_dataset_size = self.dataset_config[0].chunk_size

        for config in self.dataset_config:
            initial_shape = tuple([0] + list(config.chunk_shape[1:]))
            self.datasets_dict[config.dataset_name] = self.data_group.create_dataset(
                config.dataset_name,
                shape=initial_shape,
                maxshape=config.max_shape,
                compression=config.compression,
                dtype=config.dtype,
            )

    def write_chunk(self, data_container: DataContainer):

        self.increase_size(len(data_container))

        idx = self._internal_idx
        for config in self.dataset_config:
            new_data = data_container.get_data(config.dataset_name)
            self.datasets_dict[config.dataset_name][idx:] = new_data

        self._internal_idx += len(data_container)

    def increase_size(self, size_of_new_container):
        for config in self.dataset_config:
            new_size = list(config.chunk_shape)
            new_size[0] = size_of_new_container + self._internal_idx
            self.datasets_dict[config.dataset_name].resize(new_size)

    def __enter__(self):
        self._init_hdf5_file()

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hdf5_file_handler.close()


def test1():
    from hdf5_saver.custom_configs.hand_eye_dvrk_config import (
        HandEyeHdf5Config,
    )

    print("test1")
    dataset_config = Hdf5FullDatasetConfig.create_from_enum_list(
        [HandEyeHdf5Config.camera_l, HandEyeHdf5Config.camera_r]
    )

    h5_writer = HDF5Writer(Path("temp"), dataset_config)

    data_container = DataContainer(dataset_config)

    for i in range(dataset_config[0].chunk_size):
        data_dict = {}
        data_dict[HandEyeHdf5Config.camera_l.value[0]] = (
            np.ones((480, 640, 3), dtype=np.uint8) + i
        )
        data_dict[HandEyeHdf5Config.camera_r.value[0]] = (
            np.ones((480, 640, 3), dtype=np.uint8) + 2 * i
        )

        data_container.add_data(data_dict)

    with h5_writer as writer:
        writer.write_chunk(data_container)

    with h5py.File(h5_writer.file_path, "r") as f:
        print(f.keys())
        print(f["data"].keys())
        print(f["data"]["camera_l"].shape)
        print(f["data"]["camera_r"].shape)

        img_data = f["data"]["camera_l"][:]
        assert np.all(img_data[0] == np.ones((480, 640, 3), dtype=np.uint8))
        assert np.all(img_data[3] == (np.ones((480, 640, 3), dtype=np.uint8) + 3))


def test2():
    from hdf5_saver.custom_configs.hand_eye_dvrk_config import (
        HandEyeHdf5Config,
    )

    print("test2")
    dataset_config = Hdf5FullDatasetConfig.create_from_enum_list(
        [HandEyeHdf5Config.camera_l]
    )

    h5_writer = HDF5Writer(Path("temp"), dataset_config)

    data_container = DataContainer(dataset_config)

    with h5_writer as writer:
        for i in range(dataset_config[0].chunk_size * 3 + 1):
            if data_container.is_full():
                print("writing chunk")
                writer.write_chunk(data_container)
                data_container = DataContainer(dataset_config)

            data_dict = {}
            data_dict[HandEyeHdf5Config.camera_l.value[0]] = (
                np.ones((480, 640, 3), dtype=np.uint8) + i
            )

            data_container.add_data(data_dict)

    print(i)

    with h5py.File(h5_writer.file_path, "r") as f:
        print(f.keys())
        print(f["data"].keys())
        print(f["data"]["camera_l"].shape)

        img_data = f["data"]["camera_l"][:]
        assert np.all(img_data[0] == np.ones((480, 640, 3), dtype=np.uint8))
        assert np.all(img_data[3] == (np.ones((480, 640, 3), dtype=np.uint8) + 3))
        assert np.all(img_data[120] == (np.ones((480, 640, 3), dtype=np.uint8) + 120))
        assert np.all(img_data[220] == (np.ones((480, 640, 3), dtype=np.uint8) + 220))
        assert np.all(img_data[255] == (np.zeros((480, 640, 3), dtype=np.uint8)))


def test3():
    print("test3")

    my_config = Hdf5EntryConfig(
        "camera_l", (20, 480, 640, 3), (None, 480, 640, 3), "gzip", np.uint8
    )

    dataset_config = Hdf5FullDatasetConfig([my_config])
    h5_writer = HDF5Writer(Path("temp"), dataset_config)
    data_container = DataContainer(dataset_config)

    chunk_size = dataset_config[0].chunk_size

    with h5_writer as writer:
        for i in range(chunk_size + chunk_size // 2):
            if data_container.is_full():
                print("writing chunk")
                writer.write_chunk(data_container)
                data_container = DataContainer(dataset_config)

            data_dict = {}
            data_dict[my_config.dataset_name] = (
                np.ones((480, 640, 3), dtype=np.uint8) + i
            )
            data_container.add_data(data_dict)

        if len(data_container) > 0:
            print(len(data_container))
            print("writing last chunk")
            writer.write_chunk(data_container)

    with h5py.File(h5_writer.file_path, "r") as f:

        img_data = f["data"]["camera_l"][:]
        assert np.all(img_data[0] == np.ones((480, 640, 3), dtype=np.uint8))
        assert np.all(img_data[23] == (np.ones((480, 640, 3), dtype=np.uint8) + 23))
        assert img_data.shape[0] == chunk_size + chunk_size // 2


if __name__ == "__main__":
    test3()
