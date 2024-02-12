import pytest
from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np
import os
import h5py
from enum import Enum
from typing import Dict, List, Tuple
from hdf5_saver.Hdf5Writer import (
    Hdf5EntryConfig,
    Hdf5FullDatasetConfig,
    HDF5Writer,
    DataContainer,
)


@pytest.fixture
def config_batch_100() -> Hdf5FullDatasetConfig:
    my_config = Hdf5EntryConfig(
        "camera_l", (100, 480, 640, 3), (None, 480, 640, 3), "gzip", np.uint8
    )

    return Hdf5FullDatasetConfig([my_config])


@pytest.fixture
def hdf5_writer_conf_100(config_batch_100):
    file_name = "test_" + time.strftime("%Y%m%d_%H%M%S") + ".hdf5"
    h5_writer = HDF5Writer(Path("temp"), config_batch_100, file_name)

    yield h5_writer

    # teardown fixture system of pytest
    # https://docs.pytest.org/en/7.1.x/how-to/fixtures.html
    h5_writer.file_path.unlink()


# @pytest.fixture
# def config_batch_20() -> List[Hdf5EntryConfig]:
#     my_config = Hdf5EntryConfig(
#         "camera_l", (20, 480, 640, 3), (None, 480, 640, 3), "gzip", np.uint8
#     )
#     return [my_config]


# @pytest.fixture
# def config_multidata() -> List[Hdf5FullDatasetConfig]:
#     my_config1 = Hdf5EntryConfig(
#         "camera_l", (20, 480, 640, 3), (None, 480, 640, 3), "gzip", np.uint8
#     )
#     my_config2 = Hdf5EntryConfig(
#         "measured_cp", (20, 480, 640, 3), (None, 480, 640, 3), "gzip", np.uint8
#     )

#     return [my_config1, my_config2]


def test_single_chunk_writing(
    hdf5_writer_conf_100: HDF5Writer, config_batch_100: List[Hdf5EntryConfig]
):

    data_container = DataContainer(config_batch_100)
    hdf5_writer = hdf5_writer_conf_100

    for i in range(config_batch_100[0].chunk_size):
        data_dict = {}
        data_dict[config_batch_100[0].dataset_name] = (
            np.ones((480, 640, 3), dtype=np.uint8) + i
        )

        data_container.add_data(data_dict)

    with hdf5_writer as writer:
        writer.write_chunk(data_container)

    with h5py.File(hdf5_writer.file_path, "r") as f:
        print(f.keys())
        print(f["data"].keys())
        print(f["data"]["camera_l"].shape)

        img_data = f["data"]["camera_l"][:]
        assert np.all(img_data[0] == np.ones((480, 640, 3), dtype=np.uint8))
        assert np.all(img_data[3] == (np.ones((480, 640, 3), dtype=np.uint8) + 3))
        assert np.all(img_data[63] == (np.ones((480, 640, 3), dtype=np.uint8) + 63))


def test_multichunk_writing_and_datatypes(
    hdf5_writer_conf_100: HDF5Writer, config_batch_100: List[Hdf5EntryConfig]
):

    dataset_config = Hdf5FullDatasetConfig(config_batch_100)
    data_container = DataContainer(dataset_config)
    hdf5_writer = hdf5_writer_conf_100

    with hdf5_writer as writer:
        for i in range(dataset_config[0].chunk_size * 3 + 1):
            if data_container.is_full():
                writer.write_chunk(data_container)
                data_container = DataContainer(dataset_config)

            data_dict = {}
            data_dict[dataset_config[0].dataset_name] = (
                np.ones((480, 640, 3), dtype=np.uint8) + i
            )

            data_container.add_data(data_dict)

    with h5py.File(hdf5_writer.file_path, "r") as f:
        img_data = f["data"]["camera_l"][:]
        assert np.all(img_data[0] == np.ones((480, 640, 3), dtype=np.uint8))
        assert np.all(img_data[3] == (np.ones((480, 640, 3), dtype=np.uint8) + 3))
        assert np.all(img_data[20] == (np.ones((480, 640, 3), dtype=np.uint8) + 20))
        assert np.all(img_data[40] == (np.ones((480, 640, 3), dtype=np.uint8) + 40))
        # # Max value of uint8 is 255
        assert np.all(img_data[255] == (np.zeros((480, 640, 3), dtype=np.uint8)))


def test_writing_one_chunk_and_half(
    hdf5_writer_conf_100: HDF5Writer, config_batch_100: Hdf5FullDatasetConfig
):

    dataset_config = config_batch_100
    h5_writer = hdf5_writer_conf_100
    data_container = DataContainer(dataset_config)

    chunk_size = dataset_config[0].chunk_size

    with h5_writer as writer:
        for i in range(chunk_size + chunk_size // 2):
            if data_container.is_full():
                writer.write_chunk(data_container)
                data_container = DataContainer(dataset_config)

            data_dict = {}
            data_dict[dataset_config[0].dataset_name] = (
                np.ones((480, 640, 3), dtype=np.uint8) + i
            )
            data_container.add_data(data_dict)

        if len(data_container) > 0:
            writer.write_chunk(data_container)

    with h5py.File(h5_writer.file_path, "r") as f:
        img_data = f["data"]["camera_l"][:]
        assert np.all(img_data[0] == np.ones((480, 640, 3), dtype=np.uint8))
        assert np.all(img_data[23] == (np.ones((480, 640, 3), dtype=np.uint8) + 23))
        assert img_data.shape[0] == chunk_size + chunk_size // 2
