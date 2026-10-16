import os.path

import numpy as np
from ray import tune

from model.unet.unet_model import UNET
from model.dense121_unet.dense121_unet_model import DenseNet121

config = {
    "base_path": "C:/Users/s371513/PycharmProjects/apseg/",
    "raw_data_path": "E:/master/",
    "frame_path_cluster": "E:/frames_cluster/",
    "frame_path_buffer": "E:/frames_buffer/",
    "split": {
        "train": (0.7, "train/"),
        "validation": (0.2, "validation/"),
        "test": (0.1, "test/")
    },
    "features": {
        "completelines_clipped_buffered.gpkg": 1,
        "completecross_clipped.gpkg": 1
    },
    "auxiliary_rasters": {
        "degree.tif",
        "distance.tif"
    },
    "no_data_value": -2147483648,
    "num_workers": 0,
    "pin_memory": True,
    "blob_size": 512,
    "r_dll": r"C:\Program Files\R\R-4.3.1\bin\x64",
    "r_home": r"C:\Program Files\R\R-4.3.1",
    "ray_config": {
        "lr": tune.grid_search([5e-05]),
        "batch_size": tune.grid_search([8]),
        "aspp": tune.grid_search([True]),
        "attention": tune.grid_search([False]),
        "aux": tune.grid_search([False]),
        "frames": tune.grid_search(["buffer", "cluster"]),
    },
    "device": "cuda:0"
}


def get_frame_path(version):
    return config["frame_path_{}".format(version)]


def get_train_path(version):
    return os.path.join(get_frame_path(version), config["split"]["train"][1])


def get_validation_path(version):
    return os.path.join(get_frame_path(version), config["split"]["validation"][1])


def get_test_path(version):
    return os.path.join(get_frame_path(version), config["split"]["test"][1])


def batch_size():
    return config["batch_size"]


def num_workers():
    return config["num_workers"]


def pin_memory():
    return config["pin_memory"]


def blob_size():
    return config["blob_size"]


def get_base_path():
    return config["base_path"]


def get_raw_data_path():
    return config["raw_data_path"]


def get_r_home():
    return config["r_home"]


def get_r_x64_dll():
    return config["r_dll"]


def get_features():
    return config["features"]


def get_auxiliary_raster():
    return config["auxiliary_rasters"]


def get_split_map():
    return config["split"]


def get_no_data_value():
    return config["no_data_value"]


def get_ray_config():
    return config["ray_config"]


def get_device():
    return config["device"]

