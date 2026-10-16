import os

from matplotlib import pyplot as plt

import config.config as cfg

import numpy as np
import pickle as pkl

from objects.dataframe import DataFrame


def view_dataframe(df):
    np_frame = np.load(os.path.join(cfg.get_frame_path(), df))

    dataframe = DataFrame.to_frame(np_frame)
    dataframe.show_rgb()
    dataframe.show_overlay()
    dataframe.show_degree()
    dataframe.show_distance()


if __name__ == '__main__':
    view_dataframe("df_part7_3_15_stolsnek.npz")
