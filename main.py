import os

from preprocessing.dataframe_builder import walk
from utils.distributor import move_into_split

if __name__ == '__main__':
    # walk(["airstrip", "newclifford", "riverside56"], "koger")
    # walk(["part1", "part2", "part3", "part4", "part5", "part6", "part7"], "stolsnek")

    move_into_split()



