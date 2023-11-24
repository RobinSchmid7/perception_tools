#!/usr/bin/env python

"""
Publish torch image as grid map topic using the timestamp from the filename.
"""

import os
import rospy
import torch
from tqdm import tqdm
import time
import numpy as np
from grid_map_msgs.msg import GridMap, GridMapInfo
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Float32MultiArray, MultiArrayDimension


class GridMapConverter(object):
    def __init__(self):
        rospy.init_node("projector")

        self.pub_clock = rospy.Publisher("clock", Clock, queue_size=1)
        self.pub_grid_map = rospy.Publisher("target", GridMap, queue_size=1)

    def torch_array_to_grid_map(self, arr, res, layers, timestamp, reference_frame="odom", x=0, y=0):
        size_x = arr.shape[1]
        size_y = arr.shape[2]

        data_dim_0 = MultiArrayDimension()
        data_dim_0.label = "column_index"  # y dimension
        data_dim_0.size = size_y  # number of columns which is y
        data_dim_0.stride = size_y * size_x  # rows*cols
        data_dim_1 = MultiArrayDimension()
        data_dim_1.label = "row_index"  # x dimension
        data_dim_1.size = size_x  # number of rows which is x
        data_dim_1.stride = size_x  # number of rows
        data = []

        for i in range(arr.shape[0]):
            data_tmp = Float32MultiArray()
            data_tmp.layout.dim.append(data_dim_0)
            data_tmp.layout.dim.append(data_dim_1)
            data_tmp.data = arr[i, ::-1, ::-1].transpose().ravel()
            data.append(data_tmp)

        info = GridMapInfo()
        info.pose.orientation.w = 1
        info.header.seq = 0
        info.header.stamp = timestamp
        info.resolution = res
        info.length_x = size_x * res
        info.length_y = size_y * res
        info.header.frame_id = reference_frame
        info.pose.position.x = x
        info.pose.position.y = y
        gm_msg = GridMap(info=info, layers=layers, basic_layers=[], data=data)

        return gm_msg


if __name__ == "__main__":
    print("Start")

    image_dir = "/home/rschmid/RosBags/bevnet/mask"
    # File name format: 1677752045_162152.pt
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".pt")])

    # Initialize and run node
    gc = GridMapConverter()

    while not rospy.is_shutdown():
        for image_file in tqdm(image_files):

            # image file name to ros time stamp
            ts = image_file.replace(".pt", "")
            ts = ts.replace("_", ".")

            ts = rospy.Time.from_sec(float(ts))

            if rospy.is_shutdown():
                break

            img_path = os.path.join(image_dir, image_file)

            img = torch.load(img_path, map_location=torch.device("cpu")).cpu().numpy()

            if img is None:
                continue

            img_torch = img[np.newaxis, ...].astype(np.uint8)

            mask_msg = gc.torch_array_to_grid_map(img_torch, res=0.1, layers=["target"],
                                                    timestamp=ts,
                                                    reference_frame="odom", x=0, y=0)

            # Hack to also publish a clock and record a new bag
            gc.pub_clock.publish(ts)
            gc.pub_grid_map.publish(mask_msg)

            time.sleep(0.2)

        break