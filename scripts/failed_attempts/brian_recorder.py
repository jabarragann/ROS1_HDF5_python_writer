#!/usr/bin/env python
import numpy as np
import cv2
import os
from datetime import datetime
import numpy as np
import time
import math
import keyboard

# for ros stuff
import rospy
from std_msgs.msg import String, Float64MultiArray, Bool, Float64
from geometry_msgs.msg import Vector3, Transform, PoseStamped
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage, JointState
from std_msgs.msg import Int32
import pandas as pd
import dynamic_reconfigure.client
from concurrent.futures import ProcessPoolExecutor

import threading
import queue

usb_image_left = None
usb_image_right = None
endo_cam_psm1 = None
endo_cam_psm2 = None
psm1_pose = None
psm1_sp = None
psm1_rcm_pose = None
psm1_jaw = None
psm1_jaw_sp = None

psm2_pose = None
psm2_sp = None
psm2_rcm_pose = None
psm2_jaw = None
psm2_jaw_sp = None

ecm_pose = None
ecm_rcm_pose = None


class ros_topics:

    def __init__(self):
        self.bridge = CvBridge()

        # subscribers
        self.usb_camera_sub_left = rospy.Subscriber(
            "/jhu_daVinci/decklink/left/image_raw", Image, self.get_camera_image_left
        )

        # psm2
        self.psm2_sub = rospy.Subscriber(
            "/PSM2/measured_cp", PoseStamped, self.get_psm2_pose
        )

    def get_camera_image_left(self, data):
        global usb_image_left
        usb_image_left = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")

    def get_camera_image_right(self, data):
        global usb_image_right
        usb_image_right = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")

    def get_psm2_pose(self, data):
        global psm2_pose
        psm2_pose = data.pose


def image_saver(queue):
    while True:
        item = queue.get()
        if item is None:
            break  # None is our signal to stop
        filename, image = item
        cv2.imwrite(filename, image)
        queue.task_done()


if __name__ == "__main__":
    # Create a queue to communicate with the worker thread
    image_queue = queue.Queue()

    # Start worker thread
    worker = threading.Thread(target=image_saver, args=(image_queue,))
    worker.start()

    # Create ROS publishers and subscribers
    bridge = CvBridge()
    rospy.init_node("rostopic_recorder", anonymous=True)
    rt = ros_topics()
    time.sleep(0.5)

    rate = rospy.Rate(30)  # ROS Rate at 5Hz
    requiresNewDir = True
    requiresSaveCsv = False
    num_frames = 0
    ee_points = []

    time.sleep(1)

    isRecord = True

    while True:
        if isRecord:
            # create a new dir in the beginning
            if requiresNewDir:
                time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
                ep_dir = os.path.join("_recordings", time_stamp)
                left_img_dir = os.path.join(ep_dir, "left_img_dir")
                right_img_dir = os.path.join(ep_dir, "right_img_dir")
                endo_p1_dir = os.path.join(ep_dir, "endo_psm1")
                endo_p2_dir = os.path.join(ep_dir, "endo_psm2")

            # also reset indices and other stuff
            num_frames = 0
            ee_points = []

            if not os.path.exists(ep_dir):
                os.makedirs(ep_dir)
                os.makedirs(left_img_dir)
                os.makedirs(right_img_dir)
                os.makedirs(endo_p1_dir)
                os.makedirs(endo_p2_dir)

            requiresNewDir = False
            # since we just made a new dir, we need to save csv later
            requiresSaveCsv = True

            # PyKDL.Frame
            ee_points.append(
                [
                    psm2_pose.position.x,
                    psm2_pose.position.y,
                    psm2_pose.position.z,  # PSM 2
                    psm2_pose.orientation.x,
                    psm2_pose.orientation.y,
                    psm2_pose.orientation.z,
                    psm2_pose.orientation.w,
                ]
            )

            # save frame
            time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            save_name_left = os.path.join(
                left_img_dir, "frame{:06d}_left".format(num_frames) + ".jpg"
            )
            save_name_right = os.path.join(
                right_img_dir, "frame{:06d}_right".format(num_frames) + ".jpg"
            )
            save_name_endo_p1 = os.path.join(
                endo_p1_dir, "frame{:06d}_psm1".format(num_frames) + ".jpg"
            )
            save_name_endo_p2 = os.path.join(
                endo_p2_dir, "frame{:06d}_psm2".format(num_frames) + ".jpg"
            )

            # write the image
            # cv2.imwrite(save_name_left, usb_image_left) # UNCOMMENT ME WHEN DEBUGGING IS OVER (early)
            # cv2.imwrite(save_name_right, usb_image_right) # UNCOMMENT ME WHEN DEBUGGING IS OVER (early)
            # cv2.imwrite(save_name_endo_p1, endo_cam_psm1) # UNCOMMENT ME WHEN DEBUGGING IS OVER (early)
            # cv2.imwrite(save_name_endo_p2, endo_cam_psm2) # UNCOMMENT ME WHEN DEBUGGING IS OVER (early)

            image_queue.put(
                (save_name_left, cv2.cvtColor(usb_image_left, cv2.COLOR_BGR2RGB))
            )
            image_queue.put(
                (save_name_right, cv2.cvtColor(usb_image_right, cv2.COLOR_BGR2RGB))
            )
            image_queue.put((save_name_endo_p1, endo_cam_psm1))
            image_queue.put((save_name_endo_p2, endo_cam_psm2))

            num_frames = num_frames + 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        else:
            # if not recording, the flag for creating a new directory should be set to true
            requiresNewDir = True
            cv2.destroyAllWindows()
            if requiresSaveCsv is True:
                # save ee points
                header = [
                    "psm2_pose.position.x",
                    "psm2_pose.position.y",
                    "psm2_pose.position.z",
                    "psm2_pose.orientation.x",
                    "psm2_pose.orientation.y",
                    "psm2_pose.orientation.z",
                    "psm2_pose.orientation.w",
                ]

            csv_data = pd.DataFrame(ee_points)
            ee_save_path = os.path.join(ep_dir, "ee_csv.csv")
            csv_data.to_csv(ee_save_path, index=False, header=header)

            # make sure to set this back to False
            requiresSaveCsv = False

        # make sure we spin at 30hz
        rate.sleep()

        # When everything done, destroy all windows
        cv2.destroyAllWindows()
