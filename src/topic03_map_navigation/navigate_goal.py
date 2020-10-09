#!/usr/bin/env python  
import rospy
import numpy as np

from PIL import Image as PILImage
import actionlib
import torch
from cv2 import cv2
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import *
from geometry_msgs.msg import Point
import visualpriors
import torchvision.transforms.functional as TF
from sensor_msgs.msg import CompressedImage, Image, LaserScan
from src.sim.ros.python3_ros_ws.src.vision_opencv.cv_bridge.python.cv_bridge import CvBridge
from src.sim.ros.src.utils import process_compressed_image
from src.data.dataset_labels import SEGMENTATION_CLASSES, semantic_masks, IMAGENET_CLASSES

bridge = CvBridge()


# this method will make the robot move to the goal location
def move_to_goal(x_goal, y_goal):
    print('navigate to ('+str(x_goal)+', '+str(y_goal)+')')
    # define a client for to send goal requests to the move_base server through a SimpleActionClient
    ac = actionlib.SimpleActionClient("move_base", MoveBaseAction)

    # wait for the action server to come up
    while not ac.wait_for_server(rospy.Duration.from_sec(5.0)):
        rospy.loginfo("Waiting for the move_base action server to come up")

    goal = MoveBaseGoal()

    # set up the frame parameters
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()

    # moving towards the goal*/

    goal.target_pose.pose.position = Point(x_goal, y_goal, 0)
    goal.target_pose.pose.orientation.x = 0.0
    goal.target_pose.pose.orientation.y = 0.0
    goal.target_pose.pose.orientation.z = 0.0
    goal.target_pose.pose.orientation.w = 1.0

    rospy.loginfo("Sending goal location ...")
    ac.send_goal(goal)

    ac.wait_for_result(rospy.Duration(60))

    if ac.get_state() == GoalStatus.SUCCEEDED:
        rospy.loginfo("You have reached the destination")
        return True

    else:
        rospy.loginfo("The robot failed to reach the destination")
        return False


def process_image(data):
    image = process_compressed_image(data, sensor_stats={'height': 256,
                                                         'width': 256,
                                                         'depth': 3})
    output_images = [PILImage.fromarray((image * 255).astype(np.uint8))]

    o_t = TF.to_tensor(image) * 2 - 1
    o_t = o_t.unsqueeze_(0)
    task = 'segment_semantic'
    pred = visualpriors.feature_readout(o_t, task, device='cpu')
    labels = semantic_masks(pred)

    output_images.append(PILImage.fromarray(labels))

    widths, heights = zip(*(i.size for i in output_images))

    total_width = sum(widths)
    max_height = max(heights)
    new_im = PILImage.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in output_images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    cv2.imshow('image', np.asarray(new_im))
    cv2.waitKey(1)


if __name__ == '__main__':
    rospy.init_node('map_navigation', anonymous=False)

    sensor_topic = '/raspicam_node/image/compressed'
    rospy.Subscriber(name=sensor_topic,
                     data_class=CompressedImage,
                     callback=process_image)
    print('start go to goal')

    move_to_goal(2.0, -1.3)
    move_to_goal(4.0, -1.3)
    move_to_goal(6.7, -3.5)
    move_to_goal(5.0, -3.8)
    move_to_goal(2.0, -1.3)
    rospy.spin()
