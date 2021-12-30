#!/usr/bin/env python3
import numpy as np
import torch
import rospy
import cv2
import ros_numpy
from sensor_msgs.msg import Image

from siamesenet.srv import siameseSrv, siameseSrvResponse
from networks import EmbeddingNet, SiameseNet

def imgSender(img1, img2):
    rospy.wait_for_service('siameseNet')
    try:
        siameseSrvClient = rospy.ServiceProxy('siameseNet', siameseSrv)
        resp = siameseSrvClient(img1, img2)

        return resp.similarity

    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)


if __name__ == "__main__":
    img1Path = '/home/june/catkin_ws/src/siamesenet/imgs/02_01.png'
    img1 = cv2.imread(img1Path, cv2.IMREAD_COLOR)
    img1Msg = ros_numpy.msgify(Image, img1, 'bgr8')

    for i in range(1, 7):
        img2Path = '/home/june/catkin_ws/src/siamesenet/imgs/{:02d}_02.png'.format(i)
        img2 = cv2.imread(img2Path, cv2.IMREAD_COLOR)
        img2Msg = ros_numpy.msgify(Image, img2, 'bgr8')

        sim = imgSender(img1Msg, img2Msg)
        print(i, sim)
