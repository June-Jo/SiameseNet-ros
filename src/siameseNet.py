#!/usr/bin/env python3
import numpy as np
import torch
import rospy
import os
import ros_numpy
import cv2

from siamesenet.srv import siameseSrv, siameseSrvResponse
from networks import EmbeddingNet, SiameseNet

class siameseNet():
    def __init__(self):
        # Servie handles
        rospy.init_node('siameseNet')
        rospy.Service('siameseNet', siameseSrv, self.imgMatch)

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        modelPath = '/home/june/catkin_ws/src/siamesenet/siameseNet.pt'
        self.warmup(modelPath)

        print("SiameseNet ready")

        rospy.spin()

    def imgMatch(self, req):
        rospy.wait_for_service('siameseNet')
        try:
            img1 = ros_numpy.numpify(req.Image1)
            img2 = ros_numpy.numpify(req.Image2)

            img1 = cv2.resize(img1, dsize=(224,224), interpolation=cv2.INTER_AREA)
            img2 = cv2.resize(img2, dsize=(224,224), interpolation=cv2.INTER_AREA)

            img1 = img1[:,:,:3]/255.0
            img2 = img2[:,:,:3]/255.0

            img1 = img1[:, :, ::-1]
            img2 = img2[:, :, ::-1]

            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])

            img1 = (img1-mean)/std
            img2 = (img2-mean)/std

            img1 = torch.from_numpy(img1).reshape(224,224,3,1).permute(3,2,0,1).type(torch.FloatTensor).to(self.device)
            img2 = torch.from_numpy(img2).reshape(224,224,3,1).permute(3,2,0,1).type(torch.FloatTensor).to(self.device)

            vec1, vec2 = self.model(img1, img2)
            similarity = (vec2 - vec1).pow(2).sum(1)
            similarity = 1/similarity.detach().cpu().numpy()

            print('similarity: ', similarity)

            return siameseSrvResponse(similarity)

        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    def warmup(self, modelPath):
        self.model = EmbeddingNet()
        self.model = SiameseNet(self.model)
        self.model.load_state_dict(torch.load(modelPath))
        self.model.to(self.device)
        self.model.eval()

if __name__ == "__main__":
    siamese = siameseNet()
