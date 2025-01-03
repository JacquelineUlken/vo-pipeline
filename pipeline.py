from dataset import Dataset
import cv2
import numpy as np


class Pipeline:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.camera_matrix = dataset.camera_matrix

    def initialize(self, frame1_index, frame2_index):
        frame1 = self.dataset.get_frame(frame1_index)
        frame2 = self.dataset.get_frame(frame2_index)

        keypoints_1 = cv2.goodFeaturesToTrack(frame1, maxCorners=500, qualityLevel=0.01, minDistance=7)
        keypoints_2, status, _ = cv2.calcOpticalFlowPyrLK(frame1, frame2, keypoints_1, None)

        keypoints_1 = keypoints_1[status == 1]
        keypoints_2 = keypoints_2[status == 1]

        E, inliers = cv2.findEssentialMat(keypoints_2, keypoints_1, cameraMatrix=self.camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, _ = cv2.recoverPose(E, keypoints_2, keypoints_1, cameraMatrix=self.camera_matrix)

        P1 = self.camera_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = self.camera_matrix @ np.hstack((R, t))

        keypoints_1_h = cv2.convertPointsToHomogeneous(keypoints_1).reshape(-1, 3).T
        keypoints_2_h = cv2.convertPointsToHomogeneous(keypoints_2).reshape(-1, 3).T

        landmarks_h = cv2.triangulatePoints(P1, P2, keypoints_1_h[:2], keypoints_2_h[:2])
        landmarks = landmarks_h[:3] / landmarks_h[3]

        initial_landmarks = landmarks.T

        initial_pose = np.eye(4)
        next_pose = np.eye(4)
        next_pose[:3, :3] = R
        next_pose[:3, 3] = t.ravel()

        return initial_landmarks, initial_pose, next_pose

    def run(self):
        raise NotImplementedError()
