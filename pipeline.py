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

        pts1 = cv2.goodFeaturesToTrack(frame1, maxCorners=500, qualityLevel=0.01, minDistance=7)
        pts2, _, _ = cv2.calcOpticalFlowPyrLK(frame1, frame2, pts1, None)

        E, inliers = cv2.findEssentialMat(pts2, pts1, cameraMatrix=self.camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, _ = cv2.recoverPose(E, pts2, pts1, cameraMatrix=self.camera_matrix)

        P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = np.hstack((R, t))

        landmarks_h = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        landmarks = landmarks_h[:3] / landmarks_h[3]

        initial_landmarks = landmarks.T

        initial_pose = np.eye(4)  # First frame pose is identity
        pose_init = np.eye(4)
        pose_init[:3, :3] = R
        pose_init[:3, 3] = t.ravel()
        next_pose = pose_init  # Next frame pose

    def run(self):
        raise NotImplementedError()
