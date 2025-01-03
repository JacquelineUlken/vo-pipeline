from dataset import Dataset
import cv2
import numpy as np


class Pipeline:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.camera_matrix = dataset.camera_matrix
        self.state = {
            'keypoints': None,
            'landmarks': None,
            'pose': np.eye(4)  # Start with identity pose
        }

    def initialize(self, frame1_index, frame2_index):
        # Select two frames at the beginning of the dataset
        frame1 = self.dataset.get_frame(frame1_index)
        frame2 = self.dataset.get_frame(frame2_index)

        # Establish keypoint correspondences between the two frames using KLT
        keypoints_1 = cv2.goodFeaturesToTrack(frame1, maxCorners=500, qualityLevel=0.01, minDistance=7)
        keypoints_2, status, _ = cv2.calcOpticalFlowPyrLK(frame1, frame2, keypoints_1, None)

        keypoints_1 = keypoints_1[status == 1]
        keypoints_2 = keypoints_2[status == 1]

        # Estimate relative pose between the frames while using RANSAC to filter out outliers
        E, inliers = cv2.findEssentialMat(keypoints_2, keypoints_1, cameraMatrix=self.camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, _ = cv2.recoverPose(E, keypoints_2, keypoints_1, cameraMatrix=self.camera_matrix)

        # Triangulate a point cloud of 3D landmarks
        projection_matrix_1 = self.camera_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
        projection_matrix_2 = self.camera_matrix @ np.hstack((R, t))

        homogeneous_keypoints_1 = cv2.convertPointsToHomogeneous(keypoints_1).reshape(-1, 3).T
        homogeneous_keypoints_2 = cv2.convertPointsToHomogeneous(keypoints_2).reshape(-1, 3).T

        homogeneous_landmarks = cv2.triangulatePoints(projection_matrix_1, projection_matrix_2, homogeneous_keypoints_1[:2], homogeneous_keypoints_2[:2])
        landmarks = homogeneous_landmarks[:3] / homogeneous_landmarks[3]

        # Update state with initial keypoints, landmarks, and pose
        self.state['keypoints'] = keypoints_1
        self.state['landmarks'] = landmarks.T
        self.state['pose'] = np.eye(4)
        self.state['pose'][:3, :3] = R
        self.state['pose'][:3, 3] = t.ravel()

    def process_frame(self, previous_state, previous_image, current_image):
        # TODO implement computation of current state and pose
        current_state = previous_state
        current_pose = None

        return current_state, current_pose

    def run(self):
        raise NotImplementedError()
