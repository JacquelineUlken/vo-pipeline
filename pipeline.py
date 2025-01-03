from dataset import Dataset
import cv2
import numpy as np


class Pipeline:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.camera_matrix = dataset.camera_matrix
        self.state = {
            'keypoints': np.array([]),
            'landmarks': np.array([]),
        }

    def initialize(self, frame1_index, frame2_index):
        # Select two frames at the beginning of the dataset
        frame1 = self.dataset.get_frame(frame1_index)
        frame2 = self.dataset.get_frame(frame2_index)

        # Establish keypoint correspondences between the two frames using KLT
        keypoints_1 = cv2.goodFeaturesToTrack(frame1, maxCorners=500, qualityLevel=0.01, minDistance=7)
        keypoints_2, tracked_mask, _ = cv2.calcOpticalFlowPyrLK(frame1, frame2, keypoints_1, None)

        keypoints_1 = self.mask_points(keypoints_1, tracked_mask)
        keypoints_2 = self.mask_points(keypoints_2, tracked_mask)

        pose_1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        pose_2, inliers_mask = self.get_pose_and_inliers(keypoints_1, keypoints_2)

        inlier_keypoints_1 = self.mask_points(keypoints_1, inliers_mask)
        inlier_keypoints_2 = self.mask_points(keypoints_1, inliers_mask)

        landmarks = self.triangulate_landmarks(inlier_keypoints_1, inlier_keypoints_2, pose_1, pose_2)

        # Update state with initial keypoints and landmarks
        self.state['keypoints'] = inlier_keypoints_1
        self.state['landmarks'] = landmarks

    def process_frame(self, previous_state, previous_image, current_image):
        previous_keypoints = previous_state['keypoints']
        landmarks = previous_state['landmarks']

        current_keypoints, tracked_mask, _ = cv2.calcOpticalFlowPyrLK(previous_image, current_image, previous_keypoints, None)
        previous_keypoints = self.mask_points(previous_keypoints, tracked_mask)
        current_keypoints = self.mask_points(current_keypoints, tracked_mask)
        tracked_landmarks = self.mask_points(landmarks, tracked_mask)

        current_pose, inliers = self.get_pose_and_inliers(previous_keypoints, current_keypoints)

        inlier_current_keypoints = self.mask_points(current_keypoints, inliers)
        inlier_landmarks = self.mask_points(tracked_landmarks, inliers)

        current_state = {
            'keypoints': inlier_current_keypoints,
            'landmarks': inlier_landmarks,
        }

        return current_state, current_pose

    def get_pose_and_inliers(self, previous_keypoints, current_keypoints):
        # Estimate relative pose between the frames while using RANSAC to filter out outliers
        essential_matrix, inliers = cv2.findEssentialMat(current_keypoints, previous_keypoints, cameraMatrix=self.camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, rotation_matrix, translation_vector, _ = cv2.recoverPose(essential_matrix, current_keypoints, previous_keypoints, cameraMatrix=self.camera_matrix)

        pose = np.hstack((rotation_matrix, translation_vector))

        return pose, inliers

    def triangulate_landmarks(self, keypoints_1, keypoints_2, pose_1, pose_2):
        # Triangulate a point cloud of 3D landmarks
        projection_matrix_1 = self.camera_matrix @ pose_1
        projection_matrix_2 = self.camera_matrix @ pose_2

        landmarks_hom = cv2.triangulatePoints(projection_matrix_1, projection_matrix_2, keypoints_1.T, keypoints_2.T)
        landmarks = landmarks_hom[:3] / landmarks_hom[3]

        return landmarks.T

    def mask_points(self, points, mask):
        assert 2 <= len(points.shape) <= 3, f"Points should be of dimension (N, 1, d) or (N, d)"
        assert mask.shape == (points.shape[0], 1), f"Mask has wrong shape: {mask.shape}, should have {(points.shape[0], 1)}"
        if len(points.shape) == 2:
            masked_points = np.expand_dims(points, axis=1)[mask == 1]
        else:
            masked_points = points[mask == 1]

        return masked_points

    def run(self):
        number_of_frames = len(self.dataset)
        self.initialize(0, 2)
        poses = []
        for i in range(number_of_frames):
            number_of_landmarks = len(self.state['landmarks'])
            if number_of_landmarks < 10:
                # TODO implement triangulating new landmarks
                print()
                print("Not enough landmarks left. Implement triangulating new landmarks!")
                break
            print(f"\rNumber of landmarks: {number_of_landmarks}", end="")
            previous_state = self.state
            previous_image = self.dataset.get_frame(i)
            current_image = self.dataset.get_frame(i + 1)
            current_state, current_pose = self.process_frame(previous_state, previous_image, current_image)
            self.state = current_state
            poses.append(current_pose)
        return np.array(poses)