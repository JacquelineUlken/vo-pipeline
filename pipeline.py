from dataset import Dataset
import cv2
import numpy as np
from state import State
from tqdm import tqdm


class Pipeline:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.camera_matrix = dataset.camera_matrix
        self.state = State.empty()  # State object as described in the project statement pdf

        self.check_new_landmarks = False

    def run(self):
        number_of_frames = len(self.dataset)
        self.initialize(0, 2)
        poses = []
        for i in tqdm(range(1, number_of_frames), desc=f"Processing {number_of_frames - 1} frames."):
            if len(self.state.landmarks) < 100:
                self.check_new_landmarks = True
            current_pose = self.process_frame(i)
            poses.append(current_pose)

        return np.array(poses)

    def initialize(self, frame_1_index, frame_2_index):
        """
        Extracts an initial set of 2D - 3D correspondences from the first frames of the sequence and bootstraps the initial camera poses and landmarks.
        """
        # Select two frames at the beginning of the dataset
        frame_1 = self.dataset.get_frame(frame_1_index)
        frame_2 = self.dataset.get_frame(frame_2_index)

        # Establish keypoint correspondences between the two frames using KLT
        keypoints_1 = cv2.goodFeaturesToTrack(frame_1, maxCorners=500, qualityLevel=0.01, minDistance=7)  # shape: (K, 1, 2)
        keypoints_2, untracked_filter, _ = cv2.calcOpticalFlowPyrLK(frame_1, frame_2, keypoints_1, None)  # shapes: (K, 1, 2) and (K, 2)

        # Remove keypoints that aren't tracked
        keypoints_1 = keypoints_1[untracked_filter == 1]  # shape: (K, 2)
        keypoints_2 = keypoints_2[untracked_filter == 1]  # shape: (K, 2)

        # Get poses for frame 1 and frame 2
        pose_1 = np.eye(3, 4)  # First pose is just the (3, 4) identity matrix
        pose_2, outlier_filter = self.get_pose_and_outliers(keypoints_1, keypoints_2)

        # Remove outliers
        keypoints_1 = np.expand_dims(keypoints_1, axis=1)[outlier_filter == 1]  # shape: (K, 2)
        keypoints_2 = np.expand_dims(keypoints_2, axis=1)[outlier_filter == 1]  # shape: (K, 2)

        # Triangulate the landmarks using frame 1 and frame 2
        landmarks = self.triangulate_landmarks(keypoints_1, keypoints_2, pose_1, pose_2)  # shape: (K, 3)

        # Update state with initial keypoints and landmarks
        self.state.keypoints = keypoints_1  # shape: (K, 2)
        self.state.landmarks = landmarks  # shape: (K, 3)

    def process_frame(self, i):
        """
        Processes each frame, estimates the current pose of the camera using the existing set of landmarks and regularly triangulates new landmarks.
        """
        previous_image = self.dataset.get_frame(i - 1)
        current_image = self.dataset.get_frame(i)

        previous_keypoints = self.associate_keypoints_to_landmarks(previous_image, current_image)
        current_pose = self.estimate_current_pose(previous_keypoints)
        self.triangulate_new_landmarks(previous_image, current_image, current_pose)

        return current_pose

    def associate_keypoints_to_landmarks(self, previous_image, current_image):
        # Save previous keypoints
        previous_keypoints = self.state.keypoints.copy()

        # Establish keypoint correspondences between the two frames using KLT
        current_keypoints, untracked_filter, _ = cv2.calcOpticalFlowPyrLK(previous_image, current_image, self.state.keypoints, None)

        # Update state with new keypoints
        self.state.keypoints = current_keypoints

        # Remove keypoints and landmarks that aren't tracked
        previous_keypoints = np.expand_dims(previous_keypoints, axis=1)[untracked_filter == 1]
        self.state.filter_keypoints(untracked_filter)

        return previous_keypoints

    def estimate_current_pose(self, previous_keypoints):
        # Get the current pose
        current_pose, outlier_filter = self.get_pose_and_outliers(previous_keypoints, self.state.keypoints)

        # Remove outliers
        self.state.filter_keypoints(outlier_filter)

        return current_pose

    def triangulate_new_landmarks(self, previous_image, current_image, current_pose):
        if self.state.candidate_keypoints.shape[0] > 0:
            # Track existing candidate keypoints
            self.state.candidate_keypoints, untracked_filter, _ = cv2.calcOpticalFlowPyrLK(previous_image, current_image, self.state.candidate_keypoints, None)
            self.state.filter_candidate_keypoints(untracked_filter)

            # Triangulate new landmarks
            if self.check_new_landmarks:
                new_keypoints = []
                new_landmarks = []
                candidates_to_be_removed = []
                for i in range(len(self.state.candidate_keypoints)):
                    keypoint_1 = self.state.first_observations[i]
                    pose_1 = self.state.first_observation_poses[i]
                    keypoint_2 = self.state.candidate_keypoints[i]
                    pose_2 = current_pose
                    if self.is_valid_triangulation(keypoint_1, keypoint_2, pose_1, pose_2):
                        landmark = self.triangulate_landmarks(np.expand_dims(keypoint_1, axis=0), np.expand_dims(keypoint_2, axis=0), pose_1, pose_2).squeeze()
                        new_keypoints.append(keypoint_2)
                        new_landmarks.append(landmark)
                        candidates_to_be_removed.append(i)
                if new_keypoints:
                    self.state.add_keypoints(np.array(new_keypoints), np.array(new_landmarks))
                    self.state.remove_candidate_keypoints(candidates_to_be_removed)

        # Find new candidate keypoints
        n = 500 - len(self.state.candidate_keypoints)
        if n > 0:
            new_candidate_keypoints = cv2.goodFeaturesToTrack(current_image, maxCorners=n, qualityLevel=0.01, minDistance=7).reshape(-1, 2)  # shape: (K, 2)
            self.state.add_candidate_keypoints(new_candidate_keypoints, current_pose)

    def is_valid_triangulation(self, keypoint_1, keypoint_2, pose_1, pose_2):
        """
        Check if triangulation is valid by calculating the angle between bearing vectors.
        """
        keypoint_1_hom = np.append(keypoint_1, 1)
        keypoint_2_hom = np.append(keypoint_2, 1)

        # Backproject to normalized camera coordinates
        r1 = np.linalg.inv(self.camera_matrix) @ keypoint_1_hom
        r2 = np.linalg.inv(self.camera_matrix) @ keypoint_2_hom

        # Extract rotation and translation from camera poses
        R1, t1 = pose_1[:3, :3], pose_1[:3, 3]
        R2, t2 = pose_2[:3, :3], pose_2[:3, 3]

        # Transform rays to world coordinates
        d1 = R1 @ r1
        d2 = R2 @ r2

        # Normalize the direction vectors
        d1 /= np.linalg.norm(d1)
        d2 /= np.linalg.norm(d2)

        # Calculate the angle between the two rays
        cos_alpha = np.dot(d1, d2)
        alpha = np.arccos(np.clip(cos_alpha, -1.0, 1.0))

        # Return True if angle exceeds the threshold
        return alpha > 1/36 * np.pi

    def get_pose_and_outliers(self, previous_keypoints, current_keypoints):
        """
        Estimate relative pose between the frames while using RANSAC to filter out outliers
        """
        essential_matrix, outlier_filter = cv2.findEssentialMat(current_keypoints, previous_keypoints,
                                                                cameraMatrix=self.camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, rotation_matrix, translation_vector, _ = cv2.recoverPose(essential_matrix, current_keypoints, previous_keypoints, cameraMatrix=self.camera_matrix)
        pose = np.hstack((rotation_matrix, translation_vector))

        return pose, outlier_filter

    def triangulate_landmarks(self, keypoints_1, keypoints_2, pose_1, pose_2):
        """
        Triangulate a point cloud of 3D landmarks
        """
        projection_matrix_1 = self.camera_matrix @ pose_1
        projection_matrix_2 = self.camera_matrix @ pose_2

        landmarks_hom = cv2.triangulatePoints(projection_matrix_1, projection_matrix_2, keypoints_1.T, keypoints_2.T)
        landmarks = landmarks_hom[:3] / landmarks_hom[3]
        return landmarks.T