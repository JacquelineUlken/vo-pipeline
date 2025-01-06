from dataset import Dataset
from config import Config
import cv2
import numpy as np
from state import State
from tqdm import tqdm
from visualize import Visualization


class Pipeline:
    def __init__(self, dataset: Dataset, config: Config):
        self.dataset = dataset
        self.config = config
        self.camera_matrix = dataset.camera_matrix
        self.state = State.empty()  # State object as described in the project statement pdf
        self.use_ground_truth_for_triangulation = False  # For debugging purposes

        self.check_new_landmarks = False

    def run(self):
        number_of_frames = len(self.dataset)
        poses = []

        # Select two frames at the beginning of the dataset
        frame_1 = self.dataset.get_frame(0)
        frame_2 = self.dataset.get_frame(self.config.init_frame_2_index)

        current_pose = self.initialize(frame_1, frame_2)
        poses.append(current_pose)

        visualization = Visualization(self.dataset)
        visualization.number_of_landmarks.append(len(self.state.landmarks))
        visualization.number_of_candidates.append(0)

        current_image = frame_1

        try:
            for i in tqdm(range(1, number_of_frames), desc=f"Processing {number_of_frames - 1} frames."):
                previous_image = current_image
                current_image = self.dataset.get_frame(i)

                if len(self.state.landmarks) < self.config.min_landmarks:
                    self.check_new_landmarks = True

                current_pose = self.process_frame(i, previous_image, current_image)
                poses.append(current_pose)

                visualization.update(i, self.state, np.array(poses))

                self.check_new_landmarks = False

        except Exception as e:
            # Save the video even if something went wrong
            visualization.save_video()
            raise e

        visualization.save_video()

        return np.array(poses)

    def initialize(self, frame_1, frame_2):
        """
        Extracts an initial set of 2D - 3D correspondences from the first frames of the sequence and bootstraps the initial camera poses and landmarks.
        """

        # Establish keypoint correspondences between the two frames using KLT
        keypoints_1 = cv2.goodFeaturesToTrack(frame_1,
                                              maxCorners=self.config.init_max_corners,
                                              qualityLevel=self.config.init_quality_level,
                                              minDistance=self.config.init_min_distance)  # shape: (K, 1, 2)
        keypoints_2, tracked, _ = cv2.calcOpticalFlowPyrLK(frame_1, frame_2, keypoints_1, None)  # shapes: (K, 1, 2) and (K, 2)

        # Remove keypoints that aren't tracked
        keypoints_1 = keypoints_1[tracked == 1]  # shape: (K, 2)
        keypoints_2 = keypoints_2[tracked == 1]  # shape: (K, 2)

        keypoints_2, in_frame = self.filter_keypoints_in_frame(keypoints_2, frame_2.shape)
        keypoints_1 = np.expand_dims(keypoints_1, 1)[in_frame == 1]

        # Get poses for frame 1 and frame 2
        pose_1 = np.eye(4)  # First pose is just the (4, 4) identity matrix

        essential_matrix, inliers = cv2.findEssentialMat(keypoints_1, keypoints_2,
                                                         cameraMatrix=self.camera_matrix,
                                                         method=cv2.RANSAC,
                                                         prob=self.config.ransac_prob,
                                                         threshold=self.config.error_threshold)
        _, rotation_matrix, translation_vector, _ = cv2.recoverPose(essential_matrix, keypoints_1, keypoints_2, cameraMatrix=self.camera_matrix, mask=inliers)
        pose_2 = np.eye(4)
        pose_2[:3, :3] = rotation_matrix
        pose_2[:3, 3] = translation_vector.ravel()
        pose_2 = np.linalg.inv(pose_2)

        # Remove outliers
        keypoints_1 = np.expand_dims(keypoints_1, axis=1)[inliers == 1]  # shape: (K, 2)
        keypoints_2 = np.expand_dims(keypoints_2, axis=1)[inliers == 1]  # shape: (K, 2)

        # Triangulate the landmarks using frame 1 and frame 2
        landmarks = self.triangulate_landmarks(keypoints_1, keypoints_2, pose_1, pose_2)  # shape: (K, 3)

        # Update state with initial keypoints and landmarks
        self.state.keypoints = keypoints_1  # shape: (K, 2)
        self.state.landmarks = landmarks  # shape: (K, 3)

        return pose_1

    def process_frame(self, i, previous_image, current_image):
        """
        Processes each frame, estimates the current pose of the camera using the existing set of landmarks and regularly triangulates new landmarks.
        """

        self.associate_keypoints_to_landmarks(previous_image, current_image)
        current_pose = self.estimate_current_pose()

        if self.use_ground_truth_for_triangulation:
            ground_truth_pose = self.dataset.get_ground_truth_pose(i)
            self.triangulate_new_landmarks(previous_image, current_image, ground_truth_pose)
        else:
            self.triangulate_new_landmarks(previous_image, current_image, current_pose)

        return current_pose

    def associate_keypoints_to_landmarks(self, previous_image, current_image):
        # Save previous keypoints
        previous_keypoints = self.state.keypoints.copy()

        # Establish keypoint correspondences between the two frames using KLT
        current_keypoints, tracked, _ = cv2.calcOpticalFlowPyrLK(previous_image, current_image, previous_keypoints, None)

        # Update state with new keypoints
        self.state.keypoints = current_keypoints

        # Remove keypoints and landmarks that aren't tracked
        self.state.filter_keypoints(tracked)

    def estimate_current_pose(self):
        # Save current keypoints
        current_keypoints = self.state.keypoints.copy()
        landmarks = self.state.landmarks.copy()

        # Use PnP with RANSAC
        _, rvec, tvec, inliers_wrong_format = cv2.solvePnPRansac(landmarks, current_keypoints, self.camera_matrix, distCoeffs=None)
        inliers = np.zeros(len(current_keypoints), dtype=bool)
        if inliers_wrong_format is not None:
            inliers[inliers_wrong_format.flatten()] = True
        inliers = inliers.reshape(-1, 1)
        R, _ = cv2.Rodrigues(rvec)

        # Build 4x4 transformation matrix
        current_pose = np.eye(4)
        current_pose[:3, :3] = R
        current_pose[:3, 3] = tvec.ravel()

        # Remove outliers
        self.state.filter_keypoints(inliers)

        return np.linalg.inv(current_pose)

    def triangulate_new_landmarks(self, previous_image, current_image, current_pose):
        if self.state.candidates.shape[0] > 0:
            # Track existing candidate keypoints
            self.state.candidates, untracked_filter, _ = cv2.calcOpticalFlowPyrLK(previous_image, current_image, self.state.candidates, None)
            self.state.filter_candidates(untracked_filter)

            if self.check_new_landmarks:
                # Triangulate new landmarks
                new_keypoints = []
                new_landmarks = []
                candidates_to_be_removed = []
                for i in range(len(self.state.candidates)):
                    keypoint_1 = self.state.first_observations[i]
                    pose_1 = self.state.first_observation_poses[i]
                    keypoint_2 = self.state.candidates[i]
                    pose_2 = current_pose
                    # if self.is_valid_triangulation(keypoint_1, keypoint_2, pose_1, pose_2):
                    if self.simple_is_valid_triangulation(keypoint_1, keypoint_2):
                        landmark = self.triangulate_landmarks(np.expand_dims(keypoint_1, axis=0), np.expand_dims(keypoint_2, axis=0), pose_1, pose_2).squeeze()

                        if not self.landmark_is_behind_camera(landmark, pose_2):
                            new_keypoints.append(keypoint_2)
                            new_landmarks.append(landmark)
                            candidates_to_be_removed.append(i)

                if new_keypoints:
                    self.state.add_keypoints(np.array(new_keypoints), np.array(new_landmarks))
                    self.state.remove_candidates(candidates_to_be_removed)

        # Find new candidate keypoints
        max_corners = self.config.desired_candidates - len(self.state.candidates)
        if max_corners > 0:
            new_candidate_keypoints = cv2.goodFeaturesToTrack(current_image,
                                                              maxCorners=max_corners,
                                                              qualityLevel=self.config.quality_level,
                                                              minDistance=self.config.min_distance).reshape(-1, 2)  # shape: (K, 2)

            new_candidate_keypoints, _ = self.filter_keypoints_in_frame(new_candidate_keypoints, current_image.shape)

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
        cos_alpha = np.clip(np.dot(d1, d2), -1.0, 1.0)
        alpha = np.arccos(cos_alpha)

        # Return True if angle exceeds the threshold
        return alpha > self.config.threshold_triangulation_angle

    def simple_is_valid_triangulation(self, keypoint_1, keypoint_2):
        """
        A cheaper and simpler check for a valid triangulation, but doesn't account for real-world scale or depth.
        """
        distance = np.linalg.norm(np.array(keypoint_1) - np.array(keypoint_2))

        return distance > self.config.threshold_pixel_distance

    def triangulate_landmarks(self, keypoints_1, keypoints_2, pose_1, pose_2):
        """
        Triangulate a point cloud of 3D landmarks
        """
        projection_matrix_1 = self.camera_matrix @ np.linalg.inv(pose_1)[:3, :]
        projection_matrix_2 = self.camera_matrix @ np.linalg.inv(pose_2)[:3, :]

        landmarks_hom = cv2.triangulatePoints(projection_matrix_1, projection_matrix_2, keypoints_1.T, keypoints_2.T)  # shape: (N, 4, 4)
        landmarks = landmarks_hom[:3] / landmarks_hom[3]
        return landmarks.T

    @staticmethod
    def landmark_is_behind_camera(landmark, pose):
        landmark_hom = np.append(landmark, 1)
        landmark_cam = np.linalg.inv(pose) @ landmark_hom

        return landmark_cam[2] < 0

    @staticmethod
    def filter_keypoints_in_frame(keypoints, frame_shape):
        height, width = frame_shape[:2]
        mask = (keypoints[:, 0] >= 0) & (keypoints[:, 0] <= width) & \
               (keypoints[:, 1] >= 0) & (keypoints[:, 1] <= height)

        filtered_keypoints = keypoints[mask]

        return filtered_keypoints, mask.astype(np.uint8).reshape(-1, 1)
