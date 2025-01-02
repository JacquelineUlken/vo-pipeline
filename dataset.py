import os
import numpy as np
import cv2


class Dataset:
    def __init__(self, images_path, camera_matrix_path, poses_path=None):
        self.image_paths = self._get_image_paths(images_path)
        self.camera_matrix = self._get_camera_matrix(camera_matrix_path)
        if poses_path:
            self.ground_truth_poses = self._get_poses(poses_path)
        else:
            self.ground_truth_poses = None

    @staticmethod
    def _get_image_paths(images_path):
        return [os.path.join(images_path, img) for img in sorted(os.listdir(images_path)) if img.endswith('.png')]

    @staticmethod
    def _get_camera_matrix(camera_matrix_path):
        camera_matrix = np.genfromtxt(camera_matrix_path)[0, 1:].reshape((3, 4))[:, :3]
        assert camera_matrix.shape == (3, 3), "Invalid camera matrix shape"
        return camera_matrix

    @staticmethod
    def _get_poses(poses_path):
        raw_poses = np.loadtxt(poses_path).reshape(-1, 3, 4)
        homogenous_poses = np.tile(np.eye(4), (raw_poses.shape[0], 1, 1))
        homogenous_poses[:, :3, :] = raw_poses
        return homogenous_poses

    def get_frame(self, index):
        if index < 0 or index >= len(self.image_paths):
            raise IndexError(f"Frame index {index} out of bounds.")
        return cv2.imread(self.image_paths[index], cv2.IMREAD_GRAYSCALE)

    def get_pose(self, index):
        if self.ground_truth_poses is not None:
            return self.ground_truth_poses[index]
        else:
            raise ValueError("Ground truth poses are not available for this dataset.")
