import os
import numpy as np
import cv2


class Dataset:
    def __init__(self, images_path, camera_matrix_path, poses_path):
        self.image_paths = self._get_image_paths(images_path)
        self.camera_matrix = self._get_camera_matrix(camera_matrix_path)
        self.poses = self._get_poses(poses_path)

    @staticmethod
    def _get_image_paths(images_path):
        return [os.path.join(images_path, img) for img in sorted(os.listdir(images_path)) if img.endswith('.png')]

    @staticmethod
    def _get_camera_matrix(camera_matrix_path):
        return np.genfromtxt(camera_matrix_path)[0, 1:].reshape((3, 4))[:, :3]

    @staticmethod
    def _get_poses(poses_path):
        raw_poses = np.loadtxt(poses_path).reshape(-1, 3, 4)
        homogenous_poses = np.zeros((raw_poses.shape[0], 4, 4))
        homogenous_poses[:, :3, :] = raw_poses
        homogenous_poses[:, 3, 3] = 1
        return homogenous_poses

    def get_frame(self, index):
        image = cv2.imread(self.image_paths[index], cv2.IMREAD_GRAYSCALE)
        pose = self.poses[index]
        return image, pose
