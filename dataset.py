import os
import numpy as np
import cv2


class Dataset:
    """
    Handles loading images, camera intrinsics, and ground truth poses (if available) from KITTI, Malaga, or Parking datasets.

    Attributes:
        camera_matrix (np.array): Camera intrinsic matrix (3x3).
        ground_truth_poses (np.array): Ground truth poses (optional, for KITTI or Parking).

    Methods:
        get_frame(index):
            Loads and returns the grayscale image at the specified index.

        get_ground_truth_pose(index):
            Retrieves the ground truth pose at the specified index.
    """

    def __init__(self, dataset_type, base_path):
        assert dataset_type in ["kitti", "malaga", "parking"], "Invalid dataset type"

        self.dataset_type = dataset_type
        self.base_path = base_path

        if dataset_type == "kitti":
            self._setup_kitti()
        elif dataset_type == "malaga":
            self._setup_malaga()
        elif dataset_type == "parking":
            self._setup_parking()

    def _setup_kitti(self):
        self.image_paths = self._get_image_paths(os.path.join(self.base_path, '05/image_0'))
        self.camera_matrix = np.array([[718.856, 0, 607.1928],
                                       [0, 718.856, 185.2157],
                                       [0, 0, 1]])
        self.ground_truth_poses = self._get_poses(os.path.join(self.base_path, 'poses/05.txt'))

    def _setup_malaga(self):
        image_dir = os.path.join(self.base_path, 'malaga-urban-dataset-extract-07_rectified_800x600_Images')
        self.image_paths = self._get_image_paths(image_dir, malaga=True)
        self.camera_matrix = np.array([[621.18428, 0, 404.0076],
                                       [0, 621.18428, 309.05989],
                                       [0, 0, 1]])
        self.ground_truth_poses = None

    def _setup_parking(self):
        self.image_paths = self._get_image_paths(os.path.join(self.base_path, 'images'))
        self.camera_matrix = np.array([[331.37, 0, 320],
                                       [0, 369.568, 240],
                                       [0, 0, 1]])
        self.ground_truth_poses = self._get_poses(os.path.join(self.base_path, 'poses.txt'))

    @staticmethod
    def _get_image_paths(images_path, malaga=False):
        """
        Retrieve sorted image paths.
        For Malaga, skip the first two entries which are '.' and '..' and use only the left images (odd indices)
        """
        image_files = sorted(os.listdir(images_path))
        if malaga:
            image_files = image_files[2::2]

        return [os.path.join(images_path, img) for img in image_files if img.endswith('.png') or img.endswith('.jpg')]

    @staticmethod
    def _get_poses(poses_path):
        raw_poses = np.loadtxt(poses_path).reshape(-1, 3, 4)
        homogeneous_poses = np.tile(np.eye(4), (raw_poses.shape[0], 1, 1))
        homogeneous_poses[:, :3, :] = raw_poses
        return homogeneous_poses

    def get_frame(self, index):
        """
        Loads a grayscale image at the specified index.
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"Frame index {index} out of bounds.")
        return cv2.imread(self.image_paths[index], cv2.IMREAD_GRAYSCALE)

    def get_ground_truth_pose(self, index):
        """
        Retrieves the ground truth pose at the specified index (if available).
        """
        if self.ground_truth_poses is not None:
            return self.ground_truth_poses[index]
        else:
            raise ValueError("Ground truth poses are not available for this dataset.")

    def __len__(self):
        return len(self.image_paths)
