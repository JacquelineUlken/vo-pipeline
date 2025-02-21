import numpy as np


class State:
    def __init__(self, keypoints, landmarks, candidates, first_observations, first_observation_poses):
        self.keypoints = keypoints  # numpy array of shape (N, 2)
        self.landmarks = landmarks  # numpy array of shape (N, 3)
        self.candidates = candidates  # numpy array of shape (M, 2)
        self.first_observations = first_observations  # numpy array of shape (M, 2)
        self.first_observation_poses = first_observation_poses  # numpy array of shape (M, 4, 4)

    @classmethod
    def empty(cls):
        return cls(np.array([], dtype=np.float32).reshape(0, 2),
                   np.array([], dtype=np.float32).reshape(0, 3),
                   np.array([], dtype=np.float32).reshape(0, 2),
                   np.array([], dtype=np.float32).reshape(0, 2),
                   np.array([], dtype=np.float32).reshape(0, 4, 4))

    def filter_keypoints(self, filter_matrix):
        if not len(self.keypoints) == 0:
            self.keypoints = np.expand_dims(self.keypoints, axis=1)[filter_matrix == 1]
            self.landmarks = np.expand_dims(self.landmarks, axis=1)[filter_matrix == 1]

    def add_keypoints(self, keypoints, landmarks):
        self.keypoints = np.concatenate((self.keypoints, keypoints), axis=0)
        self.landmarks = np.concatenate((self.landmarks, landmarks), axis=0)

    def filter_candidates(self, filter_matrix):
        if not len(self.candidates) == 0:
            self.candidates = np.expand_dims(self.candidates, axis=1)[filter_matrix == 1]
            self.first_observations = np.expand_dims(self.first_observations, axis=1)[filter_matrix == 1]
            self.first_observation_poses = np.expand_dims(self.first_observation_poses, axis=1)[filter_matrix == 1]

    def add_candidate_keypoints(self, keypoints, pose):
        if not len(self.candidates) == 0:
            distances = np.linalg.norm(keypoints[:, np.newaxis, :] - self.candidates[np.newaxis, :, :], axis=2)
            min_distances = np.min(distances, axis=1)
            redundant_filter = min_distances > 5
            keypoints = keypoints[redundant_filter]

        self.candidates = np.concatenate((self.candidates, keypoints), axis=0)
        self.first_observations = np.concatenate((self.first_observations, keypoints), axis=0)
        poses = np.tile(pose, (len(keypoints), 1, 1))
        self.first_observation_poses = np.concatenate((self.first_observation_poses, poses))

    def remove_candidates(self, indices):
        # Sort indices in descending order to avoid shifting issues during deletion
        indices = np.sort(indices)[::-1]

        for i in indices:
            self.candidates = np.delete(self.candidates, i, axis=0)
            self.first_observations = np.delete(self.first_observations, i, axis=0)
            self.first_observation_poses = np.delete(self.first_observation_poses, i, axis=0)
