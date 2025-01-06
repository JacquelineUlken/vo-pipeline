import matplotlib.pyplot as plt
from dataset import Dataset
from state import State
import numpy as np
import cv2


class Visualization:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.fig, self.axs = plt.subplots(2, 2, figsize=(10, 10))
        self.number_of_landmarks = []
        self.number_of_candidates = []
        self.frames = []

    def plot_current_frame(self, frame_idx, state: State):
        ax1 = self.axs[0, 0]
        ax1.cla()

        ax1.set_title(f"Current Frame: {frame_idx}")
        ax1.set_xlim(0, self.dataset.dimensions[1])
        ax1.set_ylim(self.dataset.dimensions[0], 0)

        image = self.dataset.get_frame(frame_idx, color=True)
        ax1.imshow(image)

        keypoints = state.keypoints
        candidates = state.candidates

        ax1.plot(candidates[:, 0], candidates[:, 1], "x", color="green", markerfacecolor="none", alpha=0.8, linewidth=1, label="all candidates")
        ax1.plot(keypoints[:, 0], keypoints[:, 1], "o", color="red", markerfacecolor="none", alpha=0.8, linewidth=1, label="all keypoints")
        ax1.legend(loc="lower right")

    def plot_trajectory_and_landmarks(self, state: State, poses):
        ax2 = self.axs[0, 1]
        ax2.cla()
        ax2.set_title("Trajectory of the last 20 frames and landmarks")
        ax2.set_xlabel("x")
        ax2.set_ylabel("z")
        ax2.set_aspect("equal")

        center = poses[-1, :, 3]
        landmarks = state.landmarks
        last_poses = poses[-20:] if len(poses) > 20 else poses

        ax2.plot(landmarks[:, 0], landmarks[:, 2], "o", color="red", markerfacecolor="none", linewidth=1, label="Landmarks")
        ax2.plot(last_poses[:, 0, 3], last_poses[:, 2, 3], color="blue", marker=".", label="Trajectory")

        if landmarks.shape[0] >= 200:
            distances = np.linalg.norm(landmarks[:, :3] - center[:3], axis=1)
            sorted_indices = np.argsort(distances)
            visible_landmarks = landmarks[sorted_indices[:200]]
        else:
            visible_landmarks = landmarks

        x_min = min(np.min(visible_landmarks[:, 0]), np.min(last_poses[:, 0, 3]))
        x_max = max(np.max(visible_landmarks[:, 0]), np.max(last_poses[:, 0, 3]))
        z_min = min(np.min(visible_landmarks[:, 2]), np.min(last_poses[:, 2, 3]))
        z_max = max(np.max(visible_landmarks[:, 2]), np.max(last_poses[:, 2, 3]))

        x_range = x_max - x_min
        z_range = z_max - z_min
        max_range = max(x_range, z_range)

        ax2.set_xlim(center[0] - max_range / 2, center[0] + max_range / 2)
        ax2.set_ylim(center[2] - max_range / 2, center[2] + max_range / 2)
        ax2.legend(loc="lower right")

    def plot_landmark_counts(self):
        ax3 = self.axs[1, 0]
        ax3.cla()
        ax3.set_title("Number of landmarks and candidate keypoints")
        ax3.set_xlabel("Frame")
        ax3.set_ylabel("Number of points")
        ax3.grid(True)

        ax3.set_xlim(0, len(self.dataset))
        ax3.set_ylim(0, 600)

        ax3.plot(self.number_of_landmarks, color="red", linewidth=1, alpha=0.8, label="Number of landmarks")
        ax3.plot(self.number_of_candidates, color="green", linewidth=1, alpha=0.8, label="Number of candidates")
        ax3.legend(loc="lower right")

    def plot_global_trajectory(self, poses):
        ax4 = self.axs[1, 1]
        ax4.cla()
        ax4.set_title("Global trajectory")
        ax4.set_xlabel("x")
        ax4.set_ylabel("z")
        ax4.set_aspect("equal")
        ax4.plot(poses[:, 0, 3], poses[:, 2, 3], color="blue", label="Trajectory")

        xlim = ax4.get_xlim()
        ylim = ax4.get_ylim()

        max_range = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
        x_center = sum(xlim) / 2
        y_center = sum(ylim) / 2

        ax4.set_xlim([x_center - max_range / 2, x_center + max_range / 2])
        ax4.set_ylim([y_center - max_range / 2, y_center + max_range / 2])
        ax4.legend(loc="lower right")

    def update(self, frame_idx, state: State, poses):
        self.plot_current_frame(frame_idx, state)
        self.plot_trajectory_and_landmarks(state, poses)
        self.plot_landmark_counts()
        self.plot_global_trajectory(poses)

        plt.tight_layout()
        plt.pause(0.01)

        self.number_of_landmarks.append(len(state.landmarks))
        self.number_of_candidates.append(len(state.candidates))

        self.fig.canvas.draw()
        frame = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        self.frames.append(frame)

    def save_video(self, filename="videos/output.mp4", fps=10):
        """
        Save the collected frames as an MP4 video with corrected colors.
        """
        height, width, _ = self.frames[0].shape

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4 codec

        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

        for frame in self.frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()
