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

    def update(self, frame_idx, state: State, poses):
        """
        Update the visualization with the current state and poses.
        """
        # Displaying the image with the keypoints
        ax1 = self.axs[0, 0]
        ax1.cla()

        ax1.set_title(f"Current Frame: {frame_idx}")
        ax1.set_xlim(0, self.dataset.dimensions[1])
        ax1.set_ylim(self.dataset.dimensions[0], 0)

        image = self.dataset.get_frame(frame_idx, color=True)
        ax1.imshow(image)

        keypoints = state.keypoints
        candidates = state.candidate_keypoints

        ax1.plot(candidates[:, 0], candidates[:, 1], 'x', color='green', markerfacecolor='none', linewidth=1, label="all candidates")
        ax1.plot(keypoints[:, 0], keypoints[:, 1], 'o', color='red', markerfacecolor='none', linewidth=1, label="all keypoints")
        ax1.legend(loc='upper right')

        # Displaying local trajectory and landmarks
        ax2 = self.axs[0, 1]
        ax2.cla()
        ax2.set_title("Local trajectory and landmarks")
        ax2.set_xlabel('x')
        ax2.set_ylabel('z')
        ax2.set_aspect('equal')

        center = poses[-1, :, 3]

        landmarks = state.landmarks

        last_poses = poses[-20:] if len(poses) > 20 else poses
        ax2.plot(landmarks[:, 0], landmarks[:, 2], 'o', color='red', markerfacecolor='none', linewidth=1, label="landmarks")
        ax2.plot(last_poses[:, 0, 3], last_poses[:, 2, 3], color='blue', marker='.')

        xlim = ax2.get_xlim()
        ylim = ax2.get_ylim()

        max_range = min(max(xlim[1] - xlim[0], ylim[1] - ylim[0]), 500)

        ax2.set_xlim(center[0] - max_range / 2, center[0] + max_range / 2)
        ax2.set_ylim(center[2] - max_range / 2, center[2] + max_range / 2)

        # Displaying number of keypoints and candidate keypoints
        ax3 = self.axs[1, 0]
        ax3.cla()
        ax3.set_title("Number of landmarks and candidate keypoints")
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Number of points')
        ax3.grid(True)

        ax3.set_xlim(0, len(self.dataset))
        ax3.set_ylim(0, 500)

        ax3.plot(self.number_of_landmarks, color='red', label="Number of landmarks")
        ax3.plot(self.number_of_candidates, color='green', label="Number of candidates")

        # Displaying global trajectory
        ax4 = self.axs[1, 1]
        ax4.cla()
        ax4.set_title("Global trajectory")
        ax4.set_xlabel('x')
        ax4.set_ylabel('z')
        ax4.set_aspect('equal')
        ax4.plot(poses[:, 0, 3], poses[:, 2, 3], color='blue')

        xlim = ax4.get_xlim()
        ylim = ax4.get_ylim()

        max_range = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
        x_center = sum(xlim) / 2
        y_center = sum(ylim) / 2

        ax4.set_xlim([x_center - max_range / 2, x_center + max_range / 2])
        ax4.set_ylim([y_center - max_range / 2, y_center + max_range / 2])

        plt.tight_layout()
        plt.pause(0.01)

        # Update the number of landmarks and candidates
        self.number_of_landmarks.append(len(state.landmarks))
        self.number_of_candidates.append(len(state.candidate_keypoints))

        self.fig.canvas.draw()
        frame = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        self.frames.append(frame)

    def save_video(self, filename='output.avi', fps=10):
        """
        Save the collected frames as a video.
        """
        height, width, _ = self.frames[0].shape
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
        for frame in self.frames:
            out.write(frame)
        out.release()