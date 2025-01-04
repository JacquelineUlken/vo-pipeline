import matplotlib.pyplot as plt
import numpy as np

def VisualizeKeypoints(img, axs, camera_pose):
    """
    Visualize the keypoints and camera poses
    """
    axs[0].cla()
    axs[1].cla()

    axs[1].set_xlim([-5, 200])
    axs[1].set_ylim([-5, 5])

    axs[0].imshow(img, cmap='gray')

    axs[1].plot(camera_pose[0, 3], camera_pose[1, 3], 'bo', linewidth = 2, label = 'camera_poses')
    axs[1].legend(loc='upper right')

    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
    axs[1].set_title("Camera Poses")

    plt.tight_layout()

    plt.pause(0.01)

    return