import matplotlib.pyplot as plt
import numpy as np

def VisualizeKeypoints(img, axs, camera_pose, keypoints, poses_list, landmarks):
    """
    Visualize the keypoints and camera poses
    """
    axs[0, 0].cla()
    axs[0, 1].cla()
    axs[1, 1].cla()

    # axs[0, 1].set_xlim([-300, 300])
    # axs[0, 1].set_ylim([-300, 300])

    axs[0, 0].imshow(img, cmap='gray')

    keypoints_filter = []
    for j in range(keypoints.shape[0]):
        if keypoints[j, 0] >=0 and keypoints[j, 0] <= img.shape[1] and keypoints[j, 1] >=0 and keypoints[j, 1] <= img.shape[0]:
            keypoints_filter.append(keypoints[j, :])

    keypoints_filter = np.array(keypoints_filter)

    axs[0, 0].plot(keypoints_filter[:, 0], keypoints_filter[:, 1], 'o', color='green', markerfacecolor='none', linewidth=2, label = 'keypoints') 
    axs[0, 0].legend(loc='upper right')

    for k in range(len(poses_list)):
        axs[0, 1].plot(poses_list[k][0, 3], poses_list[k][1, 3], 'bo', linewidth = 2, label = 'camera_poses')

    axs[0, 1].set_xlabel('x')
    axs[0, 1].set_ylabel('y')
    axs[0, 1].set_title("Camera Poses")

    axs[1, 1].plot(landmarks[:, 0], landmarks[:, 2]) 
    axs[1, 1].set_xlabel('x')
    axs[1, 1].set_ylabel('z')
    axs[1, 1].set_title("Landmarks")

    axs[1, 0].axis("off")

    plt.tight_layout()

    plt.pause(0.01)

    return