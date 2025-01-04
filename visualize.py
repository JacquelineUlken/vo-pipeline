import matplotlib.pyplot as plt
import numpy as np

def VisualizeKeypoints(img, keypoints_1, keypoints_2, fig, axs, camera_pose):
    """
    Visualize the keypoints and camera poses
    """
    axs[0].cla()
    axs[1].cla()

    axs[1].set_xlim([-200, 200])
    axs[1].set_ylim([-100, 100])

    axs[0].imshow(img, cmap='gray')
    keypoints_1 = keypoints_1.reshape(-1, 2)

    axs[0].plot(keypoints_1[:, 0], keypoints_1[:, 1], 'rx', linewidth = 2, label = 'keypoints_1')

    keypoints_2 = keypoints_2.reshape(-1, 2)

    keypoints_2_filter = []
    for j in range(keypoints_2.shape[0]):
        if keypoints_2[j, 0] >=0 and keypoints_2[j, 0] <= img.shape[1] and keypoints_2[j, 1] >=0 and keypoints_2[j, 1] <= img.shape[0]:
            keypoints_2_filter.append(keypoints_2[j, :])

    keypoints_2_filter = np.array(keypoints_2_filter)

    axs[0].plot(keypoints_2_filter[:, 0], keypoints_2_filter[:, 1], 'o', color='green', markerfacecolor='none', linewidth=2, label = 'keypoints_2') 

    axs[0].legend(loc='upper right')

    # print(camera_pose[0])
    # print(camera_pose[1])
    axs[1].plot(camera_pose[0], camera_pose[1], 'bo', linewidth = 2, label = 'camera_poses')
    axs[1].legend(loc='upper right')

    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
    axs[1].set_title("Camera Poses")

    plt.tight_layout()

    plt.pause(5.0)
    # plt.show()

    return