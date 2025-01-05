import matplotlib.pyplot as plt
import numpy as np


def VisualizeKeypoints(img, axs, camera_pose, keypoints, poses_list, landmarks, candidate_keypoints, frame_id):
    """
    Visualize the keypoints and camera poses
    """
    axs[0].cla()
    axs[1].cla()

    axs[0].imshow(img, cmap='gray')
    axs[0].set_title("Current Frame No." + str(frame_id))

    keypoints_filter = []
    for j in range(keypoints.shape[0]):
        if keypoints[j, 0] >= 0 and keypoints[j, 0] <= img.shape[1] and keypoints[j, 1] >= 0 and keypoints[j, 1] <= img.shape[0]:
            keypoints_filter.append(keypoints[j, :])

    keypoints_filter = np.array(keypoints_filter)

    axs[0].plot(keypoints_filter[:, 0], keypoints_filter[:, 1], 'o', color='red', markerfacecolor='none', linewidth=1, label="all keypoints")

    candidate_keypoints_filter = []
    for j in range(candidate_keypoints.shape[0]):
        if candidate_keypoints[j, 0] >= 0 and candidate_keypoints[j, 0] <= img.shape[1] and candidate_keypoints[j, 1] >= 0 and candidate_keypoints[j, 1] <= img.shape[0]:
            candidate_keypoints_filter.append(candidate_keypoints[j, :])

    candidate_keypoints_filter = np.array(candidate_keypoints_filter)

    axs[0].plot(candidate_keypoints_filter[:, 0], candidate_keypoints_filter[:, 1], 'x', color='green', linewidth=1, label="candidate keypoints")

    axs[0].legend(loc='upper right')

    axs[1].plot(landmarks[:, 0], landmarks[:, 2], 'o', linewidth=1, color='red', label="Landmarks")

    flag = True
    for k in range(len(poses_list)):
        axs[1].plot(poses_list[k][0, 3], poses_list[k][1, 3], 'bx', linewidth=1, label='Camera poses')
        if flag:
            axs[1].legend(loc='upper right')
            flag = False

    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y (z)')
    axs[1].set_title("Camera Poses and Landmarks")

    plt.tight_layout()

    plt.pause(0.01)

    return