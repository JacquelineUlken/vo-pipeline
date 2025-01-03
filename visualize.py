import matplotlib.pyplot as plt
import numpy as np

def VisualizeKeypoints(img, keypoints_1, keypoints_2):
    """
    Visualize the keypoints
    """

    plt.clf()
    plt.imshow(img, cmap='gray')
    keypoints_1 = keypoints_1.reshape(-1, 2)

    plt.plot(keypoints_1[:, 0], keypoints_1[:, 1], 'rx', linewidth = 2, label = 'keypoints_1')

    keypoints_2 = keypoints_2.reshape(-1, 2)

    keypoints_2_filter = []
    for j in range(keypoints_2.shape[0]):
        if keypoints_2[j, 0] >=0 and keypoints_2[j, 0] <= img.shape[1] and keypoints_2[j, 1] >=0 and keypoints_2[j, 1] <= img.shape[0]:
            keypoints_2_filter.append(keypoints_2[j, :])

    keypoints_2_filter = np.array(keypoints_2_filter)

    plt.plot(keypoints_2_filter[:, 0], keypoints_2_filter[:, 1], 'o', color='green', markerfacecolor='none', linewidth=2, label = 'keypoints_2') 

    plt.legend(loc='upper right')

    plt.pause(1.0)
    plt.show()

    return