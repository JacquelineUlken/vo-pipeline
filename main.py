from dataset import Dataset
from pipeline import Pipeline
import cv2
import matplotlib.pyplot as plt
import numpy as np

from visualize import VisualizeKeypoints

if __name__ == "__main__":
    print(f"Starting VO Pipeline")
    images_path_kitti = "data/kitti/05/image_0/"
    images_path_parking = "data/parking/images/img_"
    # camera_matrix_path = "data/kitti/05/calib.txt"
    poses_path = "data/kitti/poses/05.txt"
    # dataset = Dataset(images_path, camera_matrix_path, poses_path)
    # pipeline = Pipeline(dataset)

    # pipeline.initialize(0, 2)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # camera_pose = np.zeros((2, 3))
    # camera_pose = np.array([[1.5, 1.9, 0], [2.5, 2.9, 0]])
    # print(camera_pose.shape)
    # print(camera_pose[0])

    camera_trans = np.empty((0, 3)) # Nx3
    with open(poses_path, "r") as file:
        for line in file:
            row = [float(value) for value in line.split()]
            # print(row[0])
            trans_vector = np.array([row[3], row[7], row[11]])
            # print(trans_vector)
            camera_trans = np.vstack((camera_trans, trans_vector))

    # print(camera_trans[2760, :])

    for i in range(2):
        # for kitti
        img = cv2.imread(images_path_kitti + '{0:06d}.png'.format(i), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(images_path_kitti + '{0:06d}.png'.format(i+5), cv2.IMREAD_GRAYSCALE)
        
        # for parking
        # img = cv2.imread(images_path_parking + '{0:05d}.png'.format(i), cv2.IMREAD_GRAYSCALE)
        # img2 = cv2.imread(images_path_parking + '{0:05d}.png'.format(i+5), cv2.IMREAD_GRAYSCALE)
        
        keypoints_1 = cv2.goodFeaturesToTrack(img, maxCorners=500, qualityLevel=0.01, minDistance=7)
        keypoints_2, status, _ = cv2.calcOpticalFlowPyrLK(img, img2, keypoints_1, None)

        VisualizeKeypoints(img, keypoints_1, keypoints_2, fig, axs, camera_trans[i, :])