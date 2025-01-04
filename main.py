from dataset import Dataset
from pipeline import Pipeline
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

from visualize import VisualizeKeypoints

if __name__ == "__main__":
    print(f"Starting VO Pipeline")
    images_path_kitti = "data/kitti/05/image_0/"
    images_path_parking = "data/parking/images/img_"
    # camera_matrix_path = "data/kitti/05/calib.txt"
    poses_path_kitti = "data/kitti/poses/05.txt"
    poses_path_parking = "data/parking/poses.txt"
    # dataset = Dataset(images_path, camera_matrix_path, poses_path_kitti)
    # pipeline = Pipeline(dataset)

    malaga_images = "data/malaga-urban-dataset-extract-07/Images/"

    filenames = sorted(os.listdir(malaga_images))

    # left camera images
    malaga_left_images = []
    for f in filenames:
        if 'left' in f:
            malaga_left_images.append(f)

    # print(malaga_left_images[:5])

    # pipeline.initialize(0, 2)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # camera_pose = np.zeros((2, 3))
    # camera_pose = np.array([[1.5, 1.9, 0], [2.5, 2.9, 0]])
    # print(camera_pose.shape)
    # print(camera_pose[0])

    malaga_gps_poses_path = "data/malaga-urban-dataset-extract-07/malaga-urban-dataset-extract-07_all-sensors_GPS.txt"

    gps_poses_list = np.empty((0, 4))
    with open(malaga_gps_poses_path, "r") as file:
        for pose in file:
            if pose.startswith("%"):
                continue
            poses = pose.split()  
            gps_poses = [float(part) for part in poses]
            gps_poses = np.array((gps_poses[0], gps_poses[8], gps_poses[9], gps_poses[10]))
            gps_poses_list = np.vstack((gps_poses_list, gps_poses))

    # print(gps_poses_list.shape)

    # print(gps_poses_list)
            
    camera_trans = np.empty((0, 3)) # Nx3
    camera_rot = np.empty((0, 3, 3))
    yaw_arr = np.empty((0, 1))
    # with open(poses_path_kitti, "r") as file:
    with open(poses_path_parking, "r") as file:
        for line in file:
            row = [float(value) for value in line.split()]
            # print(row[0])
            trans_vector = np.array([row[3], row[7], row[11]])
            # print(trans_vector)
            camera_trans = np.vstack((camera_trans, trans_vector))

            rot_mat = np.array([[row[0], row[1], row[2]], [row[4], row[5], row[6]], [row[8], row[9], row[10]]])
            # print(rot_mat)
            yaw = np.arctan2(rot_mat[1, 0], rot_mat[0, 0]) # [-pi, pi], considering negligible roll and pitch
            if yaw < 0:
                yaw += 2 * np.pi

            yaw_deg = yaw * 180 / np.pi
            yaw_arr = np.vstack((yaw_arr, yaw_deg))

    # print(yaw_arr)

    # print(camera_trans[2760, :])

    for i in range(gps_poses_list.shape[0]):
        # # for kitti
        # img = cv2.imread(images_path_kitti + '{0:06d}.png'.format(i), cv2.IMREAD_GRAYSCALE)
        # img2 = cv2.imread(images_path_kitti + '{0:06d}.png'.format(i+5), cv2.IMREAD_GRAYSCALE)
        
        # # for parking
        # img = cv2.imread(images_path_parking + '{0:05d}.png'.format(i), cv2.IMREAD_GRAYSCALE)
        # img2 = cv2.imread(images_path_parking + '{0:05d}.png'.format(i+5), cv2.IMREAD_GRAYSCALE)

        # # for malaga
        img = cv2.imread(malaga_images + malaga_left_images[i], cv2.IMREAD_GRAYSCALE)
        # print(img)
        
        time_malaga_img = float(malaga_left_images[i][12:28])
        # print(time_malaga_img)

        # hack for getting the poses of the images 
        # later interpolte with diff drive geometry and yaw values from the IMU
        #       v     = ( vr + vl ) / 2.0;
        # omega = ( vr - vl ) / m_wheel_base;

        # delta_s     = v     * dt;
        # delta_theta = omega * dt;

        # x     += delta_s * std::cos ( theta + delta_theta/2.0 );
        # y     += delta_s * std::sin ( theta + delta_theta/2.0 );
        # theta += delta_theta;
        # m_ts_prev = ts;
        for j in range(gps_poses_list.shape[0]):
            gps_timestamp = gps_poses_list[j, 0] 

            time_diff = abs(gps_timestamp - time_malaga_img)

            if time_diff < 0.5:
                corresponding_pose = gps_poses_list[j]
                print(corresponding_pose)

        img2 = cv2.imread(malaga_images + malaga_left_images[i + 1], cv2.IMREAD_GRAYSCALE)
        # print(img2)
        
        keypoints_1 = cv2.goodFeaturesToTrack(img, maxCorners=500, qualityLevel=0.01, minDistance=7)
        keypoints_2, status, _ = cv2.calcOpticalFlowPyrLK(img, img2, keypoints_1, None)

        VisualizeKeypoints(img, keypoints_1, keypoints_2, fig, axs, camera_trans[i, :])