from config import Config
from dataset import Dataset
from pipeline import Pipeline

if __name__ == "__main__":
    print(f"Starting VO Pipeline")
    images_path = "data/kitti/05/image_0"
    camera_matrix_path = "data/kitti/05/calib.txt"
    poses_path = f"data/kitti/poses/05.txt"
    dataset = Dataset(images_path, camera_matrix_path, poses_path)
    config = Config()
    pipeline = Pipeline(dataset, config)

    poses = pipeline.run()
    print(f"Finished VO Pipeline, got {len(poses)} poses, each in the form of a {poses.shape[1]} x {poses.shape[2]} matrix.")
    print("State Dimensions:")
    print(f"keypoints: {pipeline.state.keypoints.shape}")
    print(f"landmarks: {pipeline.state.landmarks.shape}")
    print(f"candidate_keypoints: {pipeline.state.candidate_keypoints.shape}")
    print(f"first_observations: {pipeline.state.first_observations.shape}")
    print(f"first_observation_poses: {pipeline.state.first_observation_poses.shape}")