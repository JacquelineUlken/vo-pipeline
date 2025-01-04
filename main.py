from config import Config
from dataset import Dataset
from pipeline import Pipeline

DATA_FOLDER = "data"

if __name__ == "__main__":
    dataset_to_use = "parking"

    if dataset_to_use == "kitti":
        folder = f"{DATA_FOLDER}/kitti"

        config = Config()
        dataset = Dataset(dataset_to_use, folder)

    elif dataset_to_use == "parking":
        folder = f"{DATA_FOLDER}/parking"

        config = Config()
        dataset = Dataset(dataset_to_use, folder)

    elif dataset_to_use == "malaga":
        folder = f"{DATA_FOLDER}/malaga-urban-dataset-extract-07"

        config = Config()
        dataset = Dataset(dataset_to_use, folder)

    else:
        print(f"Unknown dataset {dataset_to_use}. Exciting...")
        exit()

    print(f"Starting VO Pipeline with {dataset_to_use} dataset.")

    pipeline = Pipeline(dataset, config)
    poses = pipeline.run()

    print(f"Finished VO Pipeline, got {len(poses)} poses, each in the form of a {poses.shape[1]} x {poses.shape[2]} matrix.")
    print("State Dimensions:")
    print(f"keypoints: {pipeline.state.keypoints.shape}")
    print(f"landmarks: {pipeline.state.landmarks.shape}")
    print(f"candidate_keypoints: {pipeline.state.candidate_keypoints.shape}")
    print(f"first_observations: {pipeline.state.first_observations.shape}")
    print(f"first_observation_poses: {pipeline.state.first_observation_poses.shape}")
