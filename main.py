import argparse
import numpy as np
from config import Config
from dataset import Dataset
from pipeline import Pipeline
from visualize import Visualization

DATA_FOLDER = "data"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["parking", "kitti", "malaga"], help="Which dataset to use", default="malaga")
    args = parser.parse_args()

    dataset_to_use = args.dataset
    config = Config()

    config.use_simple_triangulation_validation = True

    if dataset_to_use == "kitti":
        folder = f"{DATA_FOLDER}/kitti"
        dataset = Dataset(dataset_to_use, folder)

    elif dataset_to_use == "parking":
        folder = f"{DATA_FOLDER}/parking"
        dataset = Dataset(dataset_to_use, folder)

    elif dataset_to_use == "malaga":
        folder = f"{DATA_FOLDER}/malaga-urban-dataset-extract-07"
        dataset = Dataset(dataset_to_use, folder)

    else:
        print(f"Unknown dataset {dataset_to_use}. Exciting...")
        exit()

    print(f"Starting VO Pipeline with {dataset_to_use} dataset.")

    visualization = Visualization(dataset)
    pipeline = Pipeline(dataset, config, visualization)
    poses = pipeline.run()

    flattened_poses = np.array([pose.flatten()[:12] for pose in poses])
    np.savetxt(f"final_poses/poses_{dataset_to_use}.txt", flattened_poses, delimiter=" ", fmt="%.6f")

    visualization.save_video(f"videos/{dataset_to_use}.mp4")

    print(f"Finished VO Pipeline")
