import argparse
from config import Config
from dataset import Dataset
from pipeline import Pipeline

DATA_FOLDER = "data"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["parking", "kitti", "malaga"], help="Which dataset to use", default="malaga")
    args = parser.parse_args()

    dataset_to_use = args.dataset

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

    print(f"Finished VO Pipeline")
