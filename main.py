from dataset import Dataset
from pipeline import Pipeline

if __name__ == "__main__":
    print(f"Starting VO Pipeline")
    images_path = "data/kitti/05/image_0"
    camera_matrix_path = "data/kitti/05/calib.txt"
    poses_path = f"data/kitti/poses/05.txt"
    dataset = Dataset(images_path, camera_matrix_path, poses_path)
    pipeline = Pipeline(dataset)

    initial_landmarks, initial_pose, next_pose = pipeline.initialize(1, 3)