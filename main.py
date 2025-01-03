from dataset import Dataset
from pipeline import Pipeline

if __name__ == "__main__":
    print(f"Starting VO Pipeline")
    images_path = "data/kitti/05/image_0"
    camera_matrix_path = "data/kitti/05/calib.txt"
    poses_path = f"data/kitti/poses/05.txt"
    dataset = Dataset(images_path, camera_matrix_path, poses_path)
    pipeline = Pipeline(dataset)

    poses = pipeline.run()
    print(f"Finished VO Pipeline, got {len(poses)} poses, each in the form of a {poses.shape[1]} x {poses.shape[2]} matrix.")