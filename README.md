# A Visual Odometry Pipeline
A simple, monocular, visual odometry (VO) pipeline with the most essential features:
- **Initialization of 3D landmarks**  
- **Keypoint tracking** between consecutive frames  
- **Pose estimation** using 2D-3D correspondences  
- **Triangulation of new landmarks**  

The pipeline is designed to work with datasets like KITTI (05) and Malaga and can be extended to other datasets with minor adjustments.

---

## Datasets
The pipeline has been tested with the following datasets:

- **KITTI Odometry Dataset (Sequence 05)**
- **Malaga Stereo Urban Dataset**
- **Parking Dataset**

Which are all available on the course website [here](https://rpg.ifi.uzh.ch/teaching.html).

To use custom datasets, adjust the dataset loader in `dataset.py` to match your data structure.

---

## Installation
To set up the environment and install dependencies, follow the steps below:

### 1. Clone the Repository
```bash
git clone https://github.com/username/vo-pipeline.git
cd vo-pipeline
```
### 2. Set Up the Environment
An `environment.yml` file is provided to create the required Python environment using Conda.
Run the following command to create the environment:
```bash
conda env create -f environment.yml
```
Activate the environment:
```bash
conda activate vo-pipeline
```

---

## How to run
1. Extract the dataset (e.g., kitti05.zip) into the data folder. The folder structure should look like this:
```kotlin
data/
└── kitti/
    ├── 05
    ├── poses
```
2. Use the following command to execute the pipeline, specifying the dataset to use:
```bash
python main.py --dataset name
```
- `--dataset` – Choose between three datasets:
  - `parking` – Parking lot dataset
  - `kitti` – KITTI odometry dataset
  - `malaga` – Malaga urban dataset

Example for running with the KITTI dataset:
```bash
python main.py --dataset kitti
```

If no dataset is specified, the **Parking** dataset will be used by default:
```bash
python main.py
```

3. Poses are saved in the `final_poses/` directory as `poses_[dataset].txt`.
Visualization videos of the results are saved in the `videos/` directory, e.g., `videos/kitti.mp4`.

---

## Visualization
A demo of the pipeline working with the three datasets can be found [here](https://www.youtube.com/watch?v=jwkrQaLIJNo&list=PLsmaGYpKB9GvD1bOp2xj7y9h44MYgsv36).
To create these visualizations, the visual odometry pipeline was run on a machine equipped with an Intel processor running at 4.2 GHz, 16 GB of RAM, and a multiprocessor configuration supporting multiple threads.

![Pipeline Visualization](pictures/kitti_frame_1400.jpg)