from dataclasses import dataclass


@dataclass
class Config:
    # INITIALIZATION
    init_frame_2_index: int = 2  # Typical between 1 and 10
    init_max_corners: int = 400  # Typical between 100 and 1000
    init_quality_level: float = 0.05  # Typical between 0.01 and 0.1
    init_min_distance: int = 7  # Typical between 5 and 10

    # CONTINUOUS OPERATION
    # When to check for new landmarks
    min_landmarks: int = 300
    # Finding new candidate keypoints
    desired_candidates: int = 600  # Typical between 100 and 1000
    quality_level: float = 0.01  # Typical between 0.01 and 0.1
    min_distance: int = 7  # Typical between 5 and 10
    # Triangulating new landmarks
    use_simple_triangulation_validation: bool = False
    threshold_triangulation_angle: float = 1 / 36 * 3.14  # Typical between 1/36 pi and 5/36 pi
    threshold_pixel_distance: float = 5
    # ransac
    ransac_prob: float = 0.99  # Typical between 0.95 and 0.99
    error_threshold: float = 1.0  # Typical between 0.5 and 2.0 pixels
