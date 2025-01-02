from dataset import Dataset

class Pipeline:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def initialize(self, frame1_index, frame2_index):
        frame1 = self.dataset.get_frame(frame1_index)
        frame2 = self.dataset.get_frame(frame2_index)

    def run(self):
        raise NotImplementedError()
