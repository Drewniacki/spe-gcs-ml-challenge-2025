import random

from torch.utils.data import Sampler

from dataset import TemperatureDataset


class HoleShuffler(Sampler):
    def __init__(self, tds: TemperatureDataset, test_hole_num: int, test_size: int):
        self.tds = tds
        self.train_index, self.test_index, _ = tds.get_train_test_split(
            test_hole_num=test_hole_num,
            test_size=test_size,
        )
        self.index_per_hole = self.tds.get_index_per_hole(exclude=self.test_index.tolist())

        self.pos = 0
        self.hole = 0
        self.num_holes = len(self.index_per_hole)
        self.remaining_holes = [i for i in range(self.num_holes)]
        random.shuffle(self.remaining_holes)

    def __iter__(self):
        # Reshuffle holes and start from beginning
        self.remaining_holes = [i for i in range(self.num_holes)]
        random.shuffle(self.remaining_holes)
        self.hole = self.remaining_holes.pop()
        self.pos = 0
        return self

    def __len__(self):
        return len(self.train_index)

    def __next__(self):
        if self.pos >= len(self.index_per_hole[self.hole]):
            # If possible, take next hole, if not -> stop iteration
            if len(self.remaining_holes) == 0:
                raise StopIteration
            self.hole = self.remaining_holes.pop()
            self.pos = 0

        value = self.index_per_hole[self.hole][self.pos]
        self.pos += 1

        return value
