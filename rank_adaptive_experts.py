import torch

class RunningMean():
    def __init__(self, value_dtype=torch.float64):
        self.value_dtype=value_dtype
        self.reset()
    
    def reset(self):
        self.n = torch.tensor(0, dtype=torch.int64)
        self.accum = torch.tensor(0, dtype=self.value_dtype)

    def update_multiple(self, values):
        assert len(values.shape) == 1
        self.n += values.shape[0]
        self.accum += values.sum()
    
    def update(self, value):
        self.n += 1
        self.accum += value
    
    def get_mean(self):
        return self.accum / self.n
