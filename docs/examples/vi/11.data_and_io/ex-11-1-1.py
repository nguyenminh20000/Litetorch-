import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

class CustomTextDataset(lt.data.Dataset):
    def __init__(self, data_list):
        super().__init__()
        self.samples = data_list

    def size(self):
        return len(self.samples)

    def get(self, index):
        sample = self.samples[index]
        x_tensor = lt.Tensor.from_vector(sample[0], [len(sample[0])], lt.auto_device(), False)
        y_tensor = lt.Tensor.from_vector([sample[1]], [1], lt.auto_device(), False)
        return (x_tensor, y_tensor)
