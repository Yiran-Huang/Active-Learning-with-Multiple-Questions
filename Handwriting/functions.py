from torch.utils.data import Dataset, TensorDataset, random_split, DataLoader
def get_bs(batch_size,datalen):
    if datalen<= batch_size:
        return batch_size
    bs_true = batch_size if datalen % batch_size == 0 or datalen % batch_size >= (batch_size // 2) else (datalen - batch_size // 2) // (datalen // batch_size)
    return bs_true

#### Now we build the model ####
class CustomDataSet(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        # return len(self.x)
        return len(self.x)

class CustomDatasetx(Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index]


    