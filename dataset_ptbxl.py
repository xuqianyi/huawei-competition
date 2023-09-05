import torch
import numpy as np
from torch.utils.data import Dataset

import warnings
# np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)  

class MyDataset(Dataset):
    def __init__(self, dataset):
        super(MyDataset, self).__init__()

        X_train = dataset["samples"]
        y_train = dataset["labels"]

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).float()
        else:
            self.x_data = X_train
            self.y_data = y_train

    def __getitem__(self, index):

        return self.x_data[index].float(), self.y_data[index].float()

    def __len__(self):
        return len(self.x_data)

def get_dataset(data_path):

    train_dataset = torch.load(data_path)

    train_dataset = MyDataset(train_dataset)

    print("Training data size: ", len(train_dataset))

    return train_dataset



if __name__ == "__main__":
    from tqdm import tqdm

    data = '/data/lanx/ecgs/processed/Diagnostic/data/val.pt'

    train_dataset = get_dataset(data)
    for i, (x, y) in enumerate(train_dataset):
        print(x.shape)
        print(y.shape)
        np.save("sampleECG.npy", x)
        break

    #label map:
    # array(['1AVB', '2AVB', '3AVB', 'ALMI', 'AMI', 'ANEUR', 'ASMI', 'CLBBB',
    #     'CRBBB', 'DIG', 'EL', 'ILBBB', 'ILMI', 'IMI', 'INJAL', 'INJAS',
    #     'INJIL', 'INJIN', 'INJLA', 'IPLMI', 'IPMI', 'IRBBB', 'ISCAL',
    #     'ISCAN', 'ISCAS', 'ISCIL', 'ISCIN', 'ISCLA', 'ISC_', 'IVCD',
    #     'LAFB', 'LAO/LAE', 'LMI', 'LNGQT', 'LPFB', 'LVH', 'NDT', 'NORM',
    #     'NST_', 'PMI', 'RAO/RAE', 'RVH', 'SEHYP', 'WPW'], dtype=object)    

