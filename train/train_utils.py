import pandas as pd 
from config import Config
from torch.utils.data import DataLoader

cfg = Config()

def get_data(train_csv, valid_csv):
    train_df = pd.read_csv(train_csv)
    valid_df = pd.read_csv(valid_csv)

    train_img, valid_img = [], []
    for i in range(len(train_df)):
        train_img.append((train_df['path'][i], train_df['label'][i]))

    for i in range(len(valid_df)):
        valid_img.append((valid_df['path'][i], valid_df['label'][i]))

    return train_img, valid_img

def prepare_dataloader(Dataset, train_csv, valid_csv, train_batch, valid_batch):
    train_img, valid_img = get_data(train_csv, valid_csv)

    print('Train set: {} images'.format(len(train_img)))
    print('Valid set: {} images'.format(len(valid_img)))

    train_ds = Dataset(train_img)
    valid_ds = Dataset(valid_img)

    train_loader = DataLoader(train_ds, batch_size = train_batch, shuffle=True, num_workers = 4)
    valid_loader = DataLoader(valid_ds, batch_size = valid_batch, num_workers = 4)

    return train_loader, valid_loader