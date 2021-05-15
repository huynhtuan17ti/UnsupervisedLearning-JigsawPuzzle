import numpy as np 
import os
import cv2
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from config import Config
from .data_utils import get_all_imgs, rgb_jittering

cfg = Config()

class AnimalDataset(data.Dataset):
    def __init__(self, data_path, classes = 10):
        self.data = data_path

        self.image_transformer = transforms.Compose([
            transforms.Resize(256, Image.BILINEAR),
            transforms.CenterCrop(255),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])])
    
    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        filename, label = self.data[index]

        img = Image.open(filename).convert('RGB')

        img = self.image_transformer(img)

        return img, int(cfg.label_dict[label])


class JigsawDataset(data.Dataset):
    def __init__(self, data_path, classes = 1000):
        self.data = data_path

        self.permutations = self.get_permutations(classes)

        self.image_transformer = transforms.Compose([
            transforms.Resize(256, Image.BILINEAR),
            transforms.CenterCrop(255)])

        self.augment_tile = transforms.Compose([
            transforms.RandomCrop(64),
            transforms.Resize((75, 75), Image.BILINEAR),
            transforms.Lambda(rgb_jittering),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        filename, _ = self.data[index]

        img = Image.open(filename).convert('RGB')
        if np.random.rand() < 0.30:
            img = img.convert('LA').convert('RGB')

        if img.size[0] != 255:
            img = self.image_transformer(img)

        s = float(img.size[0]) / 3
        a = s / 2
        tiles = [None] * 9
        for n in range(9):
            i = n / 3
            j = n % 3
            c = [a * i * 2 + a, a * j * 2 + a]
            c = np.array([c[1] - a, c[0] - a, c[1] + a + 1, c[0] + a + 1]).astype(int)
            tile = img.crop(c.tolist())
            tile = self.augment_tile(tile)
            # Normalize the patches indipendently to avoid low level features shortcut
            m, s = tile.view(3, -1).mean(dim=1).numpy(), tile.view(3, -1).std(dim=1).numpy()
            s[s == 0] = 1
            norm = transforms.Normalize(mean=m.tolist(), std=s.tolist())
            tile = norm(tile)
            tiles[n] = tile

        order = np.random.randint(len(self.permutations))
        data = [tiles[self.permutations[order][t]] for t in range(9)]
        data = torch.stack(data, 0)

        return data, int(order), torch.stack(tiles)

        
    def get_permutations(self, classes):
        all_perm = np.load('../UnsupervisedLearning-JigsawPuzzle/dataset_factory/permutations_%d.npy' % (classes))
        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm

if __name__ == '__main__':
    # testing
    import matplotlib.pyplot as plt
    list_img = get_all_imgs('../UnsupervisedLearning-JigsawPuzzle/dataset/train')
    train_ds = JigsawDataset(list_img)
    train_loader = data.DataLoader(train_ds, batch_size=2)
    col = 3
    row = 3
    cnt = 1
    fig = plt.figure(figsize=(8, 8))
    for data in train_loader:
        batch_data, batch_order, batch_tiles = data
        for data, order, tiles in zip(batch_data, batch_order, batch_tiles):
            print("Permutation: ", order.tolist())
            for tile in data:
                new_tile = tile.permute(1, 2, 0)
                new_tile = np.array(new_tile, np.float32)
                fig.add_subplot(row, col, cnt)
                cnt += 1
                plt.imshow(new_tile)
            plt.show()

            print(tiles.shape)
            fig = plt.figure(figsize=(8, 8))
            cnt = 1
            for tile in tiles:
                new_tile = tile.permute(1, 2, 0)
                new_tile = np.array(new_tile, np.float32)
                fig.add_subplot(row, col, cnt)
                cnt += 1
                plt.imshow(new_tile)
            plt.show()
            break