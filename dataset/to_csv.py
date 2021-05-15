import sys
sys.path.insert(1, '../UnsupervisedLearning-JigsawPuzzle')
import pandas as pd 
import numpy as np 
from sklearn.model_selection import StratifiedKFold, train_test_split
import cv2
import os
from config import Config
from dataset_factory.data_utils import get_all_imgs

cfg = Config()

def get_data():
    data = get_all_imgs(cfg.train_path, return_label=True)
    path_img, label_img = [], []
    for (img, label) in data:
        path_img.append(img)
        label_img.append(label)
    
    return np.array(path_img), np.array(label_img)

def count_unique(arr):
    '''
        print counts for unique values in an array
    '''
    unique, counts = np.unique(arr, return_counts=True)
    for i in range(len(unique)):
        print(unique[i], counts[i])

if __name__ == '__main__':
    path_img, label_img = get_data()
    count_unique(label_img)
    print('-'*30)

    X_train, X_valid, y_train, y_valid = train_test_split(path_img, label_img, test_size = 0.2, random_state = 2021)
    

    print("Length (train): {}, length (valid): {}".format(len(X_train), len(X_valid)))
    count_unique(y_train)
    train_df = pd.DataFrame({'path': X_train, 'label': y_train})
    train_df.to_csv('train.csv')

    print('-'*30)

    valid_df = pd.DataFrame({'path': X_valid, 'label': y_valid})
    count_unique(y_valid)
    valid_df.to_csv('test.csv')