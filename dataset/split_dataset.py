import os
import cv2
import translate
import math
import random
import shutil

split_ratio = 0.2
raw_img = '../dataset/raw-img'

if __name__ == '__main__':
    trans = translate.translate
    for obj in os.listdir(raw_img):
        obj_path = os.path.join(raw_img, obj)
        img_list = os.listdir(obj_path)
        train_sz = math.ceil(len(img_list)*split_ratio)
        test_sz = len(img_list) - train_sz
        
        print('Found {} images in class {}, split training: {}, testing: {}'.format(len(img_list), trans[obj], train_sz, test_sz))

        print('Splitting ...', end = ' ')

        random.shuffle(img_list) # shuffle images

        train_path = '../dataset/train/' + trans[obj]
        test_path = '../dataset/test/' + trans[obj]

        if not os.path.isdir(train_path):
            os.mkdir(train_path)

        if not os.path.isdir(test_path):
            os.mkdir(test_path)

        for iter, img in enumerate(img_list):
            if iter < train_sz:
                shutil.copy2(os.path.join(obj_path, img), train_path)
            else:
                shutil.copy2(os.path.join(obj_path, img), test_path)
        
        print('Done')

