import numpy as np 
import os
import cv2

def get_all_imgs(data_path, return_label = False):
    imgs = []
    for obj in os.listdir(data_path):
        obj_path = os.path.join(data_path, obj)
        for img in os.listdir(obj_path):
            if return_label:
                imgs.append((os.path.join(obj_path, img), obj))
            else:
                imgs.append(os.path.join(obj_path, img))
    return imgs

def rgb_jittering(im):
    im = np.array(im, 'int32')
    for ch in range(3):
        im[:, :, ch] += np.random.randint(-2, 2)
    im[im > 255] = 255
    im[im < 0] = 0
    return im.astype('uint8')