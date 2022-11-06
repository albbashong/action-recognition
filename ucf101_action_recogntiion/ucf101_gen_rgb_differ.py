import cv2
import os
import math
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import math
from typing import Tuple
from torch import nn, Tensor
import torchvision.transforms as transforms
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from PIL import Image
import time
#dir_path="E:\\data\\data\\ava\\frames\\_7oWZq_s_Sk\\"
#0V4B3 writing
#0SXRS clean the floor
#reading a book and walking 6L2J7



#00T1E bed room
#dir_path="C:/Users/시각지능/Downloads/Charades_v1_rgb/Charades_v1_rgb/0SXRS/"
#dir_path="C:/Users/시각지능/Downloads/새 폴더/"
dir_path="D:/ucf101_jpegs_256/jpegs_256/"
save_path="D:/ucf101_rgb2/"

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def img_difference(snippet,img_save_path,img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    fps = 30
    rgb_differ=[]
    save_img=[]

    time_list = []


    for idx_frames in range(0,len(snippet)):
        start = time.time()

        if(idx_frames==0):
            rgb_differ.append(snippet[idx_frames]*0.8)
        else:
            # rgb_differ.append(
            #     ((snippet[idx_frames]*0.3+snippet[idx_frames - 1]*0.7)+3).astype(np.uint8) - snippet[idx_frames - 1])
            rgb_differ.append(
                (abs((snippet[idx_frames] + snippet[idx_frames - 1] * 0.6) + 3) - snippet[idx_frames - 1]).astype(
                    np.uint8))

        time_list.append(time.time() - start)
        sum_list=sum(time_list)
        mean=sum_list/len(time_list)
        print(str(len(time_list))+" mean: "+str(mean))
        cv2.imwrite(img_save_path+'frame{0:06d}.jpg'.format(idx_frames+1),rgb_differ[-1])


def load_img(path):

    img_list=os.listdir(dir_path+path)
    img_list.sort()
    frames=[]

    img_idx=0

    img_save_path = save_path + path+"/"
    try:
        os.mkdir(save_path+path)
    except Exception as e:
        print(e)

    for img in img_list:
        img_array = np.fromfile(dir_path+path+"/"+img, np.uint8)
        #frame_image=cv2.cvtColor(cv2.imdecode(img_array, 1),cv2.COLOR_BGR2GRAY)
        frame_image = cv2.imdecode(img_array, 1)

        #dst=cv2.resize(frame_image, (224, 224), interpolation=cv2.INTER_CUBIC)
        dst = cv2.GaussianBlur(frame_image, (0, 0), 2)
        #dst = cv2.Sobel(dst, cv2.CV_8U, 0, 1, ksize=3)
        frames.append(dst)
        img_idx += 1
        if(img_idx%len(img_list)==0 and img_idx!=0):
            img_difference(frames,img_save_path,img)

if __name__=="__main__":

    dir_list = os.listdir(dir_path)
    for get_dir in range(len(dir_list)):
        frames = load_img(dir_list[get_dir])






