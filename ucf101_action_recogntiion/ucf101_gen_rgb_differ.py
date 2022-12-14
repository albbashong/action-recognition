import cv2
import os
import numpy as np
from PIL import Image
import time
import argparse

parser = argparse.ArgumentParser(description="Action Recognition UCF101 Dataset")
parser.add_argument('--save_pathh',type=str, default='./ucf101_rgb_difference')
parser.add_argument('--dir_path', type=str,default='./ucf101')

args = parser.parse_args()

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



'''

load image list

'''
def load_img(path):
    img_list=os.listdir(args.dir_path+path)
    img_list.sort()
    frames=[]

    img_idx=0

    img_save_path = args.save_path + path+"/"
    try:
        os.mkdir(args.save_path+path)
    except Exception as e:
        print(e)

    for img in img_list:
        img_array = np.fromfile(args.dir_path+path+"/"+img, np.uint8)
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

    dir_list = os.listdir(args.dir_path)
    for get_dir in range(len(dir_list)):
        frames = load_img(dir_list[get_dir])






