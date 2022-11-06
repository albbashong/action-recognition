import cv2
import os
import math
import numpy as np
from PIL import Image
import time
import argparse



parser = argparse.ArgumentParser(description="RGB difference for Action Recognition")
# parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51', 'kinetics'])
# parser.add_argument('modality', type=str, choices=['RGB', 'Flow', 'RGBDiff'])
parser.add_argument('--color', type=str, default='rgb',choices=['rgb','gray'])
parser.add_argument('--gen_tpye', type=str, default='rgb_differ',choices=['rgb_differ','convolution','optical_flow'])
parser.add_argument('--save_type',type=str, defualt='view',choises=['img','view','video'])
parser.add_argument('--view_on',type=bool, default=true)
parser.add_argument('--dir_path', type=str,default='./images')
parser.add_argument('--resize',type=bool,default='true')
parser.add_argument('--gaussian',type=bool,default='true')

args = parser.parse_args()



'''
rgb_difference

square(prev_frame - next_frame)

'''

noise_matrix = np.random.randint(1, size=(224, 224, 3),dtype=np.uint8)



motion_conv_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)

motion_conv_y = torch.tensor([[-1., -2., -1.],
                                [0., 0., 0.],
                                [1., 2., 1.]]).unsqueeze(0).unsqueeze(0)

motion_conv_vertical = torch.tensor([[-1., 1., -1.],
                                    [0., 1., 0.],
                                    [-1., 1., -1.]]).unsqueeze(0).unsqueeze(0)

motion_conv_horizontal = torch.tensor([[-1., 0., -1.],
                                    [1., 1., 1.],
                                    [-1., 0., -1.]]).unsqueeze(0).unsqueeze(0)



transform = transforms.Compose([
    transforms.ToTensor()
])
tf = transforms.ToPILImage()
resize_torch=[transforms.Resize((320,180))]


class Motion_Conv(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_x=nn.Conv2d(3,3,kernel_size=(3,3),stride=3,padding=1)
        self.conv_x.weight=nn.Parameter(motion_conv_x.expand(3,3,-1,-1),requires_grad=False)

        self.conv_y = nn.Conv2d(3, 3, kernel_size=(3, 3), stride=3, padding=1)
        self.conv_y.weight = nn.Parameter(motion_conv_y.expand(3, 3, -1, -1),requires_grad=False)

        self.avg=nn.AvgPool2d(2,1,padding=1)
        self.max=nn.MaxPool2d(2,1,padding=1)
        self.relu=nn.ReLU()

        self.thresh=nn.Threshold(0.01, 1, inplace=False)
    def forward(self,data):
        x = self.conv_x(data)
        x = self.max(x)
        x = self.relu(x)

        y=self.convY(data)
        y=self.max(y)
        y=self.relu(y)


        return x,y




def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def max_point(img):
    for i in range(3):
        avg = np.mean(img)
        max = np.max(img)
        img =np.where(img < avg,0,img)
        img =np.where(img>=max,0,img)
    return img


def img_difference(snippet,img):

    font = cv2.FONT_HERSHEY_SIMPLEX
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    fps = 30

    rgb_differ=[]
    save_img=[]
    motion_test = Motion_Conv()
    time_list = []


    for idx_frames in range(0,len(snippet)):
        start = time.time()

        #rgb_differ.append(abs(np.square(cv2.addWeighted(snippet[idx_frames],0.1,snippet[idx_frames-1], 0.1, 0)-snippet[idx_frames-1])))

        if(idx_frames==0):
            rgb_differ.append(np.zeros((snippet[idx_frames].shape),dtype=np.uint8))
        else:

            if args.gen_tpye=="rgb_differ":
                rgb_differ.append(
                (abs((snippet[idx_frames]+snippet[idx_frames - 1]*0.2)) - snippet[idx_frames - 1]).astype(np.uint8))
            # rgb_differ.append(
            #       ((snippet[idx_frames]) - snippet[idx_frames - 1]).astype(np.uint8))

            if args.gen_type=="filter":
                filtered = snippet[idx_frames] - snippet[idx_frames - 1]
                arr=np.where(filtered>128,0,filtered)
                rgb_differ.append(arr)


            #rgb_differ.append((snippet[idx_frames]*0.5 + snippet[idx_frames - 1]*0.5 -filtered).astype(np.uint8))
            #rgb_differ[-1]=max_point(rgb_differ[-1])



            if args.gen_type=="convolution":
                torch_img = transform(rgb_differ[-1])
                data_x,data_y = motion_test(torch_img)
                data_x=data_x.squeeze(0)
                data_y=data_y.squeeze(0)
                data_x = tf(data_x)
                data_y = tf(data_y)
                data_x=np.array(data_x)
                data_y=np.array(data_y)
                '''text in image'''
                # data_x = cv2.putText(data_x, 'data_x', (10,220), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                # data_y = cv2.putText(data_y, 'data_y', (10, 220), font, 1, (255, 255, 255), 2, cv2.LINE_AA)



        '''heat map test'''
        #if len(rgb_differ)>=2 :
        #   rgb_differ[-1][:,:,1:2]=0
        #   rgb_differ[-2][:,:,0]=0
        #   rgb_differ[-2][:,:,2]=0
        #   rgb_differ[-1]=rgb_differ[-1]+rgb_differ[-2]
            #rgb_differ[-1]=rgb_differ[-1]*0.


        '''concat image when compare it'''
        # prvs = cv2.cvtColor(snippet[idx_frames-1], cv2.COLOR_BGR2GRAY)
        # hsv = np.zeros_like(snippet[idx_frames-1])
        # hsv[..., 1] = 255
        # next = cv2.cvtColor(snippet[idx_frames], cv2.COLOR_BGR2GRAY)
        # flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # hsv[..., 0] = ang * 180 / np.pi / 2
        # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        #bgr = cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
        #prvs = next

        time_list.append(time.time() - start)
        sum_list=sum(time_list)
        mean=sum_list/len(time_list)
        print(str(len(time_list))+" mean: "+str(mean))

        if args.view_on:
            cv2.imshow("test",rgb_differ[-1])
            cv2.waitKey(0)


        if args.save_type=="img":
            cv2.imwrite('./rgb_difference/'+img.split('-')[0]+'-'+str(idx_frames)+'.jpg',data_x)
        if args.save_type=="video":
            height,width,layers =save_img[0].shape
            out = cv2.VideoWriter("./cow.mp4", fourcc, fps, (width,height))
            for i in range(len(save_img)):
                 # writing to a image array
                 out.write(save_img[i])
            out.release()

def load_img(dir_path):

    img_list=os.listdir(dir_path)
    img_list.sort()

    frames=[]

    img_idx=0


    for img in img_list:
        img_array = np.fromfile(dir_path+img, np.uint8)
        if args.color=="rgb":
            frame_image = cv2.imdecode(img_array, 1)
        if args.color=="gray":
            frame_image=cv2.cvtColor(cv2.imdecode(img_array, 1),cv2.COLOR_BGR2GRAY)

        if args.resize=='true':
            frame_image=cv2.resize(frame_image, (224, 224), interpolation=cv2.INTER_CUBIC)
        if args.gaussian=='true':
            frame_image = cv2.GaussianBlur(frame_image, (7, 7), 2)



        frames.append(frame_image)
        img_idx += 1
        if(img_idx%len(img_list)==0 and img_idx!=0):
            img_difference(frames,img=img)
            frames=[]

if __name__=="__main__":


    frames = load_img(dir_path)







