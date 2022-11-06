import random

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import cv2
import numpy as np
import pandas
import torch
import math
import torchvision.transforms as transforms
from PIL import Image
import Transformer.utils.model as model
import numbers
import random
from apmeter import APMeter
from torch.autograd import Variable
from torch.utils.data.dataloader import default_collate
class Rescale(object):

    def __init__(self,output_size):
        assert isinstance(output_size,int)
        self.output_size = output_size

    def __call__(self,sample):
        image = sample

        h,w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h> w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transforms.Resize()


class RandomCrop(object):
    """Crop the given video sequences (t x h x w) at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        t, h, w, c = img.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th) if h != th else 0
        j = random.randint(0, w - tw) if w != tw else 0
        return i, j, th, tw

    def __call__(self, imgs):

        i, j, h, w = self.get_params(imgs, self.size)

        imgs = imgs[:, i:i + h, j:j + w, :]
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class CenterCrop(object):
    """Crops the given seq Images at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        t, h, w, c = imgs.shape
        th, tw = self.size
        i = int(np.round((h - th) / 2.))
        j = int(np.round((w - tw) / 2.))

        return imgs[:, i:i + th, j:j + tw, :]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RandomHorizontalFlip(object):
    """Horizontally flip the given seq Images randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs):
        """
        Args:
            img (seq Images): seq Images to be flipped.
        Returns:
            seq Images: Randomly flipped seq images.
        """
        if random.random() < self.p:
            # t x h x w
            return np.flip(imgs, axis=2).copy()
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

def get_dir_list(dir_path):
    '''
    :param dir_path: dir path last word '/' musb be exist
    :return: list of dir
    '''

    dir_list = os.listdir(dir_path)
    return dir_list

def get_img_list(img_path):
    '''
    :param dir_list: dir_list_idx
    :param dir_path: dir path last word '/' musb be exist
    :return: list of img
    '''
    img_list=(os.listdir(img_path))
    return img_list

def get_label(label_path):
    label_info= pandas.read_csv(label_path, usecols=[0, 9,10])
    return label_info



class Custom_Dataset(Dataset):

    def __init__(self,dir_path,label_path,data_type,classes=157,clip_size=64,gap=1,height=224,width=224):
        self.classes=classes ## Charades
        self.label_info=get_label(label_path)
        self.fix_label_info=self.fix_label()
        self.dir_path=dir_path
        self.dir_list=get_dir_list(dir_path)
        self.gap=gap
        self.height=height
        self.width=width
        self.vid_list = self.label_info['id'].tolist()
        self.clip_size=clip_size
        self.data_type=data_type
        self.transform = transforms.Compose(
            [CenterCrop(224)]
        )

    def fix_label(self):
        '''

        read label information about action class
        :return:
        '''
        fix_label_info=[]
        label_actions = self.label_info['actions'].tolist()
        for idx in range(len(self.label_info)):
            vid_action = label_actions[idx]
            if (type(vid_action) != float): # vid_action is not null
                fix_label_info.append(idx)
        return fix_label_info
    def __len__(self):
        return len(self.fix_label_info)

    def __getitem__(self, idx):
        img, target, vid = self.extract_img(idx)
        #img=self.transform(img)
        start_frame=random.randint(0,len(img)-(self.clip_size+1))

        clip_img=img[start_frame:start_frame+self.clip_size] # cut frames action duration
        clip_target=target[:,start_frame:start_frame+self.clip_size] # target each frame
        clip_vid=vid[start_frame:start_frame+self.clip_size] # video information

        origin_img = clip_img

        clip_img=torch.from_numpy(clip_img.transpose([0,3,1,2]))

        #target=torch.FloatTensor(target)


        return clip_img,origin_img, clip_target.transpose(1,0), clip_vid





    def extract_img(self, idx):
        idx=self.fix_label_info[idx]
        return_img_data=[]
        return_vid=[]
        # label = pandas.read_csv(train_csv_path,names=["id","subject","scene","quality","relevance",
        # "verified","script","objects","description","actions"])

        label_actions = self.label_info['actions'].tolist()
        '''
        dir_name == vid_id
        '''
        img_list=get_img_list(self.dir_path+self.vid_list[idx])
        duration = self.label_info['length'].tolist()
        return_label = np.zeros((self.classes, len(img_list)),np.float32)
        vid_action=label_actions[idx]
        fps=len(img_list)/duration[idx]

        '''
        random frame position extract
        
        '''


        if (type(vid_action)!=float):
            class_actions = vid_action.split(';')

            '''
            actions split class, start_time,end_time
            '''
            for img_idx in range(0, len(img_list)):
                #return_img_data.append(self.transform(Image.open(self.dir_path + self.vid_list[idx] + "/" + img_list[img_idx]).convert('RGB')))
                #return_img_data.append((self.dir_path + self.vid_list[idx] + "/" + img_list[img_idx]).convert('RGB'))
                img_array = np.fromfile(dir_path +self.vid_list[idx] + "/" + img_list[img_idx], np.uint8)
                frame_image = cv2.imdecode(img_array, 1)
                frame_image = cv2.resize(frame_image,(self.height,self.width))
                frame_image = frame_image/255
                return_img_data.append(frame_image)
                return_vid.append(self.vid_list[idx])

            for each_action in class_actions:
                sp_class = each_action.split(' ')[0]
                start_time = each_action.split(' ')[1]
                end_time = each_action.split(' ')[2]
                '''
                data type val or train
                '''
                if self.data_type != 'val':
                    for  img_idx in range(0,len(img_list)):
                        if float(start_time)< img_idx/float(fps) <float(end_time):
                            return_label[int(sp_class[1:]),img_idx]=1 ## ex sp_class 'c120' to 120

                else:
                    for  img_idx in range(0,len(img_list)):
                        target = torch.IntTensor(157).zero_()
                        if float(start_time)< img_idx/float(fps) <float(end_time):
                            return_label[int(sp_class[1:]),img_idx]=1



            return np.asarray(return_img_data, dtype=np.float32),return_label,return_vid


def mt_collate_fn(batch):
    "Pads data and puts it into a tensor of same dimensions"

    max_len_clips = 0
    max_len_labels = 0
    for b in batch:
        if b[0].shape[2] > max_len_clips:
            max_len_clips = b[0].shape[2]
        if b[1].shape[1] > max_len_labels:
            max_len_labels = b[1].shape[1]

    new_batch = []
    for b in batch:
        clips = np.zeros((b[0].shape[0], b[0].shape[1], max_len_clips, b[0].shape[3], b[0].shape[4]), np.float32)
        label = np.zeros((b[1].shape[0], max_len_labels), np.float32)
        mask = np.zeros((max_len_labels), np.float32)

        clips[:,:,:b[0].shape[2],:,:] = b[0] #[:,:,:min(cap_clip,b[0].shape[2]),:,:]
        label[:,:b[1].shape[1]] = b[1] #[:,:min(cap_label,b[1].shape[1])]
        mask[:b[1].shape[1]] = 1

        new_batch.append([torch.from_numpy(clips), torch.from_numpy(label), torch.from_numpy(mask), b[2]])

    return default_collate(new_batch)


def focal_loss(preds, targets):
  '''
  Action focal loss.
  '''
  targets=targets.transpose(1,2)
  pos_inds = targets.eq(1).float()
  neg_inds = targets.lt(1).float()

  neg_weights = torch.pow(1 - targets, 4)

  loss = 0
  for pred in preds:
    pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
      loss = loss - neg_loss
    else:
      loss = loss - (pos_loss + neg_loss) / num_pos
  return loss / len(preds)


def val_data(model, dataloader,epoch):
    model.train(False)
    apm = APMeter()
    sampled_apm = APMeter()
    tot_loss = 0.0
    error = 0.0
    num_iter = 0.
    full_probs = {}
    for data in dataloader:
        num_iter += 1

        x_data, origin_img, y_data, vid = clip_data
        classes = model(x_data, origin_img)

        label_loss = torch.nn.functional.binary_cross_entropy_with_logits(classes, y_data.squeeze(0).cuda(),
                                                                          size_average=False)

        tot_loss += label_loss.data

    epoch_loss = tot_loss / num_iter




    print(100 * sampled_apm.value())
    apm.reset()
    sampled_apm.reset()
    return full_probs, epoch_loss,


if __name__=="__main__":
    dir_path="C:/Users/시각지능/Downloads/Charades_v1_rgb/Charades_v1_rgb/"
    train_label_path="C:/Users/시각지능/Downloads/Charades/Charades_v1_train.csv"
    val_label_path="C:/Users/시각지능/Downloads/Charades/Charades_v1_test.csv"
    epoch=50
    lr_rate=0.0001
    image_size=224
    clip_size=64
    validation_clip=25
    apm = APMeter()
    font = cv2.FONT_HERSHEY_PLAIN
    model=model.Concat_model(image_size=image_size,clip_size=clip_size)
    #dir_path,label_path,fps,gap,data_type,classes=157
    dataset_config=Custom_Dataset(dir_path=dir_path,label_path=train_label_path,gap=2,clip_size=clip_size,data_type="train",classes=157)
    val_dataset_config=Custom_Dataset(dir_path=dir_path,label_path=val_label_path,gap=2,clip_size=validation_clip,data_type="test",classes=157)
    charades_dataset=DataLoader(dataset_config,batch_size=2,shuffle=True,num_workers=0)
    charades_val_dataset=DataLoader(val_dataset_config,batch_size=3,shuffle=True,num_workers=0)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr_rate)
    label_loss=torch.nn.BCEWithLogitsLoss()
    for epoch_num in range(epoch):
        for data in enumerate(charades_dataset):
            batch_idx, clip_data=data
            x_data, origin_img, y_data, vid=clip_data
            classes=model(x_data,origin_img)

            optimizer.zero_grad()
            # probs = torch.nn.Sigmoid()(classes)
            # apm.add(probs.data.cpu().numpy()[0], y_data.numpy()[0])
            # train_map = 100 * apm.value().mean()
            # print('epoch', epoch_num, 'train-map:', train_map)
            apm.reset()
            #cls_loss=torch.nn.BCELoss(torch.max(probs, dim=1), torch.max(y_data, dim=2))
            loss=label_loss(classes,y_data.cuda())
            loss.backward()
            optimizer.step()
            print('batch_idx: ', batch_idx, ' loss:', loss/batch_idx)

            np_img=origin_img.numpy()
            for frame in range(len(np_img[0])):
                #img = cv2.putText(np_img[0][frame], "gt: "+str(y_data[0][frame].argmax())+"pr: "+str(classes.argmax()),(0, 200), font, 1, (0,0,0), 1, cv2.LINE_AA)
                print("gt: "+str(torch.topk(y_data[0][frame],5))+"\n pr: "+str(torch.topk(classes[0][frame],5)))
                cv2.imshow("test",np_img[0][frame])
                cv2.waitKey(100)


        #val_data(model=model,dataloader=charades_val_dataset,epoch=epoch_num)








