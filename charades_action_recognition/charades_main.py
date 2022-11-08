import torch
import numpy as np
import apmeter as APMeter
from charades_dataloader import *
import argparse


parser = argparse.ArgumentParser(description="Action Recognition Charades Dataset")
parser.add_argument('--test_label_path',type=str, defualt='./Charades/Charades_v1_train.csv')
parser.add_argument('--train_label_path',type=str, default='./Charades/Charades_v1_test.csv')
parser.add_argument('--dir_path', type=str,default='./images')

args = parser.parse_args()




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
    return jy, epoch_loss,


if __name__=="__main__":
    dir_path=args.dir_path
    train_label_path=args.train_label_path
    val_label_path=args.test_label_path
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
            probs = torch.nn.Sigmoid()(classes)
            apm.add(probs.data.cpu().numpy()[0], y_data.numpy()[0])
            train_map = 100 * apm.value().mean()
            print('epoch', epoch_num, 'train-map:', train_map)
            apm.reset()
            #cls_loss=torch.nn.BCELoss(torch.max(probs, dim=1), torch.max(y_data, dim=2))
            loss=label_loss(classes,y_data.cuda())
            loss.backward()
            optimizer.step()
            print('batch_idx: ', batch_idx, ' loss:', loss/batch_idx)

            np_img=origin_img.numpy()


        #val_data(model=model,dataloader=charades_val_dataset,epoch=epoch_num)