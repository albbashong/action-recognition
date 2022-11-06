import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange



def out_ch(data):
    height,width=data.size()



class Temporal(nn.Module):

    def __init__(self,image_size,clip_size,patch_size,ch=3,dim=192):
        super().__init__()
        patch_dim = ch * patch_size ** 2
        self.temporal_mult= nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )


    def forward(self,x):
        #x = np.squeeze(x,axis=0)
        x = self.rgb_differ(x)
        x =x.permute(0, 1, 4, 3, 2).cuda() # b,t,c,h,w
        #x = self.temporal_mult(x)
        # x = torch.flatten(x, 1)
        # x = self.temporal_mult(x)

        return x

    def rgb_differ(self, snippet):
        rgb_differ = None

        for batch_idx in range(0, len(snippet)):
            for idx_frames in range(0, len(snippet[0])):

                if (idx_frames == 0):
                    rgb_differ=snippet
                    rgb_differ[:][1:]=0
                    #rgb_differ=torch.zeros((snippet.shape))
                else:
                    # rgb_differ=torch.cat([rgb_differ[batch_idx],
                    #     torch.unsqueeze((snippet[batch_idx][idx_frames] * 0.3 + snippet[batch_idx][idx_frames - 1] * (0.7) + 3)- snippet[batch_idx][idx_frames - 1],dim=0)])
                    #rgb_differ[batch_idx][idx_frames]=(snippet[batch_idx][idx_frames] * 0.3 + snippet[batch_idx][idx_frames - 1] * (0.7) + 3)- snippet[batch_idx][idx_frames - 1]
                    rgb_differ[batch_idx][idx_frames]=(snippet[batch_idx][idx_frames] + snippet[batch_idx][idx_frames - 1] * 0.6 + 3) - snippet[batch_idx][idx_frames - 1]



        return rgb_differ

