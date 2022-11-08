import Transformer.utils.Custom_Resnet3d as Custom_Resnet
import Transformer.utils.RGB_to_differ as RGB_to_differ
import torch.nn as nn
import Transformer.utils.ViVit_with_differ as ViVit_with_differ
import Transformer.utils.Custom_CLIP as Custom_CLIP
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
Bottleneck= Custom_Resnet.Bottleneck
model= Custom_Resnet.ResNet(block=Bottleneck, layers=[3, 4, 6, 3], num_classes=157) # resnet50



class Concat_model(nn.Module):
    def __init__(self,image_size,clip_size=64):
        super().__init__()
        self.RGB_to_differ=RGB_to_differ.RGB_differ(image_size=image_size,clip_size=clip_size,patch_size=16).cuda()
        # self.resnet=model.cuda()
        self.ViVit_with_differ=ViVit_with_differ.ViViT(image_size=image_size,patch_size=16,num_classes=157,num_frames=clip_size).cuda()
        #self.Custom_CLIP=Custom_CLIP.CLIP(embed_dim=157,image_resolution=224,vision_layers=[3,4,6,3],vision_width=512,vision_patch_size=16,transformer_width=512,transformer_heads=8,transformer_layers=64)
    def forward(self,rgb,origin_img):
        rgb_differ = self.RGB_to_differ(origin_img)

        rgb_results = self.ViVit(rgb_differ.cuda())



        return rgb_results


