import torch
from network import ENet
from collections import OrderedDict
#tf_efficientnet_b5_ns
net = ENet(num_classes=59895, feat_dim=512, cos_layer=True, xbm=None, dropout=0., m=0.30, pool='gem', image_net='tf_efficientnet_b6_ns', pretrained=True)
torch.save({'model_state_dict': net.state_dict()}, '/root/Tianchi-Taobao-infer/metric_sub/pretrained/pretrained_effnet_b6.pt')