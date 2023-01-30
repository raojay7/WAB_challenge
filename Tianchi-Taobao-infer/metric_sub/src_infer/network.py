import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from loss import ArcFace
from cross_batch_memory import CrossBatchMemory
from layers import GeM


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, model='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, model='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class ENet(nn.Module):
    def __init__(self, num_classes=3, feat_dim=512, cos_layer=True, xbm=None, dropout=0., m=0.5, pool='gem_freeze', image_net='tf_efficientnet_b3_ns', pretrained=True):
        super().__init__()

        self.feat_dim = feat_dim
        self.cos_layer = cos_layer
        self.xbm = xbm

        if pretrained == True:
            backbone = timm.create_model(image_net, pretrained=True)
        else:
            backbone = timm.create_model(image_net, pretrained=False)

        self.base = nn.Sequential(
            backbone.conv_stem,
            backbone.bn1,
            backbone.act1,
            backbone.blocks,
            backbone.conv_head,
            backbone.bn2,
            backbone.act2)

        if pool == 'gem_freeze':
            self.pool = GeM(p=3.0, freeze_p=True)
        elif pool == 'gem':
            self.pool = GeM(p=3.0, freeze_p=False)
        elif pool == 'gap':
            self.pool = backbone.global_pool
        self.dropout = nn.Dropout(p=dropout)

        features_num = backbone.num_features * backbone.global_pool.feat_mult()
        # self.neck = nn.Sequential(
        #     nn.BatchNorm1d(features_num),
        #     nn.Linear(features_num, feat_dim, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm1d(feat_dim),
        #     nn.Linear(feat_dim, feat_dim, bias=False)
        # )
        self.neck = nn.Linear(features_num, feat_dim, bias=False)

        self.bottleneck = nn.BatchNorm1d(feat_dim)

        self.num_classes = num_classes

        if self.cos_layer:
            print('using cosine layer')
            self.arcface = ArcFace(self.feat_dim, self.num_classes, s=30.0, m=m)
            if xbm is not None:
                self.arcface = CrossBatchMemory(self.arcface, self.feat_dim, memory_size=xbm)
        else:
            self.classifier = nn.Linear(self.feat_dim, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)


    def forward(self, x, label=None): # label is only used when using cos_layer
        x = self.base(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        global_feat = self.neck(x) # not being used in triplet loss when bias=False in nn.Linear

        feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                if self.xbm is not None:
                    cls_score, combined_label = self.arcface(feat, label)
                    return cls_score, combined_label, global_feat  # global feature for triplet loss
                else:
                    cls_score = self.arcface(feat, label)
                    return cls_score, global_feat
            else:
                cls_score = self.classifier(feat)
                return cls_score, global_feat
        else:
            return feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i or 'arcface' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

class Vit(nn.Module):
    def __init__(self, num_classes=3, feat_dim=512, cos_layer=True, xbm=None, dropout=0., m=0.5, pool='gem_freeze', image_net='vit_deit_base_distilled_patch16_384', pretrained=True):
        super().__init__()

        self.feat_dim = feat_dim
        self.cos_layer = cos_layer
        self.xbm = xbm

        if pretrained == True:
            backbone = timm.create_model(image_net, pretrained=True)
        else:
            backbone = timm.create_model(image_net, pretrained=False)

        self.patch_embed=backbone.patch_embed
        self.cls_token =backbone.cls_token# nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop=backbone.pos_drop
        self.num_tokens = backbone.num_tokens #if distilled 2 else 1
        self.pos_embed = backbone.pos_embed
        self.blocks=backbone.blocks
        self.norm=backbone.norm
        self.pre_logits=backbone.pre_logits
        self.dist_token = backbone.dist_token
        self.head_dist = backbone.head_dist
        self.dropout = nn.Dropout(p=dropout)

        features_num = backbone.head.in_features
        self.neck = nn.Linear(features_num, feat_dim, bias=False)

        self.bottleneck = nn.BatchNorm1d(feat_dim)

        self.num_classes = num_classes
        if (self.dist_token is not None):
            print('using distill')
        if (self.head_dist is not None):
            print('original model uses dis-classifier')
        if self.cos_layer:
            print('using cosine layer')
            self.arcface = ArcFace(self.feat_dim, self.num_classes, s=30.0, m=m)
            if xbm is not None:
                self.arcface = CrossBatchMemory(self.arcface, self.feat_dim, memory_size=xbm)
        else:
            self.classifier = nn.Linear(self.feat_dim, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x, label=None): # label is only used when using cos_layer
        x,_ = self.forward_features(x)
        x = self.dropout(x)
        global_feat = self.neck(x) # not being used in triplet loss when bias=False in nn.Linear
        feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                if self.xbm is not None:
                    cls_score, combined_label = self.arcface(feat, label)
                    return cls_score, combined_label, global_feat  # global feature for triplet loss
                else:
                    cls_score = self.arcface(feat, label)
                    return cls_score, global_feat
            else:
                cls_score = self.classifier(feat)
                return cls_score, global_feat
        else:
            return feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i or 'arcface' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))
