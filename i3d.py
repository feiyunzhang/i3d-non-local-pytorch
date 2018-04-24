import torch
import torchvision
from torch.nn.init import normal, constant
from transforms import *
import models.i3dnon
class I3DModel(torch.nn.Module):
    def __init__(self, num_class, sample_frames, modality,
                 base_model='resnet101',
                 dropout=0.8):
        super(I3DModel, self).__init__()
        self.modality = modality
        self.sample_frames = sample_frames
        self.reshape = True
        self.dropout = dropout

        print(("""
Initializing I3D with base model: {}.
I3D Configurations:
    input_modality:     {}
    sample_frames:      {}
    dropout_ratio:      {}
        """.format(base_model, self.modality, self.sample_frames, self.dropout)))

        self._prepare_base_model(base_model)
        self._prepare_i3d(num_class)


    def _prepare_i3d(self, num_class):
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, torch.nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, torch.nn.Dropout(p=self.dropout))
            self.new_fc = torch.nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            normal(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            normal(self.new_fc.weight, 0, std)
            constant(self.new_fc.bias, 0)

    def _prepare_base_model(self, base_model):
        if 'resnet101' in base_model or 'resnet152' in base_model:
            self.base_model = getattr(models, base_model)(pretrained=True)
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))


    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.BatchNorm3d):  # enable BN
                bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]


    def forward(self, input):
        out = self.base_model(input)
        if self.dropout > 0:
            out = self.new_fc(out)

        return out


    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self,mode='train'):
        resize_range_min = self.scale_size
        if mode == 'train':
            resize_range_max = self.input_size * 320 // 224
            return torchvision.transforms.Compose(
                [GroupRandomResizeCrop([resize_range_min, resize_range_max], self.input_size),
                 GroupRandomHorizontalFlip(is_flow=False),
                 GroupColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05)])
        elif mode == 'val':
            return torchvision.transforms.Compose([GroupScale(resize_range_min),
                                                   GroupCenterCrop(self.input_size)])
