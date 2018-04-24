import torch
import torch.nn as nn
from torch.nn import functional as F
__all__ = ['I3DResNet', 'resnet50', 'resnet101', 'resnet152']

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, time_kernel=1, space_stride=1, downsample=None,addnon = None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(time_kernel,1,1), padding=(int((time_kernel-1)/2), 0,0),bias=False) # timepadding: make sure time-dim not reduce
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1,3,3), stride=(1,space_stride,space_stride),
                               padding=(0,1,1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=(1,1,1), bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.addnon = addnon
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        if self.addnon is not None:
            out = nonlocalnet(out,out.size(1))
        return out



class I3DResNet(nn.Module):

    def __init__(self, block, layers, frame_num=32, num_classes=400):
        self.inplanes = 64
        super(I3DResNet, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(5,7,7), stride=(2,2,2), padding=(2,3,3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1))
        self.layer1 = self._make_layer_inflat(block, 64, layers[0])
        self.temporalpool = nn.MaxPool3d(kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0))
        self.layer2 = self._make_layer_inflat(block, 128, layers[1], space_stride=2)
        self.layer3 = self._make_layer_inflat(block, 256, layers[2], space_stride=2)
        self.layer4 = self._make_layer_inflat(block, 512, layers[3], space_stride=2)
        self.avgpool = nn.AvgPool3d((int(frame_num/8),7,7))
        self.avgdrop =nn.Dropout(0.5)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

   
    def _make_layer_inflat(self, block, planes, blocks, space_stride=1):
        downsample = None
        if space_stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=(1,1,1), stride=(1,space_stride,space_stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        time_kernel = 3 #making I3D(3*1*1)
        

        layers.append(block(self.inplanes, planes, time_kernel, space_stride, downsample,addnon= None))
        self.inplanes = planes * block.expansion
        if  blocks == 3:
            for i in range(1, blocks):
                if i % 2 == 1:
                    time_kernel = 3
                else:
                    time_kernel = 1
                layers.append(block(self.inplanes, planes, time_kernel))

        elif  blocks == 4:
            for i in range(1, blocks):
                
                if i % 2 == 1:
                    time_kernel = 3
                    layers.append(block(self.inplanes, planes, time_kernel,addnon= True))
                else:
                    time_kernel = 1
                    layers.append(block(self.inplanes, planes, time_kernel))

        elif  blocks == 23:
            for i in range(1, blocks):
                if i % 2 == 1 :
                    #addnon = True
                    time_kernel = 3
                else:
                    time_kernel = 1
                if i % 7 == 6:
                    addnon=True
                    layers.append(block(self.inplanes, planes, time_kernel,addnon=True))
                    
                else:
                    layers.append(block(self.inplanes, planes, time_kernel))             
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.temporalpool(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(x.size(0), -1)
        x = self.avgdrop(x)
        x = self.fc(x)

        return x


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = I3DResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = I3DResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        import torchvision
        if torch.cuda.is_available():
            pretrained_model = torch.load('/workspace/fyzhang/projects/i3d-nonlocal-affine/kinetics_i3dresnet101__rgb_model_best.pth.tar')
            base_dict = {'.'.join(k.split('.')[2:]): v for k,v in list(pretrained_model['state_dict'].items())}
            model = affine_weights(base_dict, model)
            print("ok")
        else:
            #pretrained_model = torchvision.models.resnet101(pretrained=True)
            pretrained_model = torch.load('/Users/fyzhang/Downloads/vscode/nonlocal/i3d-pytorch/kinetics_i3dresnet101__rgb_model_best.pth.tar')
            model = affine_weights(pretrained_model, model)
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = I3DResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

    return model


# def inflat_weights(model_2d, model_3d):
#     pretrained_dict_2d = model_2d.state_dict()
#     model_dict_3d = model_3d.state_dict()
#     for key,weight_2d in pretrained_dict_2d.items():
#         if key in model_dict_3d:
#             if 'conv' in key:
#                 time_kernel_size = model_dict_3d[key].shape[2]
#                 if 'weight' in key:
#                     weight_3d = weight_2d.unsqueeze(2).repeat(1,1,time_kernel_size,1,1)
#                     weight_3d = weight_3d / time_kernel_size
#                     model_dict_3d[key] = weight_3d
#                 elif 'bias' in key:
#                     model_dict_3d[key] = weight_2d
#             elif 'bn' in key:
#                     model_dict_3d[key] = weight_2d
#             #elif 'fc' in key:
#              #   if 'weight' in key:
#               #      time_kernel_size = model_dict_3d[key].shape[1] / weight_2d.shape[1]
#                #     weight_3d = weight_2d.repeat(1, int(time_kernel_size))
#               #      weight_3d = weight_3d / time_kernel_size
#                #     model_dict_3d[key] = weight_3d
#                 #elif 'bias' in key:
#                  #   model_dict_3d[key] = weight_2d
#             elif 'downsample' in key:
#                 if '0.weight' in key:
#                     time_kernel_size = model_dict_3d[key].shape[2]
#                     weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_kernel_size, 1, 1)
#                     weight_3d = weight_3d / time_kernel_size
#                     model_dict_3d[key] = weight_3d
#                 else:
#                     model_dict_3d[key] = weight_2d

#     model_3d.load_state_dict(model_dict_3d)
#     return model_3d


def affine_weights(model_2d, model_3d):
    #pretrained_dict_2d = model_2d.state_dict()
    pretrained_dict_2d = model_2d#.state_dict()
    pretrained_dict_2d['fc.weight']=pretrained_dict_2d.pop('weight')
    pretrained_dict_2d['fc.bias']=pretrained_dict_2d.pop('bias')
    model_dict_3d = model_3d.state_dict()
    for key,weight_2d in pretrained_dict_2d.items():
        if key in model_dict_3d:
            if 'conv' in key:
                #time_kernel_size = model_dict_3d[key].shape[2]
                if 'weight' in key:
                    weight_3d = weight_2d#.unsqueeze(2).repeat(1,1,time_kernel_size,1,1)
                    #weight_3d = weight_3d / time_kernel_size
                    model_dict_3d[key] = weight_3d
                elif 'bias' in key:
                    model_dict_3d[key] = weight_2d
            elif 'bn' in key:
                    model_dict_3d[key] = weight_2d
            elif 'fc' in key:
                if 'weight' in key:
                    weight_3d = weight_2d#.unsqueeze(2).repeat(1,1,time_kernel_size,1,1)
                    #weight_3d = weight_3d / time_kernel_size
                    model_dict_3d[key] = weight_3d
                elif 'bias' in key:
                    model_dict_3d[key] = weight_2d
            elif 'downsample' in key:
                if '0.weight' in key:
                    #time_kernel_size = model_dict_3d[key].shape[2]
                    weight_3d = weight_2d#.unsqueeze(2).repeat(1, 1, time_kernel_size, 1, 1)
                    #weight_3d = weight_3d / time_kernel_size
                    model_dict_3d[key] = weight_3d
                else:
                    model_dict_3d[key] = weight_2d

    model_3d.load_state_dict(model_dict_3d)
    return model_3d.cuda()


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, mode='embedded_gaussian',
                 sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]
        assert mode in ['embedded_gaussian', 'gaussian', 'dot_product', 'concatenation']

        # print('Dimension: %d, mode: %s' % (dimension, mode))

        self.mode = mode
        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool = nn.MaxPool3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool = nn.MaxPool2d
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool = nn.MaxPool1d
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_normal(self.g.weight)
        nn.init.constant(self.g.bias,0)
        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.kaiming_normal(self.W[0].weight)
            nn.init.constant(self.W[0].bias, 0)
            nn.init.constant(self.W[1].weight, 0)
            nn.init.constant(self.W[1].bias, 0)

            
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.kaiming_normal(self.W.weight)
            nn.init.constant(self.W.bias, 0)

        self.theta = None
        self.phi = None

        if mode in ['embedded_gaussian', 'dot_product']:
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                                 kernel_size=1, stride=1, padding=0)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

            if mode == 'embedded_gaussian':
                self.operation_function = self._embedded_gaussian
            else:
                self.operation_function = self._dot_product

        elif mode == 'gaussian':
            self.operation_function = self._gaussian
        else:
            raise NotImplementedError('Mode concatenation has not been implemented.')

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool(kernel_size=2))
            if self.phi is None:
                self.phi = max_pool(kernel_size=2)
            else:
                self.phi = nn.Sequential(self.phi, max_pool(kernel_size=2))

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        output = self.operation_function(x)
        return output

    def _embedded_gaussian(self, x):
        batch_size = x.size(0)

        # g=>(b, c, t, h, w)->(b, 0.5c, t, h, w)->(b, thw, 0.5c)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # theta=>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, thw, 0.5c)
        # phi  =>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, 0.5c, thw)
        # f=>(b, thw, 0.5c)dot(b, 0.5c, twh) = (b, thw, thw)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        # (b, thw, thw)dot(b, thw, 0.5c) = (b, thw, 0.5c)->(b, 0.5c, t, h, w)->(b, c, t, h, w)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _gaussian(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = x.view(batch_size, self.in_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        if self.sub_sample:
            phi_x = self.phi(x).view(batch_size, self.in_channels, -1)
        else:
            phi_x = x.view(batch_size, self.in_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _dot_product(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian', sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, mode=mode,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian', sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, mode=mode,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian', sub_sample=True, bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, mode=mode,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer)

def nonlocalnet(input_layer,input_channel):
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        net = NONLocalBlock3D(in_channels=input_channel,mode='embedded_gaussian')
        out = net(input_layer)
    else:
        net = NONLocalBlock3D(in_channels=input_channel,mode='embedded_gaussian')
        out = net(input_layer)
    return out



if __name__ == '__main__':
    import torchvision
    import numpy as np
    import torch
    from torch.autograd import Variable

    resnet = torchvision.models.resnet101(pretrained=True)
    resnet_i3d = resnet101(pretrained=True)

    data = np.ones((1, 3, 224, 224), dtype=np.float32)
    tensor = torch.from_numpy(data)
    inputs = Variable(tensor)
    out1 = resnet(inputs)
    print(out1)

    data2 = np.ones((1, 3, 32, 224, 224), dtype=np.float32)
    tensor2 = torch.from_numpy(data2)
    inputs2 = Variable(tensor2)
    inputs2 = inputs2.cuda()
    out2 = resnet_i3d(inputs2)
    print(out2)
