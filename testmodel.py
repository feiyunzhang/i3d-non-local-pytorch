import argparse
import time

import numpy as np
import torch.nn.parallel
import torch.optim
import torchvision
from sklearn.metrics import confusion_matrix

from dataset import I3DDataSet
from i3d import I3DModel
from transforms import *

# options
parser = argparse.ArgumentParser(
    description="Standard video-level testing")
parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51', 'kinetics', 'ucf-crime'])
parser.add_argument('modality', type=str, choices=['RGB', 'Flow', 'RGBDiff'])
parser.add_argument('test_list', type=str)
parser.add_argument('weights', type=str)
parser.add_argument('--arch', type=str, default="resnet101")
parser.add_argument('--save_scores', type=str, default=None)
parser.add_argument('--test_clips', type=int, default=10)
parser.add_argument('--sample_frames', type=int, default=32)
parser.add_argument('--test_crops', type=int, default=1)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', type=str, default='')

args = parser.parse_args()


if args.dataset == 'ucf101':
    num_class = 101
elif args.dataset == 'hmdb51':
    num_class = 51
elif args.dataset == 'kinetics':
    num_class = 400
else:
    raise ValueError('Unknown dataset '+args.dataset)

i3d_model = I3DModel(num_class, args.sample_frames, args.modality,
                         base_model=args.arch, dropout=args.dropout)

checkpoint = torch.load(args.weights)
print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
i3d_model.load_state_dict(base_dict)

if args.test_crops == 1:
    cropping = torchvision.transforms.Compose([
        GroupScale(i3d_model.scale_size),
        GroupCenterCrop(i3d_model.input_size),
    ])
elif args.test_crops == 10:
    cropping = torchvision.transforms.Compose([
        GroupOverSample(i3d_model.input_size, i3d_model.scale_size)
    ])
else:
    raise ValueError("Only 1 and 10 crops are supported while we got {}".format(args.test_crops))

data_loader = torch.utils.data.DataLoader(
        I3DDataSet("", args.test_list, sample_frames=args.sample_frames,
                    modality=args.modality,
                    image_tmpl="image_{:05d}.jpg" if args.modality in ['RGB', 'RGBDiff'] else args.flow_prefix+"{}_{:05d}.jpg",
                    train_mode=False, test_clips=args.test_clips,
                    transform=torchvision.transforms.Compose([
                       cropping,
                       Stack(),
                       ToTorchFormatTensor(),
                       GroupNormalize(i3d_model.input_mean, i3d_model.input_std),
                    ])),
        batch_size=1, shuffle=False,
        num_workers=args.workers * 2, pin_memory=True)

if args.gpus is not None:
    devices = [args.gpus[i] for i in range(args.workers)]
else:
    devices = list(range(args.workers))

i3d_model = torch.nn.DataParallel(i3d_model.cuda(devices[0]), device_ids=devices)
i3d_model.eval()

data_gen = enumerate(data_loader)

total_num = len(data_loader.dataset)
output = []

def eval_video(video_data):
    i, data, label = video_data

    if args.modality == 'RGB':
        num_channel = 3
        num_depth = 32
    else:
        raise ValueError("Unknown modality "+args.modality)
    data = data.squeeze(0)
    data = data.view(num_channel,-1,num_depth,data.size(2),data.size(3)).contiguous()
    data = data.permute(1,0,2,3,4).contiguous()
    #data = data.view(data.size(0),num_channel,-1,num_depth,data.size(3),data.size(4))
    #data = data.squeeze(0).permute(1,0,2,3,4).contiguous()
    input_var = torch.autograd.Variable(data, volatile=True)
    rst = i3d_model(input_var).data.cpu().numpy().copy()
    return i, rst.reshape((args.test_clips*args.test_crops, num_class)).mean(axis=0).reshape((1, num_class)), \
           label[0]


proc_start_time = time.time()

for i, (data, label) in data_gen:
    rst = eval_video((i, data, label))
    output.append(rst[1:])
    cnt_time = time.time() - proc_start_time
    print('video {} done, total {}/{}, average {} sec/video'.format(i, i+1,
                                                                    total_num,
                                                                    float(cnt_time) / (i+1)))

video_pred = [np.argmax(x[0]) for x in output]

video_labels = [x[1] for x in output]


cf = confusion_matrix(video_labels, video_pred).astype(float)

cls_cnt = cf.sum(axis=1)
cls_hit = np.diag(cf)

cls_acc = cls_hit / cls_cnt

print(cls_acc)

print('Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))

if args.save_scores is not None:

    # reorder before saving
    name_list = [x.strip().split()[0] for x in open(args.test_list)]

    order_dict = {e:i for i, e in enumerate(sorted(name_list))}

    reorder_output = [None] * len(output)
    reorder_label = [None] * len(output)

    for i in range(len(output)):
        idx = order_dict[name_list[i]]
        reorder_output[idx] = output[i]
        reorder_label[idx] = video_labels[i]

    np.savez(args.save_scores, scores=reorder_output, labels=reorder_label)

