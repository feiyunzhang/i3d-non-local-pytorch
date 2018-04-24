import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class I3DDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 sample_frames=32, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, train_mode=True, test_clips=10):

        self.root_path = root_path
        self.list_file = list_file
        self.sample_frames = sample_frames
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.train_mode = train_mode
        if not self.train_mode:
            self.num_clips = test_clips

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            img_path = os.path.join(directory, self.image_tmpl.format(idx))
            try:
                return [Image.open(img_path).convert('RGB')]
            except:
                print("Couldn't load image:{}".format(img_path))
                return None
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(directory, self.image_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(directory, self.image_tmpl.format('y', idx))).convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def _sample_indices(self, record):
        """
        :param record: VideoRecord
        :return: list
        """
        expanded_sample_length = self.sample_frames * 4  # in order to drop every other frame
        if record.num_frames >= expanded_sample_length:
            start_pos = randint(record.num_frames - expanded_sample_length + 1)
            offsets = range(start_pos, start_pos + expanded_sample_length, 4)
        elif record.num_frames > self.sample_frames*2:
            start_pos = randint(record.num_frames - self.sample_frames*2 + 1)
            offsets = range(start_pos, start_pos + self.sample_frames*2, 2)
        elif record.num_frames > self.sample_frames:
            start_pos = randint(record.num_frames - self.sample_frames + 1)
            offsets = range(start_pos, start_pos + self.sample_frames, 1)
        else:
            offsets = np.sort(randint(record.num_frames, size=self.sample_frames))

        offsets =[int(v)+1 for v in offsets]  # images are 1-indexed
        return offsets

    def _get_test_indices(self, record):
        tick = (record.num_frames - self.sample_frames*2 + 1) / float(self.num_clips)
        sample_start_pos = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_clips)])
        offsets = []
        for p in sample_start_pos:
            offsets.extend(range(p,p+self.sample_frames*2,2))

        checked_offsets = []
        for f in offsets:
            new_f = int(f) + 1
            if new_f < 1:
                new_f = 1
            elif new_f >= record.num_frames:
                new_f = record.num_frames - 1
            checked_offsets.append(new_f)

        return checked_offsets


    def __getitem__(self, index):
        record = self.video_list[index]

        if self.train_mode:
            segment_indices = self._sample_indices(record)
            process_data, label = self.get(record, segment_indices)
            while process_data is None:
                index = randint(0, len(self.video_list) - 1)
                process_data, label = self.__getitem__(index)
        else:
            segment_indices = self._get_test_indices(record)
            process_data,label = self.get(record, segment_indices)
            if process_data is None:
                raise ValueError('sample indices:', record.path, segment_indices)

        return process_data,label


    def get(self, record, indices):

        images = list()
        for ind in indices:
            seg_img = self._load_image(record.path, ind)
            if seg_img is None:
                return None,None
            images.extend(seg_img)

        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)
