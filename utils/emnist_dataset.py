from torch.utils.data import Dataset
import json
import os
from collections import OrderedDict
import numpy as np

from IPython import embed

class EMNISTDataset_by_write(Dataset):
    """Face Landmarks dataset."""

    _image_list = []
    _image_idx_by_writer = {}
    _image_idx_by_class = {}
    _shuffle_idx = None
    _reverse_shuffle_idx = None
    _train_test_split = 0.8
    _data_usage = 1.00
    _writer_name_by_idx = []
    _writer_name_list = []
    _idx_fmap_train = []
    _idx_fmap_test = []
    _raw_data_root = '../leaf/data/femnist/data/all_data'
    _total_num_users = 3500

    def __init__(self, root='../leaf/data/femnist/data/all_data', train=True, transform=None, train_test_split=0.8, data_usage=1.00):
        if not EMNISTDataset_by_write._image_list:
            EMNISTDataset_by_write._raw_data_root = root
            EMNISTDataset_by_write._train_test_split = train_test_split
            EMNISTDataset_by_write._data_usage = data_usage
            files = os.listdir(EMNISTDataset_by_write._raw_data_root)
            files = [f for f in files if f.endswith('.json')]
            for f in files:
                file_dir = os.path.join(EMNISTDataset_by_write._raw_data_root, f)
                with open(file_dir, 'r') as inf:
                    # Load data into an OrderedDict, to prevent ordering changes
                    # and enable reproducibility
                    data = json.load(inf, object_pairs_hook=OrderedDict)
                for writer, im_c in data['user_data'].items():
                    for image, target in zip(im_c['x'], im_c['y']):
                        # Without astype('float32'), it becomes torch.float64 at the Pytorch model input, and it throws an error.
                        image = np.asarray(image).reshape(28,28).astype('float32')
                        if not writer in EMNISTDataset_by_write._image_idx_by_writer:
                            EMNISTDataset_by_write._image_idx_by_writer[writer] = []
                            EMNISTDataset_by_write._writer_name_by_idx.append(writer)
                        if not target in EMNISTDataset_by_write._image_idx_by_class:
                            EMNISTDataset_by_write._image_idx_by_class[target] = []
                        writer_idx = EMNISTDataset_by_write._writer_name_by_idx.index(writer)
                        EMNISTDataset_by_write._image_list.append((image, target, writer_idx))
                        image_idx = len(EMNISTDataset_by_write._image_list) - 1
                        EMNISTDataset_by_write._image_idx_by_writer[writer].append(image_idx)
                        EMNISTDataset_by_write._image_idx_by_class[target].append(image_idx)
            EMNISTDataset_by_write._writer_name_list = list(set(EMNISTDataset_by_write._writer_name_by_idx))

        if EMNISTDataset_by_write._shuffle_idx is None:
            EMNISTDataset_by_write._shuffle_idx = np.arange(len(EMNISTDataset_by_write._image_list))
            np.random.shuffle(EMNISTDataset_by_write._shuffle_idx)
            # EMNISTDataset_by_write._reverse_shuffle_idx = np.argsort(EMNISTDataset_by_write._shuffle_idx)
            end_at = int((len(EMNISTDataset_by_write._image_list)-1) * EMNISTDataset_by_write._data_usage)
            split_at = int(end_at * EMNISTDataset_by_write._train_test_split)
            EMNISTDataset_by_write._idx_fmap_train = EMNISTDataset_by_write._shuffle_idx[:split_at]
            EMNISTDataset_by_write._idx_fmap_test = EMNISTDataset_by_write._shuffle_idx[split_at:end_at]            

        self.transform = transform
        if train:
            self._idx_fmap = list(EMNISTDataset_by_write._idx_fmap_train)
        else:
            self._idx_fmap = list(EMNISTDataset_by_write._idx_fmap_test)

        self.data = np.asarray([EMNISTDataset_by_write._image_list[idx][0] for idx in self._idx_fmap])
        self.targets = np.asarray([EMNISTDataset_by_write._image_list[idx][1] for idx in self._idx_fmap])
        self.writers = np.asarray([EMNISTDataset_by_write._image_list[idx][2] for idx in self._idx_fmap])
        # self.classes = 
        # self.class_to_idx = 
        self.dict_users = [] # This is for client training
        if train:
            for writer_idx in range(EMNISTDataset_by_write._total_num_users):
                self.dict_users.append(list(np.where(self.writers == writer_idx)[0]))

    def __getitem__(self, index):
        # stuff
        image, target, writer_idx = self.data[index], self.targets[index], self.writers[index]
        if self.transform:
            image = self.transform(image)
        return (image, target)

    def __len__(self):
        return len(self._idx_fmap) # of how many examples(images?) you have


