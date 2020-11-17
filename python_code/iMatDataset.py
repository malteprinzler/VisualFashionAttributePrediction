from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import re
from torch._six import container_abcs, string_classes, int_classes
import numpy as np
from torch.utils.data._utils.collate import *
import io
import os
import sys
import json
from time import time
import urllib3
import multiprocessing

from PIL import Image
from tqdm import tqdm
from urllib3.util import Retry

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import torch
import torchvision
from torch.utils.data.dataset import Dataset
import json
import os
from PIL import Image
from torchvision.transforms.transforms import *
import matplotlib.pyplot as plt
import csv
import numpy as np
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from argparse import ArgumentParser


class iMatDataModule(pl.LightningDataModule):
    """
    Pytorch Lightning DataModule for Dataset from iMaterialist Challenge (Fashion) at FGVC5 at Kaggle
    https://www.kaggle.com/c/imaterialist-challenge-fashion-2018/overview/evaluation

    Assumes that you have downloaded the provided json files and unzipped them into a common directory. Also assumes,
    that you have put the label name file (can be found here: https://storage.googleapis.com/kaggle-forum-message-attachments/527517/13166/iMat_fashion_2018_label_map_228.csv)
    in the same directory and named it: 'iMat_fashion_2018_label_map_228.csv'
    """

    def __init__(self, data_root, batch_size=10, image_augmentations=list(), dataset_ratio=1.,
                 train_filename="train.json",
                 val_filename="validation.json", attr_descr_filename="iMat_fashion_2018_label_map_228.csv",
                 img_height=512, img_width=512, num_workers=4, **kwargs):
        """
        :param data_root: directory containing the json files
        :param image_augmentations: list of torchvision PIL image transformations to apply to the images during training
        :param img_height: height of images -> resizes raw images to img_height. If None -> doesn't resize but dataloader
                            returns batches with varying dimensions based on biggest image in batch. Smaller images are
                            centered and padded with 0
                            Attention: might change images aspect dataset_ratio (around 70% of raw images are of shape 600x600)
        :param img_width: same as img_height
        """
        super(iMatDataModule, self).__init__()

        # defining file paths and names
        self.root_dir = data_root
        self.train_filename = train_filename
        self.train_img_path = os.path.join(self.root_dir, "train")
        self.val_filename = val_filename
        self.val_img_path = os.path.join(self.root_dir, "validation")
        self.attr_descr_file = attr_descr_filename

        self.image_augmentations = image_augmentations
        self.ratio = dataset_ratio
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width

        self.num_workers = num_workers

    @staticmethod
    def add_dataset_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", type=int, default=10)
        parser.add_argument("--num_workers", type=int, default=4, help="# workers for dataloader")
        parser.add_argument("--data_root", type=str, help="root directory of dataset", required=True)
        parser.add_argument("--train_filename", type=str, help="Name of training set json file", default="train.json")
        parser.add_argument("--val_filename", type=str, help="Name of validation set json file",
                            default="validation.json")
        parser.add_argument("--attr_descr_filename", type=str, help="Name of attribute description file",
                            default="iMat_fashion_2018_label_map_228.csv")
        parser.add_argument("--dataset_ratio", type=float, default=1., help="Which ratio of the dataset to download")
        return parser

    def prepare_data(self, *args, **kwargs):
        """
        downloading all the images if not downloaded yet
        :param args:
        :param kwargs:
        :return:
        """
        # Training Set
        n_train_images = int(self._get_nr_of_samples_in_json(self.train_filename) * self.ratio)
        if not os.path.exists(self.train_img_path) or len(os.listdir(self.train_img_path)) < n_train_images:
            print("need to download train data")
            download_iMaterialistFashion(os.path.join(self.root_dir, self.train_filename), self.train_img_path,
                                         n_train_images)

        # Validation Set
        n_val_images = int(self._get_nr_of_samples_in_json(self.val_filename) * self.ratio)
        if not os.path.exists(self.val_img_path) or len(os.listdir(self.val_img_path)) < n_val_images:
            print("need to download validation data")
            download_iMaterialistFashion(os.path.join(self.root_dir, self.val_filename), self.val_img_path,
                                         n_val_images)

        # length already calculated here so that this time consuming process does not have to take place on every
        # gpu individually
        self.train_length = len(os.listdir(self.train_img_path))
        self.val_length = len(os.listdir(self.val_img_path))

    def setup(self, stage=None):
        """
        Initializing the Datasets. No Test-Set included yet
        :param stage:
        :return:
        """

        self.train_set = iMatDataset(os.path.join(self.root_dir, self.train_filename), self.train_img_path,
                                     os.path.join(self.root_dir, self.attr_descr_file), self.image_augmentations,
                                     img_height=self.img_height, img_width=self.img_width,
                                     length=self.train_length)

        self.val_set = iMatDataset(os.path.join(self.root_dir, self.val_filename), self.val_img_path,
                                   os.path.join(self.root_dir, self.attr_descr_file),
                                   img_height=self.img_height, img_width=self.img_width,
                                   length=self.val_length)

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          # collate_fn=collate_varying_img_sizes,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          # collate_fn=collate_varying_img_sizes,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers
                          )

    def _get_nr_of_samples_in_json(self, fname):
        with open(os.path.join(self.root_dir, fname), "r") as f:
            n = len(json.load(f)["images"])
        return n


class iMatDataset(Dataset):
    """
    Pytorch iMatDataset from iMaterialist Challenge (Fashion) at FGVC5 at Kaggle
    https://www.kaggle.com/c/imaterialist-challenge-fashion-2018/overview/evaluation

    Returns normalized image tensors and OH-encoded label tensors

    Integrates lots of functionalities to convert between normalized tensors and PIL images + converting from OH-Lables
    to Attribute names, classes, indices and so on.

    Automatically resizes raw images to specified img_height and img_width unless explicitly set to None. In this case
    """

    def __init__(self, labels_file, img_dir, attr_descr_file, image_augmentations=list(), img_height=512, img_width=512,
                 length=None, rgb_mean=[0.6765103936195374, 0.6347460150718689, 0.6207206845283508],
                 rgb_std=[0.3283524215221405, 0.33712077140808105, 0.3378842771053314]):
        """
        :param labels_file: path to json file with labels
        :param img_dir: path to directory with image files
        :param attr_descr_file: path to attribute description file
        :param image_augmentations: list of torchvision PIL image transformations to apply to the images during training
        :param img_height: height of images -> resizes raw images to img_height. If None -> doesn't resize but dataloader
                            returns batches with varying dimensions based on biggest image in batch. Smaller images are
                            centered and padded with 0
                            Attention: might change images aspect dataset_ratio (around 70% of raw images are of shape 600x600)
        :param img_width: same as img_height
        """
        super(iMatDataset, self).__init__()

        self.img_dir = img_dir

        # getting labels
        if labels_file and os.path.exists(labels_file):
            with open(labels_file, "r") as f:
                self.labels = json.load(f)["annotations"]
        else:
            self.labels = []

        ###
        # Transformations
        ###
        # Resizing
        if img_height is not None and img_width is not None:
            resize = [Resize((img_height, img_width))]
        else:
            resize = []

        Normalization = Normalize(mean=rgb_mean, std=rgb_std)
        self.trafo_pil2tensor = Compose(image_augmentations + resize + [ToTensor(), Normalization])
        mean_bwd, std_bwd = torch.tensor(rgb_mean, dtype=torch.float), torch.tensor(rgb_std, dtype=torch.float)
        self.trafo_tensor2pil = Compose([
            Normalize(mean=(-mean_bwd / std_bwd).tolist(), std=(1. / std_bwd).tolist()),
            ToPILImage()
        ])

        self.attr_descr = self.get_attribute_description(attr_descr_file)
        self.n_attrs = len(self.attr_descr)
        self.len = (length if length is not None else min(len(self.labels),
                                                          len(os.listdir(self.img_dir)) if self.img_dir else 0))

    def __getitem__(self, item):
        """
        returns image as normalized tensor and ground truth attribute label as OH tensor.
        Attention: OH positions of attributes are attributeID - 1 !!! (attrIDs start from 1 but index from 0)

        :param item:
        :return:
        """
        if item >= self.__len__():
            raise IndexError
        attr_OH = self.AttrIdxList2OH(self.labels[item]["labelId"], n=self.n_attrs)
        img_path = os.path.join(self.img_dir, self.labels[item]["imageId"] + ".jpg")

        # img retrieval
        img_pil = Image.open(img_path)
        img_tensor = self.trafo_pil2tensor(img_pil)

        return img_tensor, attr_OH

    def get_attribute_description(self, attr_descr_file):
        with open(attr_descr_file, "r") as f:
            f.readline()  # skip first line
            reader = csv.reader(f)
            attr_descriptor = np.array(list(reader))  # cols: labelId,taskId,labelName,taskName

        return attr_descriptor

    def __len__(self):
        return self.len

    def get_mean_n_std(self, n):
        # get tensor with rgb values of n images of shape 3 x *
        colors = torch.cat([self.__getitem__(i)[0].flatten(start_dim=1) for i in range(n)], dim=1)
        mean = colors.mean(dim=1).tolist()
        std = colors.std(dim=1).tolist()
        return dict(mean=mean, std=std)

    def attrIdcs2attrNames(self, attr_idc):
        if isinstance(attr_idc, torch.Tensor):
            attr_idc = attr_idc.tolist()
        ret = []
        for attr_idx in attr_idc:
            ret.append(self.attr_descr[self.attr_descr[:, 0] == str(attr_idx)][0][2])
        return ret

    def attrNames2attrIdcs(self, attr_names):
        ret = []
        for attr_name in attr_names:
            ret.append(self.attr_descr[self.attr_descr[:, 2] == str(attr_name)][0][0])
        return ret

    def attrIdcs2GroupNames(self, attr_idc):
        if isinstance(attr_idc, torch.Tensor):
            attr_idc = attr_idc.tolist()
        ret = []
        for attr_idx in attr_idc:
            ret.append(self.attr_descr[self.attr_descr[:, 0] == str(attr_idx)][0][3])
        return ret

    def attrNames2GroupNames(self, attr_names):
        if type(attr_names) == torch.tensor:
            attr_names = attr_names.tolist()
        ret = []
        for attr_name in attr_names:
            ret.append(self.attr_descr[self.attr_descr[:, 2] == str(attr_name)][0][3])
        return ret

    def AttrIdxList2OH(self, idx_list, n):
        ret = torch.zeros(n, dtype=torch.float)
        for idx in idx_list:
            idx = int(idx) - 1  # for explanation of -1 see __getitem__()
            if idx < 0:
                raise ValueError
            ret[idx] = 1
        return ret

    def OH2Idx(self, OH):
        """
        takes torch.Tensor of shape (C) [OH tensor for 1 sample] and returns attribute idx list
        Args:
            OH:

        Returns:

        """
        return torch.where(OH > 0)[0] + 1  # for explanation of +1 see __getitem__()

    def OH2AttrName(self, OH):
        """
        takes OH-tensor of shape C (one sample and returns list of attribute names
        :param OH:
        :return:
        """
        return self.attrIdcs2attrNames(self.OH2Idx(OH))


###
# Download Utils
###


def download_image(fnames_and_urls):
    """
    download image and save its with 100% quality as JPG format
    skip image downloading if image already exists at given path
    :param fnames_and_urls: tuple containing absolute path and url of image
    """
    fname, url = fnames_and_urls
    if not os.path.exists(fname):
        http = urllib3.PoolManager(retries=Retry(connect=3, read=2, redirect=3))
        response = http.request("GET", url)
        image = Image.open(io.BytesIO(response.data))
        image_rgb = image.convert("RGB")
        image_rgb.save(fname, format='JPEG', quality=100)


def parse_dataset(_dataset, _outdir, _max=100000000):
    """
    parse the dataset to create a list of tuple containing absolute path and url of image
    :param _dataset: dataset to parse
    :param _outdir: output directory where data will be saved
    :param _max: maximum images to download (change to download all dataset)
    :return: list of tuple containing absolute path and url of image
    """
    _fnames_urls = []
    with open(_dataset, 'r') as f:
        data = json.load(f)
        for image in data["images"]:
            url = image["url"]
            fname = os.path.join(_outdir, "{}.jpg".format(image["imageId"]))
            _fnames_urls.append((fname, url))
    return _fnames_urls[:_max]


def download_iMaterialistFashion(dataset, outdir, nmax=1000000000):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # parse json dataset file
    fnames_urls = parse_dataset(dataset, outdir)[:nmax]

    # download data
    pool = multiprocessing.Pool(processes=12)
    with tqdm(total=len(fnames_urls)) as progress_bar:
        for _ in pool.imap_unordered(download_image, fnames_urls):
            progress_bar.update(1)


###
# Collate Function, not needed if all images are resized to the same size
###
def collate_varying_img_sizes(batch):
    """If batch includes images with different sizes
    -> doesn't squeeze images but adds neutral padding to smaller images
    -> output size of batch determined by dimension of biggest image"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        ###
        # changed content START
        # adds black padding so that all images have same dimensions
        ###
        unique_shapes = np.unique(np.array([list(elem.shape) for elem in batch]), axis=0)
        if len(unique_shapes) != 1:  # if samples have different dimensions -> add black padding around smaller images
            batch = list(batch)
            max_height = max([elem.shape[-2] for elem in batch])
            max_width = max([elem.shape[-1] for elem in batch])

            for i, elem in enumerate(batch):
                one_size = torch.zeros((3, max_height, max_width), dtype=elem.dtype, device=elem.device)
                h, w = elem.shape[1:]
                u = int((max_height - h) / 2)
                l = int((max_width - w) / 2)
                one_size[:, u:u + h, l:l + w] = elem
                batch[i] = one_size
            batch = tuple(batch)
        ###
        # changed content END
        ###
        return torch.stack(batch, 0, out=out)

    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return collate_varying_img_sizes([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: collate_varying_img_sizes([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(collate_varying_img_sizes(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [collate_varying_img_sizes(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


if __name__ == "__main__":
    image_augmentations = [ColorJitter(brightness=1)]
    dataset = iMatDataset("../Data/iMaterialist/validation.json", "../Data/iMaterialist/validation",
                          "../Data/iMaterialist/iMat_fashion_2018_label_map_228.csv",
                          image_augmentations=image_augmentations)
    # img, att = dataset[0]
    #
    # print(att)
    # plt.imshow(dataset.trafo_tensor2pil(img))
    # plt.show()

    dm = iMatDataModule("../Data/iMaterialist", dataset_ratio=0.001)
    dm.prepare_data()
    dm.setup()

    for batch in dm.train_dataloader():
        x, _ = batch
        img = dm.train_set.trafo_tensor2pil(x[0])
        plt.imshow(img)
        plt.show()
