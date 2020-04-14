#-*-coding:utf-8-*-
import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
from glob import glob
import cv2


class AlignedDatasetResized(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.maskroot = opt.maskroot
        self.GTroot = opt.GTroot
        self.dir_A = opt.dataroot

        self.dirfilenames,self.A_paths= sorted(make_dataset(self.dir_A))


        assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]
        self.transform_img = transforms.Compose(transform_list)
        # MEAN = [0.485, 0.456, 0.406]
        # STD = [0.229, 0.224, 0.225]
        size = (32, 32) #2020-03-15
        # self.img_tf = transforms.Compose([transforms.Resize(size=size), transforms.ToTensor(),
        #      transforms.Normalize(mean=MEAN, std=STD)])


        # <editor-fold desc="GZB add mask ">
        self.mask_paths = glob('{:s}/*.jpg'.format(opt.maskroot))
        self.N_mask = len(self.mask_paths)

        self.transform_mask = transforms.Compose([transforms.Resize(size=size), transforms.ToTensor()])
        #gt data
        self.GT_paths = glob('{:s}/*.jpg'.format(opt.GTroot))
        self.N_GT = len(self.GT_paths)
        transform_list2 = [transforms.Resize(size=size),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]
        self.transform_GT = transforms.Compose(transform_list2)
        # </editor-fold>

    def __getitem__(self, index):
        A_path = self.dirfilenames[index]
        A = Image.open(A_path).convert('RGB')
        A = A.resize((self.opt.fineSize, self.opt.fineSize), Image.BICUBIC)
        A = self.transform_img(A)
    
        filepath, tempfilename = os.path.split(A_path)
        A_mask = filepath.split('/')[-1]
        A_mask_img_path = self.maskroot + '/' + A_mask + '.jpg'
        A_mask_img = Image.open(A_mask_img_path).convert('RGB')
        mask = self.transform_mask(A_mask_img)
    
        # GT_lists_path=[]
        A_tempfilename = A_path.split('/')[-1].split('.')[0].split('_')[-1]
    
        imagename = self.GTroot + '/' + A_tempfilename + '.jpg'
    
        GT_image = Image.open(imagename).convert('RGB')
        GT_images = self.transform_GT(GT_image)
        return {'input_img': A, 'mask': mask, 'input_img_paths': A_path, 'GTimg':GT_images,'mask_path':A_mask_img_path}

    def __len__(self):
        #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        #print(len(self.A_paths))
        #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        return len(self.dirfilenames)

    def name(self):
        return 'AlignedDatasetResized'