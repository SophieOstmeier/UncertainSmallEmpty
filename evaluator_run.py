#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from batchgenerators.utilities.file_and_folder_operations import*
from collections import OrderedDict
from natsort import natsorted
import os
import numpy as np
import nibabel as nib
import glob
import argparse

from evaluation.evaluator import evaluate_folder

def change_spacing_header(image_file, label_file):
    img0 = nib.load(image_file)
    print(img0.header.get_zooms(),
          "\n")

    img1 = nib.load(label_file)
    print("old:")
    print(img1.header.get_zooms(),
          "\n")

    img1_ = nib.Nifti1Image(img1.get_fdata(), affine=img0.affine, header=img0.header)
    print("new:")
    print(img1_.header.get_zooms(),
          "\n")

    return img1_


def check_shape1(folder):
    list_p = natsorted(os.listdir(folder))
    count = 0
    for img in list_p:
        path_p = os.path.join(folder, img)
        if img == '.DS_Store':
            os.remove(path_p)
        load_p = nib.load(path_p)
        shape_p = load_p.shape
        count += 1
        print(count)
        print(str('original: ' + img))
        print(str(shape_p))

def check_shape2(folder_with_predictions, folder_with_gts):
    list_p = subfiles(folder_with_predictions, prefix='NCCT', sort=True)
    list_gts = subfiles(folder_with_gts, prefix='NCCT', sort=True)
    list_p_shp = []
    list_gts_shp = []
    count = 0
    fls = 0
    for img, lab in zip(list_p, list_gts):
        load_p = nib.load(img)
        load_gts = nib.load(lab)
        shape_p = load_p.shape
        shape_gts = load_gts.shape
        count += 1
        if shape_p != shape_gts:
            print(f'{img}{lab}: {shape_p} and {shape_gts}')
            fls += 1
            # print(count)
            # print(str('original: ' + img), str('prediction: '+lab))
            # print(str(shape_p), str(shape_gts))
            list_p_shp.append(img)
            # list_gts_shp.append(shape_gts)
    if len(list_p_shp) != 0:
        print('Number of non-matching files: ' + str(fls))
        print('The shapes of these files do not match: ' + list_p_shp)
        print('Please check if all ground truths have a corresponding segementation')
    else:
        print('All good. The shapes of all files match!')

def check_spacing2(folder_with_predictions, folder_with_gts):
    list_p = subfiles(folder_with_predictions, prefix='NCCT', sort=True)
    list_gts = subfiles(folder_with_gts, prefix='NCCT', sort=True)
    count = 0
    fls = 0
    for img, lab in zip(list_p, list_gts):
        path_p = os.path.join(folder_with_predictions, img)
        path_gts = os.path.join(folder_with_gts, lab)
        if img == '.DS_Store':
            os.remove(path_p)
        if lab == '.DS_Store':
            os.remove(path_gts)
        load_p = nib.load(path_p)
        load_gts = nib.load(path_gts)
        space_p = load_p.header.get_zooms()
        space_gts = load_gts.header.get_zooms()
        count += 1
        if space_p != space_gts:
            fls += 1
            print(count)
            print(str('original: ' + img), str('prediction: '+lab))
            print(str(space_p) + str(space_gts))
    print('false: ' + str(fls))

def check_volume2(folder_with_predictions, folder_with_gts):
    list_p = natsorted(os.listdir(folder_with_predictions))
    list_gts = natsorted(os.listdir(folder_with_gts))
    count = 0
    fls = 0
    for img, lab in zip(list_p, list_gts):
        path_p = os.path.join(folder_with_predictions, img)
        path_gts = os.path.join(folder_with_gts, lab)
        if img == '.DS_Store':
            os.remove(path_p)
        if lab == '.DS_Store':
            os.remove(path_gts)
        load_p = nib.load(path_p)
        load_gts = nib.load(path_gts)
        num_vox_p = np.sum(load_p.get_fdata() > 0)
        num_vox_gts = np.sum(load_gts.get_fdata() > 0)
        count += 1
        if num_vox_p != num_vox_gts:
            fls += 1
            print(count)
            print(str('original: ' + img), str('prediction: '+lab))
            print(str(num_vox_p)+ '       ' + str(num_vox_gts))
    print('false: ' + str(fls))

    def remove_hidden(path_images):
        list_images_hidden = subfiles(path_images, prefix='.', join=True)
        for i in subfiles(path_images, suffix='.nii', join=True):
            if i not in list_images_hidden:
                list_images_hidden.append(i)
        for i in list_images_hidden:
            if isfile(i):
                os.remove(i)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_with_gts", help="should be your folder with the ground truth segmentation in .nii.gz format")
    parser.add_argument("folder_with_predictions", help="should be your folder with the segmentation you would like to compare to the ground truth segmentation. Must be in in .nii.gz format")
    parser.add_argument("-hidden", help="removes all hidden files in input folders")
    parser.add_argument("-check", help="checks gt and prediction/segmentations for same shape")

    args = parser.parse_args()
    
    # input folders
    folder_with_gts = args.folder_with_gts
    folder_with_predictions = args.folder_with_predictions
    
    # checking for hidden files and dimension agreement 
    if args.hidden:
        remove_hidden(folder_with_gts)
        remove_hidden(folder_with_predictions)
    if args.check:
        check_shape2(folder_with_gts, folder_with_predictions)
    # check_spacing2(folder_with_gts, folder_with_predictions)
    
    #run
    evaluate_folder(folder_with_gts,folder_with_predictions, (0,1))
