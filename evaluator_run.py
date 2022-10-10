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

from batchgenerators.utilities.file_and_folder_operations import *
import os
import numpy as np
import nibabel as nib
import argparse
from evaluator import evaluate_folder
from time import process_time


def check_shape2(folder_with_predictions, folder_with_gts):
    list_p = subfiles(folder_with_predictions, sort=True)
    list_gts = subfiles(folder_with_gts, sort=True)
    list_p_shp = []
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
            list_p_shp.append(img)
    if len(list_p_shp) != 0:
        print('Number of non-matching files: ' + str(fls))
        print('The shapes of these files do not match: ' + list_p_shp)
        print('Please check if all ground truths have a corresponding segementation')
    else:
        print('All good. The shapes of all files match!')

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
    parser.add_argument("folder_with_gts",
                        help="Path to folder with the ground truth segmentations in .nii.gz format"
                             "The filename has to be the same as the filename of the corresponding segmentation.")
    parser.add_argument("folder_with_predictions",
                        help="Path to folder with the segmentation you would like to compare to the ground truth segmentations. "
                             "Must be in in .nii.gz format."
                             "The filename has to be the same as the filename of the corresponding gt.")
    parser.add_argument("-number_classes", required=False, default=1,
                        help="number of segmentation classes including background class, for example '2' for binary"
                             "segmentation task")
    parser.add_argument("-threshold", required=False, default=False, type=int,
                        help="If threshold necessary. By default False. If integer is used selects all cases where below"
                        "the threshold where the metric is not defined and runs detection task analysis.")
    parser.add_argument("-hidden", required=False, default=True, action='store_true',
                        help="removes all hidden files in input folders")
    parser.add_argument("-check", required=False, default=True, action='store_true',
                        help="checks gt and prediction/segmentations for same shape")

    args = parser.parse_args()

    # input folders
    folder_with_gts = args.folder_with_gts
    folder_with_predictions = args.folder_with_predictions

    # threshold
    threshold = args.threshold
    if isinstance(threshold, int):
        th = threshold
        print(f'I use a threshold of {th} ml. Below this threshold maks are considered empty. The evaluation of a '
              f'detection task will be initialized.')
    else:
        th = None
    # segmentation classes
    classes = tuple(range(int(args.number_classes)))
    print(f'I use classes {classes}')

    # checking for hidden files and dimension agreement
    if args.hidden:
        remove_hidden(folder_with_gts)
        remove_hidden(folder_with_predictions)
    if args.check:
        check_shape2(folder_with_gts, folder_with_predictions)
    # check_spacing2(folder_with_gts, folder_with_predictions)

    ################# test begin ###################
    # evaluate_folder("Test_files_gt", "Test_files_segmentation", 1, (0,1))
    ################# test end ###################

    # run
    time_start = process_time()
    evaluate_folder(folder_with_gts, folder_with_predictions, th, classes)
    time_end = process_time()
    print(f'Time needed: {round(time_end-time_start,4)} sec')
    print('Done')

