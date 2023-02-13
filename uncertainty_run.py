from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
import SimpleITK as sitk
from statistics import fmean,median
from scipy.stats import entropy
from evaluator import make_bootstrap
import glob
import os
from tqdm import tqdm
import argparse

import matplotlib.pyplot as plt


def remove_hidden(path_images):
    list_images_hidden = subfiles(path_images, prefix='.', join=True)
    for i in subfiles(path_images, suffix='.nii', join=True):
        if i not in list_images_hidden:
            list_images_hidden.append(i)
    for i in list_images_hidden:
        if isfile(i):
            os.remove(i)

def plot_histogramm(sum_array):
    _ = plt.hist(sum_array, bins=100)
    plt.axvline(x = median(sum_array), color = 'b')# arguments are passed to np.histogram
    plt.axvline(x = fmean(sum_array), color = 'r')
    plt.title("Histogram with 'auto' bins")
    plt.show()

def my_entropy(labels, base=2):
    list_ent = []
    for i in range(np.shape(labels)[1]):
        if np.sum(labels[:, i, :, :]) == 0:
            continue
        for a in range(np.shape(labels)[2]):
            if np.sum(labels[:, i, a, :]) == 0:
                continue
            for b in range(np.shape(labels)[3]):
                if np.sum(labels[:,i,a,b]) == 0:
                    continue
                else:
                    value, count = np.unique(labels[:,i,a,b], return_counts=True)
                    ent = entropy(count, base=base)
                    list_ent.append(ent)
    return fmean(list_ent)
def disagreement(dirs: list):
    dict = {}

    number_of_raters = len(dirs)

    for i in range(number_of_raters):
        dict[i] = [i for i in subfiles(dirs[i], join=True, suffix="nii.gz", sort=True)]

    cases = [[value[idx] for key, value in dict.items()] for idx in range(len(dict[0]))]

    reps = 1000
    bootstrap_list = [list(np.random.choice(range(len(cases)), size=len(list(cases)), replace=True)) for _ in range(reps)]

    mean_mean = []
    case = 0
    number_cases = len(cases)
    for case in range(number_cases):
        case_name = cases[case][0].rsplit("/",1)[-1].rsplit(".")[0]

        # check labels of raters per case to have the same shape
        shape_labels = [np.shape(sitk.GetArrayFromImage(sitk.ReadImage(label))) for label in cases[case]]
        assert all(x == shape_labels[0] for x in shape_labels), print("for case: ", case_name, "the shapes do not match: ", shape_labels)

        # comput the variance of all labels
        array_labels = np.stack([sitk.GetArrayFromImage(sitk.ReadImage(rater)) for rater in cases[case]],axis=0)

        entropy = my_entropy(array_labels)

        mean_mean.append(entropy)
        case += 1
        print(f"Running average after {case}/{number_cases} of uncertainty: ", fmean(mean_mean))

    plot_histogramm(mean_mean)

    # calculate 95% confidence interval
    CI = np.std(make_bootstrap(mean_mean,median, bootstrap_list))*1.96

    print("The final median uncertainty is: ", fmean(mean_mean), "±", CI)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("folder_with_experts_annotations",
                        help="Path to directory with the directories for each of the raters segmentations. "
                             "Please make sure the directory names start with a specific prefix that can be specified")
    parser.add_argument("prefix",
                        help="Path to directory with the directories for each of the raters segmentations. "
                             "Please make sure the directory names start with a specific prefix that can be specified")

    args = parser.parse_args()

    prefix = args.prefix
    folders_of_raters = glob.glob(f"{args.folder_with_experts_annotations}/{prefix}*")

    disagreement(dirs=folders_of_raters)
