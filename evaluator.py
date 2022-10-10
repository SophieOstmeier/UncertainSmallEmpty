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


import collections
import inspect
import json
import hashlib
from datetime import datetime
from multiprocessing.pool import Pool
import numpy as np
import pandas as pd
import SimpleITK as sitk
from metrics import ConfusionMatrix, ALL_METRICS
from batchgenerators.utilities.file_and_folder_operations import save_json, subfiles, join
from flatten_dict import flatten
from collections import OrderedDict



class Evaluator:
    """Object that holds test and reference segmentations with label information
    and computes a number of metrics on the two. 'labels' must either be an
    iterable of numeric values (or tuples thereof) or a dictionary with string
    names and numeric values.
    """

    default_metrics = [
        "Dice",
        "Surface Dice at Tolerance 0mm",
        "Surface Dice at Tolerance 5mm",
        "Surface Dice at Tolerance 10mm",
        "Hausdorff Distance 95",
        "Precision",
        "Recall",
        "Avg. Surface Distance",
        "Total Positives Test",
        "Total Positives Reference",
        "Volume Reference",
        "Volume Test",
        "Volume Absolute Difference",
        "Volume Relative Difference",
        "Volumetric Similarity",
]

    default_advanced_metrics = [
    ]

    default_detection = [
        "Detection TN",
        "Detection TP",
        "Detection FN",
        "Detection FP"
    ]

    def __init__(self,
                 test=None,
                 reference=None,
                 labels=None,
                 metrics=None,
                 advanced_metrics=None,
                 threshold=None,
                 nan_for_nonexisting=True):

        self.threshold = None
        self.test = None
        self.reference = None
        self.confusion_matrix = ConfusionMatrix()
        self.labels = None
        self.nan_for_nonexisting = nan_for_nonexisting
        self.result = None

        self.metrics = []
        if metrics is None:
            for m in self.default_metrics:
                self.metrics.append(m)
        else:
            for m in metrics:
                self.metrics.append(m)

        self.advanced_metrics = []
        if advanced_metrics is None:
            for m in self.default_advanced_metrics:
                self.advanced_metrics.append(m)
        else:
            for m in advanced_metrics:
                self.advanced_metrics.append(m)


        if threshold is not None:
            self.set_threshold(threshold)

        # else:
        #      for m in detection:
        #          self.detection_metric.append(m)

        self.set_reference(reference)
        self.set_test(test)

        if labels is not None:
            self.set_labels(labels)
        else:
            if test is not None and reference is not None:
                self.construct_labels()

    def set_test(self, test):
        """Set the test segmentation."""

        self.test = test

    def set_reference(self, reference):
        """Set the reference segmentation."""

        self.reference = reference

    def set_labels(self, labels):
        """Set the labels.
        :param labels= may be a dictionary (int->str), a set (of ints), a tuple (of ints) or a list (of ints). Labels
        will only have names if you pass a dictionary"""

        if isinstance(labels, dict):
            self.labels = collections.OrderedDict(labels)
        elif isinstance(labels, set):
            self.labels = list(labels)
        elif isinstance(labels, np.ndarray):
            self.labels = [i for i in labels]
        elif isinstance(labels, (list, tuple)):
            self.labels = labels
        else:
            raise TypeError("Can only handle dict, list, tuple, set & numpy array, but input is of type {}".format(type(labels)))
    
    def set_threshold(self, threshold):
        """Set the threshold.
        :param threshold= integer in ml to switch to detection task"""

        if isinstance(threshold, int):
            self.threshold = threshold
            self.detection = True
        else:
            raise TypeError("Can integer but input is of type {}".format(type(threshold)))
    
    def construct_labels(self):
        """Construct label set from unique entries in segmentations."""

        if self.test is None and self.reference is None:
            raise ValueError("No test or reference segmentations.")
        else:
        ######################
            labels = np.union1d(np.unique(self.test),
                                np.unique(self.reference))
        self.labels = list(map(lambda x: int(x), labels))
        ######################

    def set_metrics(self, metrics):
        """Set evaluation metrics"""

        if isinstance(metrics, set):
            self.metrics = list(metrics)
        elif isinstance(metrics, (list, tuple, np.ndarray)):
            self.metrics = metrics
        else:
            raise TypeError("Can only handle list, tuple, set & numpy array, but input is of type {}".format(type(metrics)))

    def add_metric(self, metric):

        if metric not in self.metrics:
            self.metrics.append(metric)

    def evaluate(self, test=None, reference=None,threshold=None, advanced=False, **metric_kwargs):
        """Compute metrics for segmentations."""
        if test is not None:
            self.set_test(test)

        if reference is not None:
            self.set_reference(reference)

        if threshold is not None:
            self.set_threshold(threshold)

        if self.test is None or self.reference is None:
            raise ValueError("Need both test, reference segmentations.")

        if self.labels is None:
            self.construct_labels()

        self.metrics.sort()
        # get functions for evaluation
        # somewhat convoluted, but allows users to define additional metrics
        # on the fly, e.g. inside an IPython console
        _funcs = {m: ALL_METRICS[m] for m in self.metrics + self.advanced_metrics+ self.default_detection}
        frames = inspect.getouterframes(inspect.currentframe())
        for metric in self.metrics:
            for f in frames:
                if metric in f[0].f_locals:
                    _funcs[metric] = f[0].f_locals[metric]
                    break
            else:
                if metric in _funcs:
                    continue
                else:
                    raise NotImplementedError(
                        "Metric {} not implemented.".format(metric))

        # get results
        self.result = OrderedDict()

        eval_metrics = self.metrics
        if advanced:
            eval_metrics += self.advanced_metrics
        if isinstance(self.threshold, int):
            eval_metrics += self.default_detection

        if isinstance(self.labels, dict):

            for label, name in self.labels.items():
                k = str(name)
                self.result[k] = OrderedDict()
                if not hasattr(label, "__iter__"):
                    self.confusion_matrix.set_test(self.test == label)
                    self.confusion_matrix.set_reference(self.reference == label)
                else:
                    current_test = 0
                    current_reference = 0
                    for l in label:
                        current_test += (self.test == l)
                        current_reference += (self.reference == l)
                    self.confusion_matrix.set_test(current_test)
                    self.confusion_matrix.set_reference(current_reference)
                for metric in eval_metrics:
                    self.result[k][metric] = _funcs[metric](confusion_matrix=self.confusion_matrix,
                                                               nan_for_nonexisting=self.nan_for_nonexisting,
                                                               **metric_kwargs)

        else:

            for i, l in enumerate(self.labels):
                k = str(l)
                self.result[k] = OrderedDict()
                self.confusion_matrix.set_test(self.test == l)
                self.confusion_matrix.set_reference(self.reference == l)
                for metric in eval_metrics:
                    self.result[k][metric] = _funcs[metric](confusion_matrix=self.confusion_matrix,
                                                            nan_for_nonexisting=self.nan_for_nonexisting,
                                                            threshold=self.threshold,
                                                            **metric_kwargs)

        return self.result

    def to_dict(self):

        if self.result is None:
            self.evaluate()
        return self.result

    def to_array(self):
        """Return result as numpy array (labels x metrics)."""

        if self.result is None:
            self.evaluate

        result_metrics = sorted(self.result[list(self.result.keys())[0]].keys())

        a = np.zeros((len(self.labels), len(result_metrics)), dtype=np.float32)

        if isinstance(self.labels, dict):
            for i, label in enumerate(self.labels.keys()):
                for j, metric in enumerate(result_metrics):
                    a[i][j] = self.result[self.labels[label]][metric]
        else:
            for i, label in enumerate(self.labels):
                for j, metric in enumerate(result_metrics):
                    a[i][j] = self.result[label][metric]

        return a

    def to_pandas(self):
        """Return result as pandas DataFrame."""

        a = self.to_array()

        if isinstance(self.labels, dict):
            labels = list(self.labels.values())
        else:
            labels = self.labels

        result_metrics = sorted(self.result[list(self.result.keys())[0]].keys())

        return pd.DataFrame(a, index=labels, columns=result_metrics)


class NiftiEvaluator(Evaluator):

    def __init__(self, *args, **kwargs):

        self.test_nifti = None
        self.reference_nifti = None
        super(NiftiEvaluator, self).__init__(*args, **kwargs)

    def set_test(self, test):
        """Set the test segmentation."""

        if test is not None:
            self.test_nifti = sitk.ReadImage(test)
            super(NiftiEvaluator, self).set_test(sitk.GetArrayFromImage(self.test_nifti))
        else:
            self.test_nifti = None
            super(NiftiEvaluator, self).set_test(test)

    def set_reference(self, reference):
        """Set the reference segmentation."""

        if reference is not None:
            self.reference_nifti = sitk.ReadImage(reference)
            super(NiftiEvaluator, self).set_reference(sitk.GetArrayFromImage(self.reference_nifti))
        else:
            self.reference_nifti = None
            super(NiftiEvaluator, self).set_reference(reference)

    def evaluate(self, test=None, reference=None, voxel_spacing=None, **metric_kwargs):

        if voxel_spacing is None:
            voxel_spacing = np.array(self.test_nifti.GetSpacing())[::-1]
            metric_kwargs["voxel_spacing"] = voxel_spacing

        return super(NiftiEvaluator, self).evaluate(test, reference, **metric_kwargs)

def run_evaluation(args):
    test, ref, evaluator, metric_kwargs = args
    # evaluate
    evaluator.set_test(test)
    evaluator.set_reference(ref)
    if evaluator.labels is None:
        evaluator.construct_labels()
    current_scores = evaluator.evaluate(**metric_kwargs)
    if type(test) == str:
        current_scores["test"] = test
    if type(ref) == str:
        current_scores["reference"] = ref
    return current_scores

def format_dict_for_excel(dict_scores):
    list_cases = []
    for case in dict_scores: # cases
        flatten_dict = flatten(case)
        list_cases.append(flatten_dict)
    return list_cases

def aggregate_scores(test_ref_pair,
                     threshold=None,
                     labels=None,
                     evaluator=NiftiEvaluator,
                     nanmean=True,
                     json_output_file=None,
                     excel_output_file=None,
                     json_name="",
                     json_description="",
                     json_author="Sophie",
                     json_task="",
                     num_threads=2,
                     **metric_kwargs):
    """
    test = predicted image
    :param threshold: in ml for detection task
    :param test_ref_triple:
    :param evaluator:
    :param labels: must be a dict of int-> str or a list of int
    :param nanmean:
    :param json_output_file:
    :param json_name:
    :param json_description:
    :param json_author:
    :param json_task:
    :param metric_kwargs:
    :return:
    """

    if type(evaluator) == type:
        evaluator = evaluator()

    if labels is not None:
        evaluator.set_labels(labels)
    
    if threshold is not None:
        evaluator.set_threshold(threshold)

    detection_scores = evaluator.default_detection

    all_scores = OrderedDict()
    all_scores["all"] = []
    all_scores["mean"] = OrderedDict()
    all_scores["median"] = OrderedDict()
    all_scores["detection"] = OrderedDict()

    test = [i[0] for i in test_ref_pair]
    ref = [i[1] for i in test_ref_pair]
    p = Pool(num_threads)
    all_res = p.map(run_evaluation, zip(test, ref, [evaluator]*len(ref), [metric_kwargs]*len(ref)))
    p.close()
    p.join()

    for i in range(len(all_res)):
        all_scores["all"].append(all_res[i])

        # append score list for median
        for label, score_dict in all_res[i].items():
            if label in ("test", "reference"):
                continue
            if label not in all_scores["mean"]:
                all_scores["mean"][label] = OrderedDict()
            for score, value in score_dict.items():
                if score not in detection_scores:
                    if score not in all_scores["mean"][label]:
                        all_scores["mean"][label][score] = []
                    all_scores["mean"][label][score].append(value)

        for label, score_dict in all_res[i].items():
            if label in ("test", "reference"):
                continue
            if label not in all_scores["median"]:
                all_scores["median"][label] = OrderedDict()
            for score, value in score_dict.items():
                if score not in detection_scores:
                    if score not in all_scores["median"][label]:
                        all_scores["median"][label][score] = []
                    all_scores["median"][label][score].append(value)

        for label, score_dict in all_res[i].items():
            if label in ("test", "reference"):
                continue
            if label not in all_scores["detection"]:
                all_scores["detection"][label] = OrderedDict()
            for score, value in score_dict.items():
                if score in detection_scores:
                    if score not in all_scores["detection"][label]:
                        all_scores["detection"][label][score] = []
                    all_scores["detection"][label][score].append(value)

    for label in all_scores["mean"]:
        for score in all_scores["mean"][label]:
            if nanmean:
                all_scores["mean"][label][score] = float(np.nanmean(all_scores["mean"][label][score]))
            else:
                all_scores["mean"][label][score] = float(np.mean(all_scores["mean"][label][score]))

    for label in all_scores["median"]:
        for score in all_scores["median"][label]:
            if nanmean:
                all_scores["median"][label][score] = float(np.nanmedian(all_scores["median"][label][score]))
            else:
                all_scores["median"][label][score] = float(np.median(all_scores["median"][label][score]))

    for label in all_scores["detection"]:
        for score in all_scores["detection"][label]:
            if nanmean:
                all_scores["detection"][label][score] = float(np.nansum(all_scores["detection"][label][score]))
            else:
                all_scores["detection"][label][score] = float(np.sum(all_scores["detection"][label][score]))

    # calculate image classification metric
    for label in all_scores["detection"]:
        tp = float(all_scores["detection"][label]["Detection TP"])
        tn = float(all_scores["detection"][label]["Detection TN"])
        fp = float(all_scores["detection"][label]["Detection FP"])
        fn = float(all_scores["detection"][label]["Detection FN"])
        # positive reference cases
        all_scores["detection"][label]["Positive reference cases"] = tp+fn
        # negative reference cases
        all_scores["detection"][label]["Negative reference cases"] = tn+fp
        # calculate sensitivity
        all_scores["detection"][label]["Detection Sensitivity/Recall"] = tp/(tp+fn+1e-8)
        # calculate kappa
        all_scores["detection"][label]["Detection Precision"] = tp/(tp+fp+1e-8)
        # calculate specificity
        all_scores["detection"][label]["Detection Specificity"] = tn/(tn+fp+1e-8)

    # save to file if desired
    # we create a hopefully unique id by hashing the entire output dictionary
    if json_output_file is not None:
        json_dict = OrderedDict()
        json_dict["name"] = json_name
        json_dict["description"] = json_description
        timestamp = datetime.today()
        json_dict["timestamp"] = str(timestamp)
        json_dict["task"] = json_task
        json_dict["author"] = json_author
        json_dict["results"] = all_scores
        json_dict["id"] = hashlib.md5(json.dumps(json_dict).encode("utf-8")).hexdigest()[:12]
        save_json(json_dict, json_output_file)
        df1 = pd.DataFrame(format_dict_for_excel(all_scores["all"]))
        df2 = pd.DataFrame(all_scores["mean"])
        df3 = pd.DataFrame(all_scores["median"])
        df4 = pd.DataFrame(all_scores["detection"])
        with pd.ExcelWriter(excel_output_file) as writer:
            df1.to_excel(writer, sheet_name = 'all' )
            df2.to_excel(writer, sheet_name = 'mean')
            df3.to_excel(writer, sheet_name = 'median')
            df4.to_excel(writer, sheet_name = 'detection')

    return all_scores


def evaluate_folder(folder_with_gts: str, folder_with_predictions: str,th: int, labels: tuple, **metric_kwargs):
    """
    writes a summary.json to folder_with_predictions
    :param folder_with_gts: folder where the ground truth segmentations are saved. Must be nifti files.
    :param folder_with_predictions: folder where the predicted segmentations are saved. Must be nifti files.
    :param labels: tuple of int with the labels in the dataset. For example (0, 1, 2, 3) for Task001_BrainTumour.
    :return:
    """
    if isinstance(th, int):
        threshold = th
    else:
        threshold = None

    files_gt_shape = subfiles(folder_with_gts,suffix=".nii.gz", join=True, sort=True)
    files_pred_shape = subfiles(folder_with_predictions, suffix=".nii.gz", join=True, sort=True)
    for i, a in zip(files_gt_shape,files_pred_shape):
        shp_gt = sitk.ReadImage(i).GetSize()
        shp_pred = sitk.ReadImage(a).GetSize()
        if shp_gt != shp_pred:
            print(f'Shape mismatch: shape_gt {i}: {shp_gt} spape_pred {a}: {shp_pred}')
    files_gt = subfiles(folder_with_gts,suffix=".nii.gz", join=False, sort=True)
    files_pred = subfiles(folder_with_predictions, suffix=".nii.gz", join=False, sort=True)
    assert all([i in files_pred for i in files_gt]), "files missing in folder_with_predictions"
    assert all([i in files_gt for i in files_pred]), "files missing in folder_with_gts"
    test_ref_pair = [(join(folder_with_predictions, i), join(folder_with_gts, i)) for i in files_pred]
    res = aggregate_scores(test_ref_pair, threshold=threshold,
                           json_output_file=join(folder_with_predictions, "summary.json"),
                           excel_output_file=join(folder_with_predictions, "summary.xlsx"),
                           num_threads=8, labels=labels, **metric_kwargs)
    return res

