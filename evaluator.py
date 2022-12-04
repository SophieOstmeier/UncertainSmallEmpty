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



from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from metrics import ConfusionMatrix, ALL_METRICS
import collections
import itertools
import inspect
import json
import hashlib
from datetime import datetime
from multiprocessing.pool import Pool
import numpy as np
import pandas as pd
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import save_json, subfiles, join
from flatten_dict import flatten

import time

timing_test = True

class Evaluator:
    """Object that holds test and reference segmentations with label information
    and computes a number of metrics on the two. 'labels' must either be an
    iterable of numeric values (or tuples thereof) or a dictionary with string
    names and numeric values.
    """

    default_metrics = [
        "Dice",
        # "False Discovery Rate",
        # "False Omission Rate",
        # "Surface Dice at Tolerance 0mm",
        "Surface Dice at Tolerance 2mm",
        "Surface Dice at Tolerance 5mm",
        # "Surface Dice at Tolerance 10mm",
        "Hausdorff Distance 95",
        "Precision",
        "Recall",
        "Avg. Surface Distance",
        "Total Positives Test",
        "Total Positives Reference",
        "Volume Reference",
        "Volume Test",
        "Volume Absolute Difference",
        # "Volume Relative Difference",
        "Volumetric Similarity",
    ]

    default_advanced_metrics = [
    ]

    default_detection = [
        "Image-level TN",
        "Image-level TP",
        "Image-level FN",
        "Image-level FP",
        "CCR",
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
            raise TypeError(
                "Can only handle dict, list, tuple, set & numpy array, but input is of type {}".format(type(labels)))

    def set_threshold(self, threshold):
        """Set the threshold.
        :param threshold= float in ml to switch to detection task"""

        if isinstance(threshold, float):
            self.threshold = threshold
            self.detection = True
        else:
            raise TypeError("Can float but input is of type {}".format(type(threshold)))

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
            raise TypeError(
                "Can only handle list, tuple, set & numpy array, but input is of type {}".format(type(metrics)))

    def add_metric(self, metric):

        if metric not in self.metrics:
            self.metrics.append(metric)

    def evaluate(self, test=None, reference=None, threshold=None, voxel_spacing=None, advanced=False, **metric_kwargs):
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
        _funcs = {m: ALL_METRICS[m] for m in self.metrics + self.advanced_metrics + self.default_detection}
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
        if isinstance(self.threshold, float):
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
                    self.confusion_matrix.set_threshold(self.threshold)
                    self.confusion_matrix.set_voxel_spacing(voxel_spacing)
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
                self.confusion_matrix.set_threshold(self.threshold)
                self.confusion_matrix.set_voxel_spacing(voxel_spacing)
                for metric in eval_metrics:
                    self.result[k][metric] = _funcs[metric](confusion_matrix=self.confusion_matrix,
                                                            nan_for_nonexisting=self.nan_for_nonexisting,
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
    for case in dict_scores:  # cases
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
    :param threshold: in ml for Image-level task
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
    all_scores["image-level classification"] = OrderedDict()

    test = [i[0] for i in test_ref_pair]
    ref = [i[1] for i in test_ref_pair]

    run_eval_start_time = time.perf_counter()
    p = Pool(num_threads)
    all_res = p.map(run_evaluation, zip(test, ref, [evaluator] * len(ref), [metric_kwargs] * len(ref)))
    p.close()
    p.join()
    print("run eval took ", time.perf_counter() - run_eval_start_time, "for", len(test))

    remainder_start = time.perf_counter()

    # brian code 1
#    all_scores["all"] = list(itertools.chain.from_iterable(all_res))

    for i in range(len(all_res)):
        # removed 1
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
            if label not in all_scores["image-level classification"]:
                all_scores["image-level classification"][label] = OrderedDict()
            for score, value in score_dict.items():
                if score in detection_scores:
                    if score not in all_scores["image-level classification"][label]:
                        all_scores["image-level classification"][label][score] = []
                    all_scores["image-level classification"][label][score].append(value)

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

    for label in all_scores["image-level classification"]:
        for score in all_scores["image-level classification"][label]:
            if nanmean:
                if score == 'LDR' or score == 'CCR':
                    all_scores["image-level classification"][label][score] = float(
                        np.nanmean(all_scores["image-level classification"][label][score]))
                else:
                    all_scores["image-level classification"][label][score] = float(
                        np.nansum(all_scores["image-level classification"][label][score]))
            else:
                if score == 'LDR' or score == 'CCR':
                    all_scores["image-level classification"][label][score] = float(
                        np.mean(all_scores["image-level classification"][label][score]))
                else:
                    all_scores["image-level classification"][label][score] = float(
                        np.sum(all_scores["image-level classification"][label][score]))
    # calculate image classification metric
    if isinstance(threshold, float):
        for label in all_scores["image-level classification"]:
            tp = float(all_scores["image-level classification"][label]["Image-level TP"])
            tn = float(all_scores["image-level classification"][label]["Image-level TN"])
            fp = float(all_scores["image-level classification"][label]["Image-level FP"])
            fn = float(all_scores["image-level classification"][label]["Image-level FN"])
            # positive reference cases
            all_scores["image-level classification"][label]["Positive reference studies"] = tp + fn
            # negative reference cases
            all_scores["image-level classification"][label]["Negative reference studies"] = tn + fp
            # calculate sensitivity
            all_scores["image-level classification"][label]["image-level Sensitivity/TPR"] = tp / (tp + fn + 1e-8)
            # calculate Precision
            all_scores["image-level classification"][label]["image-level Precision"] = tp / (tp + fp + 1e-8)
            # calculate specificity
            all_scores["image-level classification"][label]["image-level Specificity"] = tn / (tn + fp + 1e-8)
            # calculate specificity
            all_scores["image-level classification"][label]["image-level FPR"] = tp / (tp + fn + 1e-8)
            # calculate AUC for label > 0
            if int(label) > 0:
                y_true = np.array([i[label]['Volume Reference'] for i in all_scores["all"]])
                y_true = (y_true > threshold) * 1
                y_score = np.array([i[label]['Volume Test'] for i in all_scores["all"]])
                all_scores["image-level classification"][label]["image-level AUC"] = roc_auc_score(y_true, y_score)

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
        df4 = pd.DataFrame(all_scores["image-level classification"])
        try:
            with pd.ExcelWriter(excel_output_file) as writer:
                df1.to_excel(writer, sheet_name='all')
                df2.to_excel(writer, sheet_name='mean')
                df3.to_excel(writer, sheet_name='median')
                df4.to_excel(writer, sheet_name='image-level classification')
        except:
            print('no excel file name defined')
        print(f'results can be found here: {excel_output_file}')

    print("remainder took", time.perf_counter() - remainder_start_time)
    return all_scores


def evaluate_folder(folder_with_gts: str, folder_with_predictions: str, th: float, labels: tuple, specific: bool,
                    **metric_kwargs):
    """
    writes a summary.json to folder_with_predictions
    :param folder_with_gts: folder where the ground truth segmentations are saved. Must be nifti files.
    :param folder_with_predictions: folder where the predicted segmentations are saved. Must be nifti files.
    :param labels: tuple of int with the labels in the dataset. For example (0, 1, 2, 3) for Task001_BrainTumour.
    :return:
    """
    if isinstance(th, float):
        threshold = th
    else:
        threshold = None

    time_start = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    print('start time:', time_start)
    print('specific:', specific)

    if specific:

        list = ['NCCT_002.nii.gz', 'NCCT_003.nii.gz', 'NCCT_004.nii.gz', 'NCCT_005.nii.gz', 'NCCT_006.nii.gz',
                'NCCT_007.nii.gz', 'NCCT_009.nii.gz', 'NCCT_010.nii.gz', 'NCCT_011.nii.gz', 'NCCT_012.nii.gz',
                'NCCT_015.nii.gz', 'NCCT_019.nii.gz', 'NCCT_021.nii.gz', 'NCCT_022.nii.gz', 'NCCT_023.nii.gz',
                'NCCT_027.nii.gz', 'NCCT_028.nii.gz', 'NCCT_030.nii.gz', 'NCCT_031.nii.gz', 'NCCT_032.nii.gz',
                'NCCT_033.nii.gz', 'NCCT_034.nii.gz', 'NCCT_035.nii.gz', 'NCCT_038.nii.gz', 'NCCT_039.nii.gz',
                'NCCT_040.nii.gz', 'NCCT_041.nii.gz', 'NCCT_042.nii.gz', 'NCCT_043.nii.gz', 'NCCT_044.nii.gz',
                'NCCT_045.nii.gz', 'NCCT_046.nii.gz', 'NCCT_048.nii.gz', 'NCCT_049.nii.gz', 'NCCT_050.nii.gz',
                'NCCT_051.nii.gz', 'NCCT_052.nii.gz', 'NCCT_054.nii.gz', 'NCCT_055.nii.gz', 'NCCT_057.nii.gz',
                'NCCT_058.nii.gz', 'NCCT_059.nii.gz', 'NCCT_060.nii.gz', 'NCCT_061.nii.gz', 'NCCT_062.nii.gz',
                'NCCT_065.nii.gz', 'NCCT_070.nii.gz', 'NCCT_076.nii.gz', 'NCCT_077.nii.gz', 'NCCT_078.nii.gz',
                'NCCT_079.nii.gz', 'NCCT_082.nii.gz', 'NCCT_083.nii.gz', 'NCCT_084.nii.gz', 'NCCT_086.nii.gz',
                'NCCT_087.nii.gz', 'NCCT_088.nii.gz', 'NCCT_089.nii.gz', 'NCCT_090.nii.gz', 'NCCT_091.nii.gz',
                'NCCT_092.nii.gz', 'NCCT_095.nii.gz', 'NCCT_100.nii.gz', 'NCCT_102.nii.gz', 'NCCT_104.nii.gz',
                'NCCT_105.nii.gz', 'NCCT_106.nii.gz', 'NCCT_107.nii.gz', 'NCCT_108.nii.gz', 'NCCT_110.nii.gz',
                'NCCT_112.nii.gz', 'NCCT_113.nii.gz', 'NCCT_116.nii.gz', 'NCCT_117.nii.gz', 'NCCT_118.nii.gz',
                'NCCT_119.nii.gz', 'NCCT_121.nii.gz', 'NCCT_122.nii.gz', 'NCCT_123.nii.gz', 'NCCT_124.nii.gz',
                'NCCT_125.nii.gz', 'NCCT_126.nii.gz', 'NCCT_127.nii.gz', 'NCCT_129.nii.gz', 'NCCT_130.nii.gz',
                'NCCT_132.nii.gz', 'NCCT_134.nii.gz', 'NCCT_136.nii.gz', 'NCCT_138.nii.gz', 'NCCT_140.nii.gz',
                'NCCT_141.nii.gz', 'NCCT_143.nii.gz', 'NCCT_144.nii.gz', 'NCCT_145.nii.gz', 'NCCT_146.nii.gz',
                'NCCT_147.nii.gz', 'NCCT_148.nii.gz', 'NCCT_149.nii.gz', 'NCCT_150.nii.gz', 'NCCT_151.nii.gz',
                'NCCT_152.nii.gz', 'NCCT_153.nii.gz', 'NCCT_154.nii.gz', 'NCCT_155.nii.gz', 'NCCT_156.nii.gz',
                'NCCT_157.nii.gz', 'NCCT_160.nii.gz', 'NCCT_162.nii.gz', 'NCCT_163.nii.gz', 'NCCT_165.nii.gz',
                'NCCT_167.nii.gz', 'NCCT_169.nii.gz', 'NCCT_172.nii.gz', 'NCCT_174.nii.gz', 'NCCT_175.nii.gz',
                'NCCT_178.nii.gz', 'NCCT_179.nii.gz', 'NCCT_180.nii.gz', 'NCCT_185.nii.gz', 'NCCT_186.nii.gz',
                'NCCT_189.nii.gz', 'NCCT_190.nii.gz', 'NCCT_192.nii.gz', 'NCCT_195.nii.gz', 'NCCT_196.nii.gz',
                'NCCT_198.nii.gz', 'NCCT_199.nii.gz', 'NCCT_201.nii.gz', 'NCCT_207.nii.gz', 'NCCT_208.nii.gz',
                'NCCT_209.nii.gz', 'NCCT_210.nii.gz', 'NCCT_211.nii.gz', 'NCCT_213.nii.gz', 'NCCT_214.nii.gz',
                'NCCT_215.nii.gz', 'NCCT_216.nii.gz', 'NCCT_217.nii.gz', 'NCCT_219.nii.gz', 'NCCT_220.nii.gz',
                'NCCT_222.nii.gz', 'NCCT_226.nii.gz', 'NCCT_228.nii.gz', 'NCCT_229.nii.gz', 'NCCT_231.nii.gz',
                'NCCT_232.nii.gz', 'NCCT_235.nii.gz', 'NCCT_236.nii.gz', 'NCCT_240.nii.gz', 'NCCT_241.nii.gz',
                'NCCT_243.nii.gz', 'NCCT_244.nii.gz', 'NCCT_245.nii.gz', 'NCCT_248.nii.gz', 'NCCT_250.nii.gz',
                'NCCT_253.nii.gz', 'NCCT_255.nii.gz', 'NCCT_257.nii.gz', 'NCCT_258.nii.gz', 'NCCT_261.nii.gz']

        if timing_test:
            list = list[:8]

        files_gt_all = subfiles(folder_with_gts, suffix=".nii.gz", join=False, sort=True)
        files_gt = [g for g in files_gt_all if g in list]

        files_pred_all = subfiles(folder_with_predictions, suffix=".nii.gz", join=False, sort=True)
        files_pred = [p for p in files_pred_all if p in list]

        print('I evaluate for cases:', list)
    else:
        files_gt = subfiles(folder_with_gts, suffix=".nii.gz", join=False, sort=True)
        files_pred = subfiles(folder_with_predictions, suffix=".nii.gz", join=False, sort=True)

    assert all([i in files_pred for i in files_gt]), "files missing in folder_with_predictions"
    assert all([i in files_gt for i in files_pred]), "files missing in folder_with_gts"
    test_ref_pair = [(join(folder_with_predictions, i), join(folder_with_gts, i)) for i in files_pred]

    agg_s_time = time.perf_counter()
    res = aggregate_scores(test_ref_pair, threshold=threshold,
                           json_output_file=join(folder_with_predictions, f"summary_{time_start}.json"),
                           excel_output_file=join(folder_with_predictions, f"summary_{time_start}.xlsx"),
                           num_threads=8, labels=labels, **metric_kwargs)
    print("agg scores took: ", time.perf_counter() - agg_s_time)
    return res
