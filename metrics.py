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

import numpy as np
from medpy import metric
from surface_distance import compute_surface_distances
from scipy.spatial import distance

def assert_shape(test, reference):
    assert test.shape == reference.shape, "Shape mismatch: {} and {}".format(
        test.shape, reference.shape)

class ConfusionMatrix:

    def __init__(self, test=None, reference=None):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.reference_empty = None
        self.reference_full = None
        self.test_empty = None
        self.test_full = None
        self.set_reference(reference)
        self.set_test(test)

    def set_test(self, test):

        self.test = test
        self.reset()

    def set_reference(self, reference):

        self.reference = reference
        self.reset()

    def reset(self):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.test_empty = None
        self.test_full = None
        self.reference_empty = None
        self.reference_full = None

    def compute(self):

        if self.test is None or self.reference is None:
            raise ValueError("'test' and 'reference' must both be set to compute confusion matrix.")

        assert_shape(self.test, self.reference)

        self.tp = int(((self.test != 0) * (self.reference != 0)).sum())
        self.fp = int(((self.test != 0) * (self.reference == 0)).sum())
        self.tn = int(((self.test == 0) * (self.reference == 0)).sum())
        self.fn = int(((self.test == 0) * (self.reference != 0)).sum())
        self.size = int(np.prod(self.reference.shape, dtype=np.int64))
        self.test_empty = not np.any(self.test)
        self.test_full = np.all(self.test)
        self.reference_empty = not np.any(self.reference)
        self.reference_full = np.all(self.reference)

    def get_matrix(self):

        for entry in (self.tp, self.fp, self.tn, self.fn):
            if entry is None:
                self.compute()
                break

        return self.tp, self.fp, self.tn, self.fn

    def get_size(self):

        if self.size is None:
            self.compute()
        return self.size

    def get_existence(self):

        for case in (self.test_empty, self.test_full, self.reference_empty, self.reference_full):
            if case is None:
                self.compute()
                break

        return self.test_empty, self.test_full, self.reference_empty, self.reference_full


def dice_orig(test=None, reference=None, confusion_matrix=None,nan_for_nonexisting=True, **kwargs):
    """2TP / (2TP + FP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty and reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(2. * tp / (2 * tp + fp + fn))


def dice_smooth_1(test=None, reference=None, confusion_matrix=None, threshold=None,voxel_spacing=None, nan_for_nonexisting=True,
                  **kwargs):
    """2TP / (2TP + FP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    x, y, z = voxel_spacing
    voxel_volume = x * y * z

    volume_ref = (tp + fn) * voxel_volume * 0.001
    volume_tes = (tp + fp) * voxel_volume * 0.001

    reference_small = not volume_ref > threshold
    test_small = not volume_tes > threshold

    return float(((2. * tp) + 1) / ((2 * tp + fp + fn) + 1))



def dice_th(test=None, reference=None, confusion_matrix=None,voxel_spacing=None, nan_for_nonexisting=True,threshold=None, **kwargs):
    """2TP / (2TP + FP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    x, y, z = voxel_spacing
    voxel_volume = x * y * z

    volume_ref = (tp + fn) * voxel_volume * 0.001
    volume_tes = (tp + fp) * voxel_volume * 0.001

    reference_small = not volume_ref > threshold
    test_small = not volume_tes > threshold

    if reference_small and test_small:
        return float("NaN")
    else:
        return float(2. * tp / (2 * tp + fp + fn))


def precision(test=None, reference=None, confusion_matrix=None, threshold=None,voxel_spacing=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FP)"""
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    x, y, z = voxel_spacing
    voxel_volume = x * y * z

    volume_tes = (tp + fp) * voxel_volume * 0.001

    test_small = not volume_tes > threshold

    if test_small:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.
    return float(tp / (tp + fp))


def sensitivity(test=None, reference=None, confusion_matrix=None,threshold=None, voxel_spacing=None, nan_for_nonexisting=True,
                **kwargs):
    """TP / (TP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    x, y, z = voxel_spacing
    voxel_volume = x * y * z

    volume_ref = (tp + fn) * voxel_volume * 0.001

    reference_small = not volume_ref > threshold

    if reference_small:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.
    else:

        return float(tp / (tp + fn))


def recall(test=None, reference=None, confusion_matrix=None, threshold=None,voxel_spacing=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FN)"""

    return sensitivity(test, reference, confusion_matrix, threshold,voxel_spacing, nan_for_nonexisting, **kwargs)


def specificity(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True,
                **kwargs):
    """TN / (TN + FP)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tn / (tn + fp))


def accuracy(test=None, reference=None, confusion_matrix=None, **kwargs):
    """(TP + TN) / (TP + FP + FN + TN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return float((tp + tn) / (tp + fp + tn + fn))


def fscore(test=None, reference=None, confusion_matrix=None, threshold=None, voxel_spacing=None, nan_for_nonexisting=True, beta=1.,
           **kwargs):
    """(1 + b^2) * TP / ((1 + b^2) * TP + b^2 * FN + FP)"""

    precision_ = precision(test, reference, confusion_matrix, threshold,voxel_spacing, nan_for_nonexisting)
    recall_ = recall(test, reference, confusion_matrix,threshold, voxel_spacing, nan_for_nonexisting)

    return (1 + beta * beta) * precision_ * recall_ / \
           ((beta * beta * precision_) + recall_ + 1e-8)


def false_positive_rate(test=None, reference=None, confusion_matrix=None, voxel_spacing=None, nan_for_nonexisting=True,
                        **kwargs):
    """FP / (FP + TN)"""

    return 1 - specificity(test, reference, confusion_matrix, voxel_spacing, nan_for_nonexisting)


def false_omission_rate(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """FN / (TN + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(fn / (fn + tn))


def false_negative_rate(test=None, reference=None, confusion_matrix=None,threshold=None, voxel_spacing=None, nan_for_nonexisting=True,
                        **kwargs):
    """FN / (TP + FN)"""

    return 1 - sensitivity(test, reference, confusion_matrix, threshold, voxel_spacing, nan_for_nonexisting)


def true_negative_rate(test=None, reference=None, confusion_matrix=None, voxel_spacing=None, nan_for_nonexisting=True,
                       **kwargs):
    """TN / (TN + FP)"""

    return specificity(test, reference, confusion_matrix, voxel_spacing, nan_for_nonexisting)


def false_discovery_rate(test=None, reference=None, confusion_matrix=None,threshold=None, voxel_spacing=None, nan_for_nonexisting=True,
                         **kwargs):
    """FP / (TP + FP)"""

    return 1 - precision(test, reference, confusion_matrix, threshold, voxel_spacing, nan_for_nonexisting)


def negative_predictive_value(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TN / (TN + FN)"""

    return 1 - false_omission_rate(test, reference, confusion_matrix, nan_for_nonexisting)


def total_positives_test(test=None, reference=None, confusion_matrix=None, **kwargs):
    """TP + FP"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return tp + fp


def total_negatives_test(test=None, reference=None, confusion_matrix=None, **kwargs):
    """TN + FN"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return tn + fn


def total_positives_reference(test=None, reference=None, confusion_matrix=None, **kwargs):
    """TP + FN"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return tp + fn


def total_negatives_reference(test=None, reference=None, confusion_matrix=None, **kwargs):
    """TN + FP"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return tn + fp


def hausdorff_distance(test=None, reference=None, confusion_matrix=None, threshold=None,nan_for_nonexisting=True, voxel_spacing=None,
                       connectivity=1, **kwargs):
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    x, y, z = voxel_spacing
    voxel_volume = x * y * z

    volume_ref = (tp + fn) * voxel_volume * 0.001
    volume_tes = (tp + fp) * voxel_volume * 0.001

    reference_small = not volume_ref > threshold
    test_small = not volume_tes > threshold

    if reference_small or test_small or test_full or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0

    test, reference = confusion_matrix.test, confusion_matrix.reference

    return metric.hd(test, reference, voxel_spacing, connectivity)


def hausdorff_distance_95(test=None, reference=None, confusion_matrix=None, threshold=None,nan_for_nonexisting=True,
                          voxel_spacing=None, connectivity=1, **kwargs):
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    x, y, z = voxel_spacing
    voxel_volume = x * y * z

    volume_ref = (tp + fn) * voxel_volume * 0.001
    volume_tes = (tp + fp) * voxel_volume * 0.001

    reference_small = not volume_ref > threshold
    test_small = not volume_tes > threshold

    if reference_small or test_small or test_full or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0

    test, reference = confusion_matrix.test, confusion_matrix.reference

    return metric.hd95(test, reference, voxel_spacing, connectivity)


def avg_surface_distance(test=None, reference=None, confusion_matrix=None,threshold=None, nan_for_nonexisting=True, voxel_spacing=None,
                         connectivity=1, **kwargs):
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    x, y, z = voxel_spacing
    voxel_volume = x * y * z

    volume_ref = (tp + fn) * voxel_volume * 0.001
    volume_tes = (tp + fp) * voxel_volume * 0.001

    reference_small = not volume_ref > threshold
    test_small = not volume_tes > threshold

    if reference_small or test_small or test_full or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0

    test, reference = confusion_matrix.test, confusion_matrix.reference

    return metric.asd(test, reference, voxel_spacing, connectivity)


def avg_surface_distance_symmetric(test=None, reference=None, confusion_matrix=None,threshold=None, nan_for_nonexisting=True,
                                   voxel_spacing=None, connectivity=1, **kwargs):
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    x, y, z = voxel_spacing
    voxel_volume = x * y * z

    volume_ref = (tp + fn) * voxel_volume * 0.001
    volume_tes = (tp + fp) * voxel_volume * 0.001

    reference_small = not volume_ref > threshold
    test_small = not volume_tes > threshold

    if reference_small or test_small or test_full or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0


    test, reference = confusion_matrix.test, confusion_matrix.reference

    return metric.assd(test, reference, voxel_spacing, connectivity)


def compute_surface_dice_at_tolerance_0(test=None, reference=None, confusion_matrix=None,threshold=None, nan_for_nonexisting=True,
                                        voxel_spacing=None, **kwargs):
    """Computes the _surface_ DICE coefficient at a specified tolerance.

    Computes the _surface_ DICE coefficient at a specified tolerance. Not to be
    confused with the standard _volumetric_ DICE coefficient. The surface DICE
    measures the overlap of two surfaces instead of two volumes. A surface
    element is counted as overlapping (or touching), when the closest distance to
    the other surface is less or equal to the specified tolerance. The DICE
    coefficient is in the range between 0.0 (no overlap) to 1.0 (perfect overlap).

    Args:
    surface_distances: dict with "distances_gt_to_pred", "distances_pred_to_gt"
      "surfel_areas_gt", "surfel_areas_pred" created by
      compute_surface_distances()
    tolerance_mm: a float value. The tolerance in mm

    Returns:
    A float value. The surface DICE coefficient in [0.0, 1.0].
    """
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    x, y, z = voxel_spacing
    voxel_volume = x * y * z

    volume_ref = (tp + fn) * voxel_volume * 0.001
    volume_tes = (tp + fp) * voxel_volume * 0.001

    reference_small = not volume_ref > threshold
    test_small = not volume_tes > threshold

    if reference_small or test_small or test_full or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0

    test, reference = confusion_matrix.test, confusion_matrix.reference
    surface_distances = compute_surface_distances(test, reference, voxel_spacing)
    distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
    distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
    surfel_areas_gt = surface_distances["surfel_areas_gt"]
    surfel_areas_pred = surface_distances["surfel_areas_pred"]

    # for 10mm
    overlap_gt_0 = np.sum(surfel_areas_gt[distances_gt_to_pred <= 0])
    overlap_pred_0 = np.sum(surfel_areas_pred[distances_pred_to_gt <= 0])
    surface_dice_0 = (overlap_gt_0 + overlap_pred_0) / (
            np.sum(surfel_areas_gt) + np.sum(surfel_areas_pred))

    return surface_dice_0


def compute_surface_dice_at_tolerance_5(test=None, reference=None, confusion_matrix=None, threshold=None,nan_for_nonexisting=True,
                                        voxel_spacing=None, **kwargs):
    """Computes the _surface_ DICE coefficient at a specified tolerance.

    Computes the _surface_ DICE coefficient at a specified tolerance. Not to be
    confused with the standard _volumetric_ DICE coefficient. The surface DICE
    measures the overlap of two surfaces instead of two volumes. A surface
    element is counted as overlapping (or touching), when the closest distance to
    the other surface is less or equal to the specified tolerance. The DICE
    coefficient is in the range between 0.0 (no overlap) to 1.0 (perfect overlap).

    Args:
    surface_distances: dict with "distances_gt_to_pred", "distances_pred_to_gt"
      "surfel_areas_gt", "surfel_areas_pred" created by
      compute_surface_distances()
    tolerance_mm: a float value. The tolerance in mm

    Returns:
    A float value. The surface DICE coefficient in [0.0, 1.0].
    """
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    x, y, z = voxel_spacing
    voxel_volume = x * y * z

    volume_ref = (tp + fn) * voxel_volume * 0.001
    volume_tes = (tp + fp) * voxel_volume * 0.001

    reference_small = not volume_ref > threshold
    test_small = not volume_tes > threshold

    if reference_small or test_small or test_full or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0

    test, reference = confusion_matrix.test, confusion_matrix.reference
    surface_distances = compute_surface_distances(test, reference, voxel_spacing)
    distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
    distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
    surfel_areas_gt = surface_distances["surfel_areas_gt"]
    surfel_areas_pred = surface_distances["surfel_areas_pred"]

    # for 5mm
    overlap_gt_5 = np.sum(surfel_areas_gt[distances_gt_to_pred <= 5])
    overlap_pred_5 = np.sum(surfel_areas_pred[distances_pred_to_gt <= 5])
    surface_dice_5 = (overlap_gt_5 + overlap_pred_5) / (
            np.sum(surfel_areas_gt) + np.sum(surfel_areas_pred))

    return surface_dice_5


def compute_surface_dice_at_tolerance_10(test=None, reference=None, confusion_matrix=None, threshold=None,nan_for_nonexisting=True,
                                         voxel_spacing=None, **kwargs):
    """Computes the _surface_ DICE coefficient at a specified tolerance.

    Computes the _surface_ DICE coefficient at a specified tolerance. Not to be
    confused with the standard _volumetric_ DICE coefficient. The surface DICE
    measures the overlap of two surfaces instead of two volumes. A surface
    element is counted as overlapping (or touching), when the closest distance to
    the other surface is less or equal to the specified tolerance. The DICE
    coefficient is in the range between 0.0 (no overlap) to 1.0 (perfect overlap).

    Args:
    surface_distances: dict with "distances_gt_to_pred", "distances_pred_to_gt"
      "surfel_areas_gt", "surfel_areas_pred" created by
      compute_surface_distances()
    tolerance_mm: a float value. The tolerance in mm

    Returns:
    A float value. The surface DICE coefficient in [0.0, 1.0].
    """
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    x, y, z = voxel_spacing
    voxel_volume = x * y * z

    volume_ref = (tp + fn) * voxel_volume * 0.001
    volume_tes = (tp + fp) * voxel_volume * 0.001

    reference_small = not volume_ref > threshold
    test_small = not volume_tes > threshold

    if reference_small or test_small or test_full or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0

    test, reference = confusion_matrix.test, confusion_matrix.reference
    surface_distances = compute_surface_distances(test, reference, voxel_spacing)
    distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
    distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
    surfel_areas_gt = surface_distances["surfel_areas_gt"]
    surfel_areas_pred = surface_distances["surfel_areas_pred"]

    # for 10mm
    overlap_gt_10 = np.sum(surfel_areas_gt[distances_gt_to_pred <= 10])
    overlap_pred_10 = np.sum(surfel_areas_pred[distances_pred_to_gt <= 10])
    surface_dice_10 = (overlap_gt_10 + overlap_pred_10) / (
            np.sum(surfel_areas_gt) + np.sum(surfel_areas_pred))

    return surface_dice_10

def malahanobis_distance(test=None, reference=None):
    covariance = np.cov([test.ravel(), reference.ravel()])
    cov_inv = np.linalg.inv(covariance)  # inv. covariance matrix
    return distance.mahalanobis(test, reference, cov_inv)


def volume_test(test=None, reference=None, confusion_matrix=None, voxel_spacing=None, nan_for_nonexisting=True,
                **kwargs):
    x, y, z = voxel_spacing
    voxel_volume = x * y * z

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return (tp + fp) * voxel_volume * 0.001


def volume_reference(test=None, reference=None, confusion_matrix=None, voxel_spacing=None, nan_for_nonexisting=True,
                     **kwargs):

    x, y, z = voxel_spacing
    voxel_volume = x * y * z

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return float((tp + fn) * voxel_volume * 0.001)


def abs_volume_difference(test=None, reference=None, confusion_matrix=None, voxel_spacing=None,
                          nan_for_nonexisting=True, **kwargs):
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    x, y, z = voxel_spacing
    voxel_volume = x * y * z

    return float(abs((tp + fn) - (tp + fp)) * voxel_volume * 0.001)

def rel_volume_difference(test=None, reference=None, confusion_matrix=None, voxel_spacing=None,
                          nan_for_nonexisting=True, **kwargs):
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.
    return float(abs((tp + fn) - (tp + fp)) / (tp + fn))


def volumetric_similarity(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """ (FN - FP) / (2TP + FP + FN) """
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty and reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(1 - (abs(fn - fp)/(2*tp + fp + fn)))

def detection(test=None, reference=None, confusion_matrix=None, threshold=None,voxel_spacing=None, nan_for_nonexisting=True, **kwargs):
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    x, y, z = voxel_spacing
    voxel_volume = x * y * z

    volume_ref = (tp + fn) * voxel_volume * 0.001
    volume_tes = (tp + fp) * voxel_volume * 0.001

    reference_small = not volume_ref > threshold
    test_small = not volume_tes > threshold

    if reference_small and test_small: #  images classified true positive
        return 1
    elif not reference_small and not test_small:
        return 1
    else:
        return 0


def detection_tp(test=None, reference=None, confusion_matrix=None, threshold=None,voxel_spacing=None, nan_for_nonexisting=True, **kwargs):
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    x, y, z = voxel_spacing
    voxel_volume = x * y * z

    volume_ref = (tp + fn) * voxel_volume * 0.001
    volume_tes = (tp + fp) * voxel_volume * 0.001

    reference_small = not volume_ref > threshold
    test_small = not volume_tes > threshold

    if not reference_small and not test_small: #  images classified true positive
        return 1
    else:
        return 0


def detection_tn(test=None, reference=None, confusion_matrix=None, threshold=None, voxel_spacing=None,
                 nan_for_nonexisting=True, **kwargs):
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    x, y, z = voxel_spacing
    voxel_volume = x * y * z

    volume_ref = (tp + fn) * voxel_volume * 0.001
    volume_tes = (tp + fp) * voxel_volume * 0.001

    reference_small = not volume_ref > threshold
    test_small = not volume_tes > threshold

    if reference_small and test_small:  # images classified true negative
        return 1
    else:
        return 0

def detection_fp(test=None, reference=None, confusion_matrix=None, threshold=None, voxel_spacing=None,
                 nan_for_nonexisting=True, **kwargs):
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    x, y, z = voxel_spacing
    voxel_volume = x * y * z

    volume_ref = (tp + fn) * voxel_volume * 0.001
    volume_tes = (tp + fp) * voxel_volume * 0.001

    reference_small = not volume_ref > threshold
    test_small = not volume_tes > threshold

    if reference_small and not test_small:  # images classified false positive
        return 1
    else:
        return 0

def detection_fn(test=None, reference=None, confusion_matrix=None, threshold=None, voxel_spacing=None,
                 nan_for_nonexisting=True, **kwargs):
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    x, y, z = voxel_spacing
    voxel_volume = x * y * z

    volume_ref = (tp + fn) * voxel_volume * 0.001
    volume_tes = (tp + fp) * voxel_volume * 0.001

    reference_small = not volume_ref > threshold
    test_small = not volume_tes > threshold

    if not reference_small and test_small:  # images classified false negative
        return 1
    else:
        return 0

def ldr(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if reference_empty:
        return float("NaN")
    elif (tp+fp) > 0:
        return 1
    elif test_empty:
        return 0

def class_imbalance_alpha(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """
    Includes all voxels of the image. Use only  for comparison within a data set with a constant number of voxels.
    :param test:
    :param reference:
    :param confusion_matrix:
    :param nan_for_nonexisting:
    :param kwargs:
    :return:
    """

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return (tp+fn)/(tn+fp+tn+fn)

def relative_tp(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return tp/(tp+tn+fn+fp)

def tp(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """ (2TP + FP)/(2*TP + FN + 2*FP)"""
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty and reference_empty:
        return 1

    return tp

def fp(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """ (2TP + FP)/(2*TP + FN + 2*FP)"""
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if reference_empty:
        return float("NaN")

    return fp

ALL_METRICS = {
    "TP":tp,
    "relative FP":fp,
    "CCR": detection,
    "LDR": ldr,
    "Relative True Positives": relative_tp,
    "Class Imbalance": class_imbalance_alpha,
    "False Positive Rate": false_positive_rate,
    "Dice original": dice_orig,
    "Dice with Laplace smoothing 1": dice_smooth_1,
    "Dice": dice_th,
    "Surface Dice at Tolerance 0mm": compute_surface_dice_at_tolerance_0,
    "Surface Dice at Tolerance 5mm": compute_surface_dice_at_tolerance_5,
    "Surface Dice at Tolerance 10mm": compute_surface_dice_at_tolerance_10,
    "Hausdorff Distance": hausdorff_distance,
    "Hausdorff Distance 95": hausdorff_distance_95,
    "Precision": precision,
    "Recall": recall,
    "fscore": fscore,
    "Avg. Symmetric Surface Distance": avg_surface_distance_symmetric,
    "Avg. Surface Distance": avg_surface_distance,
    "Accuracy": accuracy,
    "False Omission Rate": false_omission_rate,
    "Negative Predictive Value": negative_predictive_value,
    "False Negative Rate": false_negative_rate,
    "True Negative Rate": true_negative_rate,
    "False Discovery Rate": false_discovery_rate,
    "Total Positives Test": total_positives_test,
    "Total Negatives Test": total_negatives_test,
    "Total Positives Reference": total_positives_reference,
    "Total Negatives Reference": total_negatives_reference,
    "Volume Reference": volume_reference,
    "Volume Test": volume_test,
    "Volume Absolute Difference": abs_volume_difference,
    "Volume Relative Difference": rel_volume_difference,
    "Volumetric Similarity": volumetric_similarity,
    "Detection TN":detection_tn,
    "Detection TP":detection_tp,
    "Detection FN":detection_fn,
    "Detection FP":detection_fp
}