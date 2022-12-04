# Evaluation of Medical Image Segmentation Models for UncertainSmallEmpty Reference Annotations

Uncertain: inter-**(!)expert(!)** variability (can be quantified by comparing segmentations of two experts with this evaluator)

Small: approx. 1% of the relevent body cavity or organ

Empty: np.sum(reference_annotation) = 0 or below threshold


You can use this evaluator to compare a directory of ground truth segmentations (binary or multiclass) to a directory with manual or predicted segmentations in nii.gz format. 
The ground truth segmentation has to match the corresponding manual or predicted segmentation in shape and filename. 

The evaluator returns a .xlsx und .csv file with all metrics for segmentation and detection/image-classification task (if a threshold was set). Please refer to the table below to choose a set of metrics that is clinically meaningful for your data set. 

Please cite as

```
Ostmeier, Sophie, et al. "Evaluation of Medical Image Segmentation Models for Uncertain, Small or Empty Reference Annotations." arXiv preprint arXiv:2209.13008 (2022).
```
 
The corresponding preprint is

<ul>
"USE-Evaluator: Performance Metrics for Medical Image Segmentation Models with Uncertain, Small or Empty Reference Annotations"
</ul>
<ul>
arXiv:2209.13008 [eess.IV]
https://doi.org/10.48550/arXiv.2209.13008
</ul>
<ul>
Abstract:
Performance metrics for medical image segmentation models are used to measure the agreement between the reference annotation and the predicted segmentation. Usually, overlap metrics, such as the Dice, are used as a metric to evaluate the performance of these models in order for results to be comparable. However, there is a mismatch between the distributions of cases and difficulty level of segmentation tasks in public data sets compared to clinical practice. Common metrics fail to measure the impact of this mismatch, especially for clinical data sets that include low signal pathologies, a difficult segmentation task, and uncertain, small, or empty reference annotations. This limitation may result in ineffective research of machine learning practitioners in designing and optimizing models. Dimensions of evaluating clinical value include consideration of the uncertainty of reference annotations, independence from reference annotation volume size, and evaluation of classification of empty reference annotations. We study how uncertain, small, and empty reference annotations influence the value of metrics for medical image segmentation on an in-house data set regardless of the model. We examine metrics behavior on the predictions of a standard deep learning framework in order to identify metrics with clinical value. We compare to a public benchmark data set (BraTS 2019) with a high-signal pathology and certain, larger, and no empty reference annotations. We may show machine learning practitioners, how uncertain, small, or empty reference annotations require a rethinking of the evaluation and optimizing procedures.
</ul>

# Installation
```
git clone https://github.com/SophieOstmeier/UncertainSmallEmpty.git
cd UncertainSmallEmpty
pip install -r requirements.txt
```
# Usage

```
 python3 evaluator_run.py Test_files_gt Test_files_segmentation -number_classes 2 -threshold 1

```
- threshold (type: int): Lower threshold in ml for the evaluator that you want set for your segmentation task, this is also the threshold for the detection evaluation
- number_classes (type: int): number of classes including background class

For more information see -h.

The R scripts used in the preprint will be made public after peer-reviewed publication. These might help you to anaylse the behaviour of metrics on your data set.

# Recommendation
| **Metric**    | **Independence from Volume of Reference Annotation** | **Consideration of Uncertainty in Reference Annotation** | **Evaluation of Volume and Location agreement** | **Evaluation of Empty Agreement** |
|:-------------:|:----------------------------------------------------:|:---------------------------------------------------------:|:-------------------------------------------:|:-------------------------------:|
| **Volumetric Similarity**        | ✅   | ✅  | ❌  | ✅                      |
| **Absolute Volume Difference**       | ❌   | ✅  | ❌                                           | ❌ set threshold                 |
| **Dice**      | ❌                                                    | ❌                                                         | ✅                                  | ❌  set threshold                |
| **Recall**    | ❌                                                    | ❌                                                         | ✅                                  | ❌ set threshold                 |
| **Precision** | ❌                                                    | ❌                                                         | ✅                                  | ❌ set threshold                 |
| **Hausdorff Distance 95**     | ✅                                           | ✅                                                | ❌                                           | ❌ set threshold                 |
| **Average Surface Distance**       | ❌                                                    | ❌                                                         | ✅                                  | ❌ set threshold                 |
| **Surface Dice at Tolerance 0mm**  | ✅                                           | ❌                                                         | ✅                                  | ❌ set threshold                 |
| **Surface Dice at Tolerance 5mm**  | ✅                                           | ✅                                                | ✅                                  | ❌ set threshold                 |
| **Surface Dice at Tolerance 10mm** | (❌)                                                    | ✅                                                | ✅                                  | ❌ set threshold                 |

