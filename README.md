# Evaluation of Medical Image Segmentation Models for UncertainSmallEmpty Reference Annotations

Uncertain: inter-expert variability (can be quantified by comparing segmentations of two experts with this evaluator)

Small: approx. 1% of the relevent body cavity or organ

Empty: np.sum(reference_annotation) = 0 or below threshold


You can use this evaluator to compare a directory of ground truth segmentations (binary or multiclass) to a directory with manual or predicted segmentations in nii.gz format. 
The ground truth segmentation has to match the corresponding manual or predicted segmentation in shape and filename. 

The evaluator returns a .xlsx und .csv file with all metrics for segmentation and detection/image-classification task (if a threshold was set). Please refer to the table below to choose a set of metrics that is clinically meaningful for your data set. 

Please cite as

```
Ostmeier, S., Axelrod, B., Bertels, J., Isensee, F., Lansberg, M.G., Christensen, S., Albers, G.W., 
Li, L.J. and Heit, J.J., 2022. Evaluation of Medical Image Segmentation Models for Uncertain, Small 
or Empty Reference Annotations. arXiv preprint arXiv:2209.13008.
```
 
The corresponding preprint is

<ul>
"Evaluation of Medical Image Segmentation Models for Uncertain, Small or Empty Reference Annotations"
</ul>
<ul>
arXiv:2209.13008 [eess.IV]
https://doi.org/10.48550/arXiv.2209.13008
</ul>
<ul>
Abstract:
Performance metrics for medical image segmentation models are used to measure agreement between the reference annotation and the prediction. A common set  of metrics is used in the development of such models to make results more comparable. However, there is a mismatch between the distributions in public data  sets and cases encountered in clinical practice. Many common metrics fail to measure the impact of this mismatch, especially for clinical data sets containing uncertain, small or empty reference annotation. Thus, models may not be validated for clinically meaningful agreement by such metrics. Dimensions of evaluating clinical value include independence from reference annotation volume size, consideration of uncertainty of reference annotations, reward of volumetric and/or location agreement and reward of correct classification of empty reference annotations. Unlike common public data sets, our in-house data set is more representative (NCCT stroke dataset). It contains uncertain, small or empty reference annotations. We examine publicly available metrics on the predictions of a deep learning framework in order to identify for which settings common metrics provide clinical meaningful results. We compare to a public benchmark data set without uncertain, small or empty reference annotations.
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

