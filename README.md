# UncertainSmallEmpty

You can use this evaluator to compare a directory of ground truth segmentations to a directory with manual or predicted segmentations in nii.gz format. 
The ground truth segmentation has to match the corresponding manual or predicted segmentation in shape and filename. 
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
Performance metrics for medical image segmentation models are used to measure agreement between the reference annotation and the prediction. A common set  of metrics is used in the development of such models to make results more comparable. However, there is a mismatch between the distributions in public data  sets and cases encountered in clinical practice. Many common metrics fail to measure the impact of this mismatch, especially for clinical data sets containing uncertain, small or empty reference annotation. Thus, models may not be validated for clinically meaningful agreement by such metrics. Dimensions of evaluating clinical value include independence from reference annotation volume size, consideration of uncertainty of reference annotations, reward of volumetric and/or location agreement and reward of correct classification of empty reference annotations. Unlike common public data sets, our in-house data set is more representative. It contains uncertain, small or empty reference annotations. We examine publicly available metrics on the predictions of a deep learning framework in order to identify for which settings common metrics provide clinical meaningful results. We compare to a public benchmark data set without uncertain, small or empty reference annotations.
</ul>

# Installation
```
git clone https://github.com/SophieOstmeier/UncertainSmallEmpty.git
cd UncertainSmallEmpty
pip install UncertainSmallEmpty
pip install -r requirements.txt
```
# Usage

```
 python3 evaluator_run.py Test_files_gt Test_files_segmentation -hidden -check
```
The R scripts used in the preprint will be made public after peer-reviewed publication. These might help you to anaylse the behaviour of metrics on your data set.

# Guideline
| **Metric**    | **Independence from Volume of Reference Annotation** | **Consideration of Uncertainty in Reference Annotation** | **Reward of Volume and Location agreement** | **Reward of Absence Agreement** |
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
| **Surface Dice at Tolerance 10mm** | ❌                                                    | ✅                                                | ✅                                  | ❌ set threshold                 |

# Backbone Model and Evaluator

```
https://github.com/MIC-DKFZ/nnUNet.git
```
```
Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2020). nnU-Net: a self-configuring method 
for deep learning-based biomedical image segmentation. Nature Methods, 1-9.
```
