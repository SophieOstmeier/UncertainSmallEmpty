# Evaluation of Medical Image Segmentation Models for UncertainSmallEmpty Reference Annotations
[paper](https://doi.org/10.1016/j.media.2023.102927) ([preprint](https://doi.org/10.48550/arXiv.2209.13008)) 
Uncertain: inter-**expert** variability (can be quantified by comparing segmentations of two experts with this evaluator)

Small: approx. 1% of the relevant body cavity or organ

Empty: np.sum(reference_annotation) = 0 or below threshold


You can use this evaluator to compare a directory of ground truth segmentations (binary or multiclass) to a directory with manual or predicted segmentations in nii.gz format. 
The ground truth segmentation has to match the corresponding manual or predicted segmentation in shape and filename. 

The evaluator returns a .xlsx und .csv file with all metrics for segmentation and detection/image-classification task (if a threshold was set). Please refer to the table below to choose a set of metrics that is clinically meaningful for your data set. 
 
The corresponding journal article is

<ul>
USE-Evaluator: Performance metrics for medical image segmentation models supervised by uncertain, small or empty reference annotations in neuroimaging
</ul>
<ul>
[https://doi.org/10.48550/arXiv.2209.13008](https://doi.org/10.1016/j.media.2023.102927)
</ul>
<ul>
Abstract:
Performance metrics for medical image segmentation models are used to measure the agreement between the reference annotation and the predicted segmentation. Usually, overlap metrics, such as the Dice, are used as a metric to evaluate the performance of these models in order for results to be comparable.

However, there is a mismatch between the distributions of cases and the difficulty level of segmentation tasks in public data sets compared to clinical practice. Common metrics used to assess performance fail to capture the impact of this mismatch, particularly when dealing with datasets in clinical settings that involve challenging segmentation tasks, pathologies with low signal, and reference annotations that are uncertain, small, or empty. Limitations of common metrics may result in ineffective machine learning research in designing and optimizing models. To effectively evaluate the clinical value of such models, it is essential to consider factors such as the uncertainty associated with reference annotations, the ability to accurately measure performance regardless of the size of the reference annotation volume, and the classification of cases where reference annotations are empty.

We study how uncertain, small, and empty reference annotations influence the value of metrics on a stroke in-house data set regardless of the model. We examine metrics behavior on the predictions of a standard deep learning framework in order to identify suitable metrics in such a setting. We compare our results to the BRATS 2019 and Spinal Cord public data sets. We show how uncertain, small, or empty reference annotations require a rethinking of the evaluation. 
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
| **Average Symmetric Surface Distance**       | ❌                                                    | ❌                                                         | ✅                                  | ❌ set threshold                 |
| **Surface Dice at Tolerance 2mm**  | ✅                                           | ✅                                                          | ✅                                  | ❌ set threshold                 |
| **Surface Dice at Tolerance 5mm**  | ✅                                           | ✅                                                | ✅                                  | ❌ set threshold                 |

Note the Jaccard Index (IoU) is equivalent to the Dice score. The choice depence on the user's preference.

If you find this helpful, please cite

```
@article{ostmeier2023use,
  title={USE-Evaluator: Performance metrics for medical image segmentation models supervised by uncertain, small or empty reference annotations in neuroimaging},
  author={Ostmeier, Sophie and Axelrod, Brian and Isensee, Fabian and Bertels, Jeroen and Mlynash, Michael and Christensen, Soren and Lansberg, Maarten G and Albers, Gregory W and Sheth, Rajen and Verhaaren, Benjamin FJ and others},
  journal={Medical Image Analysis},
  volume={90},
  pages={102927},
  year={2023},
  publisher={Elsevier}
}
```
