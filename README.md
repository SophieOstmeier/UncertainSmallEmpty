# UncertainSmallEmpty

Here are the implementations and definitions for the metrics used in 

"Evaluation of Medical Image Segmentation Models for Uncertain, Small or Empty Reference Annotations"

 Sophie Ostmeier, Brian Axelrod, Jeroen Bertels, Fabian Isensee, Maarten G.Lansberg, Soren Christensen, Gregory W. Albers, Li-Jia Li, Jeremy J. Heit

 Abstract:
 Performance metrics for medical image segmentation models are used to measure agreement between the reference annotation and the prediction. A common set  of metrics is used in the development of such models to make results more comparable. However, there is a mismatch between the distributions in public data  sets and cases encountered in clinical practice. Many common metrics fail to measure the impact of this mismatch, especially for clinical data sets containing uncertain, small or empty reference annotation. Thus, models may not be validated for clinically meaningful agreement by such metrics. Dimensions of evaluating clinical value include independence from reference annotation volume size, consideration of uncertainty of reference annotations, reward of volumetric and/or location agreement and reward of correct classification of empty reference annotations. Unlike common public data sets, our in-house data set is more representative. It contains uncertain, small or empty reference annotations. We examine publicly available metrics on the predictions of a deep learning framework in order to identify for which settings common metrics provide clinical meaningful results. We compare to a public benchmark data set without uncertain, small or empty reference annotations.

 Comments:	16 pages, 10 figures
 Subjects:	Image and Video Processing (eess.IV); Computer Vision and Pattern Recognition (cs.CV); Machine Learning (cs.LG)
 Cite as:	arXiv:2209.13008 [eess.IV]
 	(or arXiv:2209.13008v1 [eess.IV] for this version)
 
 https://doi.org/10.48550/arXiv.2209.13008
 
You can use this evaluator on two directories with .nii.gz files that contain the segmentations you would like to compare by using

`
The backbone model is nnUNet found here

https://github.com/MIC-DKFZ/nnUNet.git
