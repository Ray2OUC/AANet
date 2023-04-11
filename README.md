# LearningGeneral Descriptors for Image Matching with Regression Feedback
demo and pre-trained weight of AANet --- a dense descriptor for local feature matching.
Our work is accepted by IEEE Transactions on Circuits and Systems for Video Technology 2023

# Pre-Trained Weights
We trained our AANet with one-stage end-to-end triplet training strategy on MS-COCO, Multi-illumination and VIDIT datasets (same as [LISRD](https://github.com/rpautrat/LISRD)) and the pre-trained weight is compressed as dna.rar

# Model file
The core implementation of AANet is shown in AANet_core.py

# DEMO：SIFT+AANet
We provide the demo of exporting SIFT keypoints and AANet descriptor in export_descriptor_sift.py, and it can be easily modified to other off-the-shelf detectors and matchers for evaluation.
```
CUDA_VISIBLE_DEVICES=0 python export_descriptor_sift.py
```
For more evaluation details, please refer to the [HIFT](https://github.com/Ray2OUC/HIFT) and [LISRD](https://github.com/rpautrat/LISRD)

# Citation

If you are interested in this work, please cite the following work:

```
@ARTICLE{10058693,
  author={Rao, Yuan and Ju, Yakun and Wang, Sen and Gao, Feng and Fan, Hao and Dong, Junyu},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={Learning Enriched Feature Descriptor for Image Matching and Visual Measurement}, 
  year={2023},
  volume={72},
  number={},
  pages={1-12},
  doi={10.1109/TIM.2023.3249237}}
```

# Acknowledgments
Our work is based on [LISRD](https://github.com/rpautrat/LISRD) and we use their code.  We appreciate the previous open-source repository [LISRD](https://github.com/rpautrat/LISRD)
