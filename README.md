# AANet
demo and pre-trained weight of AANet --- a dense descriptor for local feature matching. 

# Pre-Trained Weights
We trained our AANet with one-stage end-to-end triplet training strategy on MS-COCO, Multi-illumination and VIDIT datasets (same as LISRD) and the pre-trained weight is compressed as dna.rar

# Model file
The core implementation of AANet is shown in AANet_core.py

# DEMO：SIFT+AANet
We provide the demo of exporting SIFT keypoints and AANet descriptor in export_dna_sift.py, and it can be easily modified to other off-the-shelf detectors and descriptors for evaluation.
