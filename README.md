# SR-PointNet: Point Cloud-Driven Velocity Field Super-Resolution for Wave-Structure Interaction Analysis
**Author**: Jiahui Wang (wangjiahuiSCU@outlook.com), Rundong Liu, Hong Xiao

---

## Dataset and Code
The dataset for this project can be downloaded using the following: https://6th42r-my.sharepoint.com/:u:/g/personal/pingping_6th42r_onmicrosoft_com/ERGHtQRJpElOhy04iIXY8VYBHanEeCXdFzNB-iP55XiPRA?e=l1KSN8

The "dataset.zip" file contains 8 ".h5" files The. information of these files is listed in the following table:

| File Name                          | Case Description                                      | Resolution  |
|------------------------------------|-------------------------------------------------------|-------------|
| case1LR.h5                         | Solitary Wave Interaction with a Submerged Breakwater (SWSB) - Low Resolution | 125×50      |
| case1HR.h5                         | Solitary Wave Interaction with a Submerged Breakwater (SWSB) - High Resolution | 1000×400    |
| case2LR.h5                         | Solitary Wave Interaction with a Front Step (SWFS) - Low Resolution | 125×50      |
| case2HR.h5                         | Solitary Wave Interaction with a Front Step (SWFS) - High Resolution | 1000×400    |
| case1LR_generalization.h5          | Generalization Verification (Modified Wave Height) - Low Resolution | 125×50      |
| case1HR_generalization.h5          | Generalization Verification (Modified Wave Height) - High Resolution | 1000×400    |
| case1LR_generalization_Geo.h5      | Generalization Verification (Modified Breakwater Geometry) - Low Resolution | 125×50      |
| case1HR_generalization_Geo.h5      | Generalization Verification (Modified Breakwater Geometry) - High Resolution | 1000×400    |

*Detailed information about the dataset can be found in the paper.*

All of these dataset files have the same dimensions of [Sp, N, C], where Sp denotes the number of samples in the dataset; N refers to the points in every sample; C means the number of channels (2, velocity field in X- and Z-directions).

## Model training parameters

| Parameter     |        Value | 
| ------------- |------------:|
| Learning rate |         5e-4 | 
| global_feat_dim   |       512| 
| Batch size    |           64 |  