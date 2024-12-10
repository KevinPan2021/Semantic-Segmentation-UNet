# Semantic Segmentation with UNet

This project aims to preform self driving Semantic Segmentation with UNet from Scratch.
![Segmentation](external/GUI.png)


### Dataset: 
[Kaggle lyft udacity challenge](https://www.kaggle.com/datasets/kumaresanmanickavelu/lyft-udacity-challenge)


### Build: 

	CPU: Intel i9-13900H (14 cores)
	GPU: NVIDIA RTX 4060 (VRAM 8 GB)
	RAM: 32 GB


### Training Curves

<p align="center">
  <img src="external/loss.png" alt="Loss Curve" width="45%">
  <img src="external/ACC.png" alt="Acc Curve" width="45%">
</p>

<p align="center">
  <img src="external/MIoU.png" alt="New Plot" width="45%">
</p>


### Code Structure:
```bash
├── GUI.py (Run to generate a GUI)
├── main.py (Run to train model)
├── unet.py
├── qt_main.ui
├── training.py
├── summary.py
├── visualization.py

```

### Credits:
	"U-Net: Convolutional Networks for Biomedical Image Segmentation"



	