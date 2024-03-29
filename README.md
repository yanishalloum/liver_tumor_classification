# Liver Tumor Classification

## About

This project aims at detecting tumors in liver scans using deep learning models trained on a large dataset.


## Guideline

1. Clone or download the repo.

```
git clone https://github.com/yanishalloum/liver_tumor_classification.git
```

2. Install all the dependencies:

```
pip install -r requirements.txt
```

3. Install the database and launch preprocess.py:

## Dataset Links

I used the "Liver Segmentation Database" from Kaggle:

- [Part 1](https://www.kaggle.com/datasets/andrewmvd/lits-png)
- [Part 2](https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation-part-2)
  
## Data preprocessing

The dataset contains 30 NIfTI (.nii) files: 130 thoracic 3D scans and 130 segmentations 3D scans:

<p align="center">
  <img src="/result_images/3D_torax.png" width="300">
</p>

Using the nibabel library, the 3D scans are turned into 2D slices: 

<p align="center">
  <img src="/result_images/original_image.png" width="300">
</p>

After getting rid of all the scans not containing liver, there are 18.915 scans remaining.
Then, the slices and masks are processed to be of better quality: 

<p align="center">
  <img src="/result_images/46.png" width="300">
  <img src="/result_images/46(1).png" width="300">
</p>

The images are windowed according to the [dicom parameters](https://towardsdatascience.com/a-matter-of-grayscale-understanding-dicom-windows-1b44344d92bd) for the liver to be more visible:

<p align="center">
  <img src="/result_images/enhanced_image.png" width="300">
</p>

The scans are labeled into two classes (Sane, Tumor) using the masks: 

<p align="center">
  <img src="/result_images/classification.png" width="300">
</p>

Finally, the scans and masks are divided in train (60%), validation (20%) and test (20%)

## Model training

- 1st model: simple CNN model

structure : 
- Three convolutional + Max Pooling layers (relu)
- One 512 neurons dense layer (relu)
- One 1 neuron dense layer (sigmoid)
<p align="center">
  <img src="/result_images/cnn_curves.png" width="800">
</p>

<figure>

  <!-- Train confusion matrix -->
  <p align="center">
    Train confusion matrix
    <br>
    <img src="/result_images/cnn_cm_train.png" width="517" alt="Matrice de confusion (Entraînement)">
  </p>

  <!-- Test confusion matrix -->
  <p align="center">
    Test confusion matrix
    <br>
    <img src="/result_images/cnn_cm.png" width="500" alt="Matrice de confusion">
  </p>

</figure>


- 2nd model: pretrained VGG16

<p align="center">
  <img src="result_images/pretrained_vgg16_model.jpg" width="500">
</p>

<p align="center">
  <img src="result_images/training_validation_metrics.png" width="500">
</p>

<p align="center">
  <img src="result_images/confusion_matrix.png" width="500">
</p>

Clear overfitting issue: quickly reaches >95% accuracy on test and validation but a much lower test accuracy.
Reasons: Structure may be too complex for the project.
Possible solutions: Regularization: Dropout, data augmentation...

- 3rd model: not pretrained VGG16


# Performance summary 

|          Model           | Accuracy |
| ------------------------ | -------- |
|    Simple CNN Model      |    77%   |
|    Pretrained VGG16      |    54%   |
|    Not pretrained VGG16  |          |

# References

- TWIMLfest: Fundamentals of Medical Image Processing for Deep Learning, Sam Charrington
- Tenebris97 on Kaggle
