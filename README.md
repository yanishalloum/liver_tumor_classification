# Liver Tumor Classification

## About

This project aims at detecting tumors in liver scans using deep learning models trained on a large dataset.


## Models with their Accuracy of Prediction

|          Model           | Accuracy |
| ------------------------ | -------- |
|    Simple CNN Model      |          |
|    Pretrained VGG16      |          |
|    Not pretrained VGG16  |          |


## Guideline

1. Clone or download the repo.

```
git clone https://github.com/yanishalloum/liver_tumor_classification.git
```

2. Install all the dependencies:

```
pip install -r requirements.txt
```

3. Install the database:

## Dataset Links

I used the "Liver Segmentation Database" from Kaggle:

- [Part 1](https://www.kaggle.com/datasets/andrewmvd/lits-png)
- [Part 2](https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation-part-2)
  
## Data preprocessing

The dataset contains 30 NIfTI (.nii) files: 130 thoracic 3D scans and 130 segmentations 3D scans:

![Thoracic scan](/result_images/3D_torax.png)

Using the nibabel library, the 3D scans are turned into 2D slices: 

![Slice](/result_images/original_image.png) 

After getting rid of all the scans not containing liver, there are 18.915 scans remaining.
Then, the slices and masks are processed to be more visible: 

![Processed slice](/result_images/46.png) 
![Processed mask](/result_images/46(1).png) 


# References

- TWIMLfest: Fundamentals of Medical Image Processing for Deep Learning, Sam Charrington
- Tenebris97 on Kaggle
