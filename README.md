# Spine Segmentation Training and Prediction

This project follows the binary segmentation example of Sathieshes fastMONAI notebook (Kaliyugarasan, S., & Lundervold, A. S. (2023). fastMONAI: A low-code deep learning library for medical image analysis. Software Impacts. https://doi.org/10.1016/j.simpa.2023.100583).

Setup a data folder with volumetric image files and matching volumetric (binary) label files:

```{bash}
.
├── dataset.json
├── imagesTr
│   ├── 019c3523ce0e4048407dbce0946a5c6559665398d8051ffbc6053f228e0e8586.nii.gz
│   ├── 0b33311caf923f9a669cd7d52fc7f04b19e35ff0b60be1964215127f6f75b796.nii.gz
│   ├── 1.3.6.1.4.1.45037.0061b3c200bb6b31566b6861abb49c8f668fe79000805.nii.gz
...
│   └── eff1ed223ebc7f63fb927a5bc2e071a331a12daff1031cbb9ee2dba574e50ae6.nii.gz
├── imagesTs
├── labelsTr
│   ├── 019c3523ce0e4048407dbce0946a5c6559665398d8051ffbc6053f228e0e8586.nii.gz
│   ├── 0b33311caf923f9a669cd7d52fc7f04b19e35ff0b60be1964215127f6f75b796.nii.gz
│   ├── 1.3.6.1.4.1.45037.0061b3c200bb6b31566b6861abb49c8f668fe79000805.nii.gz
...
│   └── eff1ed223ebc7f63fb927a5bc2e071a331a12daff1031cbb9ee2dba574e50ae6.nii.gz
├── models
│   └── spine-model.pth
├── spine_model.pkl
└── vars.pkl
```

The dataset.json file should contain the following information:

```{json}
{ 
    "name": "SpineSegmentation", 
    "description": "Spine segmentation from T2 weighted MRI",
    "tensorImageSize": "3D",
    "reference": "H.B.",
    "licence":"CC-BY-SA 4.0",
    "relase":"0.0 25/01/2024",
    "modality": { 
        "0": "MRI"
    }, 
    "labels": { 
        "0": "background", 
        "1": "spine body"
    }, 
    "numTraining": 131, 
    "numTest": 10,
    "training":[
        { "image": "./imagesTr/019c3523ce0e4048407dbce0946a5c6559665398d8051ffbc6053f228e0e8586.nii.gz", "label": "./labelsTr/019c3523ce0e4048407dbce0946a5c6559665398d8051ffbc6053f228e0e8586.nii.gz" },
        { "image": "./imagesTr/0b33311caf923f9a669cd7d52fc7f04b19e35ff0b60be1964215127f6f75b796.nii.gz", "label": "./labelsTr/0b33311caf923f9a669cd7d52fc7f04b19e35ff0b60be1964215127f6f75b796.nii.gz" },
        { "image": "./imagesTr/1.3.6.1.4.1.45037.0061b3c200bb6b31566b6861abb49c8f668fe79000805.nii.gz", "label": "./labelsTr/1.3.6.1.4.1.45037.0061b3c200bb6b31566b6861abb49c8f668fe79000805.nii.gz" },
...
        { "image": "./imagesTr/eff1ed223ebc7f63fb927a5bc2e071a331a12daff1031cbb9ee2dba574e50ae6.nii.gz", "label": "./labelsTr/eff1ed223ebc7f63fb927a5bc2e071a331a12daff1031cbb9ee2dba574e50ae6.nii.gz" } ],
    "test":[
        "./imagesTr/5972c95a8125a2cc1ac27d5987660fca06d763c375340aa131a893267c6fee97.nii.gz",
...
        "./imagesTr/1.3.6.1.4.1.45037.90b13ab0fe4d742b3598bab99dc27bb58d8440a972ba1.nii.gz"]
}
```

The models folder can remain empty, only after the model training we expect files inside. The same for the two pkl files. They are created by the python notebook performing the training.

### The next step - manual segmentation

The above setup is usually done using manual segmentation on a few datasets. Here we have 131, but even 10 datasets might be sufficient for an initial training procedure. Once we train our first model we will use it and obtain additional training data for a second iteration of our model training. Once we are happy with the performance of a model we will keep using it.

For this workflow I am using 3D Slicer. A script [slicer.sh](slicer.sh) is used to open up a training image and label pair. After manually editing the label is saved again in the labelsTr folder using the same filename. We can call the slicer.sh script using a number representing the training row in the dataset.json:

```{bash}
./slicer.sh 0
```

This will open the image ./imagesTr/019c3523ce0e4048407dbce0946a5c6559665398d8051ffbc6053f228e0e8586.nii.gz with the label file ./labelsTr/019c3523ce0e4048407dbce0946a5c6559665398d8051ffbc6053f228e0e8586.nii.gz.

Note: After editing the label file (Segmentation Editor) you may need to convert the visible label first into a binary label map (right-click), from there you can "Export to file". Make sure you store the resulting label file using the same filename overwriting(!) the previous version of the label nii.gz.

## Training

Use the [SpineSegmentation.ipynb](SpineSegmentation.ipynb) notebook to train the model.
