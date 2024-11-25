# VineNet: Grasp of the Grapes

## Implementation of Deep Learning Framework -- U-Net, Using Tensorflow

> The architecture was inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)



## Overiew
### Data

The Original Dataset is Given by Yamaha Motor Solutions India Pvt. Ltd. for a hackthon Organised by the company in collobration with IIT Mandi.

You can find the data soon in this Github Repo itself.

This 719 Images were divided 500(Training), 119(Validation) and 100(Testing).


### Data Augumentation

The data for Traning contained 719 Images of size 1920*960, which are far not enough to feed a deep learning neural network. So, We ended up using the [Albumentations](https://pypi.org/project/albumentations/) library.

We ended up using 3 Argumentations techiques
1. Horizontal Flip
2. Coarse Dropout
3. RandomBrightnessContrast

After Data Argumentation we ended up with total 2219 Images i.e. 2000(Training), 119(Validation) and 100(Testing).

For more Details Refer to DataArgumentation.ipynb file.

### Model 

So the Model Architecture is as follows:

![image](https://github.com/Manty2503/VineNet/assets/119813195/e15171d1-38ca-476b-b386-408b61cd2824)

> **4 Encoder Blocks, 1 Concolution Block, 4 Decoder Blocks**

**Covolution Block** 
> 2 set of Convolution layers followed by Batch Normalization Layer with ReLu Activation.

**Encoder Block** 
> 1 Convolution Block along with a MaxPooling Layer.

**Decoder Block** 
> 1 Conv2DTranspose Layer and Concatinating the Skip Connection along followed by a Convolution Block.

Last, Decoder Block is has Sigmoid as activation function.

Input of the Network - 1920* 960 RGB Image

Output of the Netwrok - 1920* 960 Binary Image


### Training

The Model is trained for 100 epochs with various Callbacks implemented like CSVLogger, ReduceLROnPlateu, ModelCheckpoint and EarlyStopping.


Loss Function - BinaryCrossEntropy

Metrics - Accuracy

Optimizer - Adam


## How to Use

### Dependencies
This tutorial uses Tensorflow(2.11.0), OpenCV(cv2), Python(3.8).

You can modify this accordinally.

### Running the Model

DataArgumentation.ipynb - For Data Argumentations

U-Net.ipynb - To train the Model and save it as .h5 file.

Predict Mask.ipynb - To predict the Mask for Testing Dataset.

Evaluation.ipynb - To calculate the scores using various metrics like Recall, Precision, MeanIoU, F! and accuracy.

### Results

The Input Image

![image](https://github.com/Manty2503/VineNet/assets/119813195/f0dd1bc2-58e0-4e0c-84b2-bd4555507675)


The Generated Mask

![image](https://github.com/Manty2503/VineNet/assets/119813195/49556087-0350-4b58-940b-57ab949b80ba)



