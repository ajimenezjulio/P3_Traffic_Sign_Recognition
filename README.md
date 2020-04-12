## Traffic Sign Recognition
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project we built a traffic sign classifier based on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) using a deep neural network trained in tensorflow.


The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

---

### 1. Examine the dataset

The dataset used for this project is [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) and was splitted in the following way.

| Dataset         	|     Number of samples	        | 
|:-----------------:|:-----------------------------:| 
| Training        |   		34,799		  	| 
| Validation      |  	    4,410         |
| Test				    |				12,630				|
| Unique classes	|				43			    	|

Every image has a dimension of **32 x 32 x 3** (widht, height, channels), below a representation of the dataset is shown.

![Data_Representation](https://github.com/ajimenezjulio/P3_Traffic_Sign_Recognition/blob/master/Markdown/Data_Representation.png)


### 2. Dataset distribution analysis

Unfortunately the dataset is unbalanced and some classes have a large number of samples while others only a few, the following table and graph show this disparity.

| Dataset         	|   Maximum Samples	   |   Minimum Samples	   | 
|:-----------------:|:--------------------:|:---------------------:| 
| Training        |  2,010  |  180  |
| Test            |  750    |  60  |

![Data_Distribution_Before_Balancing](https://github.com/ajimenezjulio/P3_Traffic_Sign_Recognition/blob/master/Markdown/Data_Distribution_Before_Balancing.png)


### 3. Preprocess

The following steps were performed to augment, distort and perturb the dataset.

1. **Data augmentation** by flippiing it (horizontally, vertically and both). We need to be aware that some classes must be classified as a different class when flipped, e.g, turn left and right classes.
1. **Balance** the data to 2,000 samples per class by extending each class with random images from the same class.
1. **Perturb** 80 percent of the images by randomly distort them and modifying the brightness, contrast and saturation levels.

These were the resultant images at each augmentation step in the training set:

| Step       	|     Number of images	        | 
|:-----------------:|:-----------------------:| 
| Augment by flipping       |   		59,788		  	| 
| Augment by balancing      |  	    86,000        |

![Data_Distribution_Before_Balancing](https://github.com/ajimenezjulio/P3_Traffic_Sign_Recognition/blob/master/Markdown/Data_Distribution_After_Balancing.png)


### 4. Model

The model used for this project is a modified version of the LeNet architecture, each layer is described below:

![Lenet](https://github.com/ajimenezjulio/P3_Traffic_Sign_Recognition/blob/master/Markdown/Lenet.png)

| Layer       	    |     Input	        |     Output	        | 
|:-----------------:|:-----------------:|:-------------------:|
| Convolution 1 - 5 x 5    | 32 x 32 x 3 (RGB Image) |  32 x 32 x 16 |
| MaxPool 2x2    | 32 x 32 x 16 |  16 x 16 x 16 |
| - | - | - |
| Convolution 2 - 5 x 5    | 16 x 16 x 16 |  16 x 16 x 32 |
| MaxPool 2x2    | 16 x 16 x 32 |  8 x 8 x 32 |
| - | - | - |
| Convolution 3 - 5 x 5    | 8 x 8 x 32 |  8 x 8 x 64 |
| MaxPool 2x2    | 8 x 8 x 64 |  4 x 4 x 64 |
| - | - | - |
| Flatten    | 4 x 4 x 64 |  1 x 1024 |
| - | - | - |
| Dense 1   |1 x 1024 |  1 x 84 |
| - | - | - |
| Dense 2   |1 x 84 |  1 x 43 |

Additionally every activation function and droput value is shown below:

| Layer       	    |     Activation	        |     Dropout	        | 
|:-----------------:|:-----------------:|:-------------------:|
| Convolution 1    | ReLU | 0.9 |
| Convolution 2    | ReLU | 0.6 |
| Convolution 3    | ReLU | 0.6 |
| Dense 1    | ReLU | 0.5 |


### 4.1 Accuracy

In order to get reproducible results a seed of **17** was used in this project, also a batch size of **128** was implemented. The table below contains the accuracy results at different epochs in the network.

| Epoch | Training set | Validation set | Test set |
|:-----:|:------------:|:--------------:|:--------:|
| 1 | 0.031814 | 0.007483 | 0.007047 |
| 5 | 0.256081 | 0.265306 | 0.283769 |
| 10 | 0.559570 | 0.537642 | 0.539667 |
| 50 | 0.962407 | 0.953515 | 0.948614 |
| **200** | **0.984674** | **0.976417** | **0.975693** |

![Accuracy_Training](https://github.com/ajimenezjulio/P3_Traffic_Sign_Recognition/blob/master/Results/3conv-2fc%203-16-32-64-1024-84-43__0.0005_128_Final.jpeg)

After 200 epochs the model could get above 97% of accuracy for all datasets which makes it a fairly decent approach.

### Feature maps

In order to understand the weights the model is using for the classification task in a better way, it is a good idea to visualize them, so we can have an idea of the abstract patterns the model is selecting. Below samples of the feature maps are displayed.

![Feature_Maps](https://github.com/ajimenezjulio/P3_Traffic_Sign_Recognition/blob/master/Markdown/Feature_Maps.png)
