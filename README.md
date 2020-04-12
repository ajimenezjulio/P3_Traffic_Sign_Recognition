## Traffic Sign Recognition
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project we built a traffic sign classifier based on the German traffic sign dataset using a deep neural network trained in the tensorflow framework.


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
