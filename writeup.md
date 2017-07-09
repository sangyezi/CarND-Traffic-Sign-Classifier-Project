#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

<!--[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./GermanTrafficSigns/original/3_speed_60.jpeg "Traffic Sign 1"
[image5]: ./GermanTrafficSigns/original/11_right-of-way.jpg "Traffic Sign 2"
[image6]: ./GermanTrafficSigns/original/14_stop.jpg "Traffic Sign 3"
[image7]: ./GermanTrafficSigns/original/25_road_work.jpeg "Traffic Sign 4"
[image8]: ./GermanTrafficSigns/original/28_children_crossing.jpg "Traffic Sign 5"
[image9]: ./GermanTrafficSigns/original/28_children_crossing2.jpg "Traffic Sign 6"-->

[image4]: ./GermanTrafficSigns/3_speed_60.jpg "Traffic Sign 1"
[image5]: ./GermanTrafficSigns/11_right-of-way.jpg "Traffic Sign 2"
[image6]: ./GermanTrafficSigns/14_stop.jpg "Traffic Sign 3"
[image7]: ./GermanTrafficSigns/25_road_work.jpg "Traffic Sign 4"
[image8]: ./GermanTrafficSigns/28_children_crossing.jpg "Traffic Sign 5"
[image9]: ./GermanTrafficSigns/28_children_crossing2.jpg "Traffic Sign 6"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/sangyezi/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43


####2. Include an exploratory visualization of the dataset.

Please refer to `Include an exploratory visualization of the dataset` section of  my [project code](https://github.com/sangyezi/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)
 or [the html exported from it](https://github.com/sangyezi/CarND-Traffic-Sign-Classifier-Project/blob/master/html/Traffic_Sign_Classifier.html) for the visualizations of the dataset and individual images.

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it was suggested by the ConvNet paper. 

As a last step, I normalized the image data because it helps to construct a model with good numeric stability.

I did not generate additional data because I did not view the individual images  until I finished the training and testing of the model. After viewing it, I realize how close images with the same labels are. I should generate additional data by agumentation techniques. Also, I would image if I could use segmentation techniques to identify sign area, and add noise to the rest of the area, I could generate more valuable images for training.




####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray image   | 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x18 |
| RELU					|						|
| Max pooling	      	| 2x2 stride,  outputs 14x14x18 |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 |
| RELU					|						|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 |
| Fully connected		| output 360|
| RELU					|						|
| Fully connected		| output 336|
| RELU					|						|
| Fully connected		| output 43|
| Softmax				| |        		 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used `epochs = 50, batch_size = 32, learning_rate = 0.001`, the weights are generated using `tf.truncated_normal` with `mu = 0, sigma = 0.1`. I introduced l2 regularization in the model, with `beta = 0.02`, in addition, I used 70% dropout in the training. Just as LeNet model, the AdamOptimizer was used to minimize the reduced mean of the softmax cross entropy with logis.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* training set accuracy of 99.7%
* validation set accuracy of 97.6%
* test set accuracy of 94.8%

If an iterative approach was chosen:

* What was the first architecture that was tried and why was it chosen?
	* I started with the LeNet model. To allow easy adjusting model parameters and hyperparameters, I refactored the code of LeNet model, created `LeNet` and `Pipeline` classes, represent the model and pipeline correspondingly, and `experiment` functions to training the model with various parameters.
	* My strategy to pick good parameters is fixed all the parameters, allow only one changes, and pick its best value. Repeat such step for parameters. Then adjust the fixed values to optimized parameter, iterate the optimization again. Here both model parameters and hyperparameters were optimzed.

	
* What were some problems with the initial architecture?
	* I achieved 91% accuracy with validation data set by adjusting the parameters to `sigma=0.01, batch_siz=32, conv3_depth=60, conv4_depth=42, conv1_filter_length=7, pool2_k=3`. All the adjustments point to use less parameters, which means LeNet model is overfitting.
	
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
	* To fix the overfiting problem, I first tried l2 regularization, and l2 regularzation + dropout, repeat my strategies to the new models. With only l2 regularization, I could reach 93% accuracy with the validation dataset; with both both l2 regularization and dropout, I could reach almost 95% accuracy. The optmized parameters are very similar between the two, both prefer more parameters at layer 3 and layer 4; besides beta for the regularization, where only l2 prefers a smaller beta 0.0005, where l2 with dropout perfer a larger beta 0.001. Of note, I could not reach good results with 50% dropout rate with this model, so I chose 30% dropout rate (70% keep probability). To use higher dropout rate, I think I will need to increase the model size by adding more layers.

* Which parameters were tuned? How were they adjusted and why?
	* The final model with both l2 regularization and ropout adjust the following parameters to `keep_prob=0.7, epochs=50, beta=0.001, sigma=0.1, batch_siz=32, conv1_output_depth=18, conv3_depth=360,conv4_depth=336` (the rest parameters are the same as LeNet model). We can see the model prefers more parameters in the depth of each layer. Higher depth is likely needed for capture keep information for various labels.
	 
	 
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
	*	The convolution layer work well with the problem, because it allows recognize patterns at various locations of images. The l2 regularization and dropout layer helps with creating a successful model by allowing more flexibility and redudancy in the model, and eliminate the overfit problem.

If a well known architecture was chosen:

* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]

The last image might be difficult to classify because its background is more complicated, and it contains a watermark.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100 km/h      		    | 100 km/h    | 
| right-of-way at the next intersection| right-of-way at the next intersection |
| Stop sign					| Stop sign	|
| Road work             | Road work  |
| children corssing      | children corssing  |
| children corssing  2	 | Road work    |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 83.3%. This compares favorably to the accuracy on the test set of 94.8%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model predicts it is a Right-of-way sign (probability of 1.0000), and the image does contain a right-of-way sign. The top five soft max probabilities were

| Probability         	|     Prediction	  | 
|:---------------------:|:---------------------------------------------:| 
| 1.0000         			| Right-of-way at the next intersection   | 
| .0000     				| Beware of ice/snow|
| .0000					| Pedestrians 	|
| .0000	      			| Children crossing |
| .0000				    | Double curve |


For the second image, the model predicts this is a Speed limit (60km/h) sign (probability of 0.9999), and the image does contain a Speed limit (60km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	  | 
|:---------------------:|:---------------------------------------------:| 
| .9999         			| Speed limit (60km/h)  | 
| .0001     				| Speed limit (80km/h) |
| .0000				| Speed limit (50km/h) |
| .0000	      			| End of all speed and passing limits|
| .0000				    | Ahead only |

For the third image, the model predicts this is a Stop sign (probability of 0.5954), and the image does contain a Stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	  | 
|:---------------------:|:---------------------------------------------:| 
| .5954         			| Stop   | 
| .3583     				| Speed limit (60km/h)|
| .0087					| End of all speed and passing limits|
| .0085	      			| Turn right ahead|
| .0075				    | Traffic signals |

For the fourth image, the model predicts this is a Road work sign (probability of 0.9994), and the image does contain a Road work sign. The top five soft max probabilities were

| Probability         	|     Prediction	  | 
|:---------------------:|:---------------------------------------------:| 
| .9994         			| Road work   | 
| .0003     				|  General caution|
| .0001					| Keep left|
| .0001	      			| Right-of-way at the next intersection|
| .0000				    | Bicycles crossing |

For the fifth image, the model predicts that this is a Children crossing sign (probability of 0.9835), and the image does contain a Children crossing sign. The top five soft max probabilities were

| Probability         	|     Prediction	  | 
|:---------------------:|:---------------------------------------------:| 
| .9835         			| Children crossing   | 
| .0078     				| Dangerous curve to the right|
| .0039					| End of speed limit (80km/h)|
| .0024	      			| Right-of-way at the next intersection|
| .0009				    | Go straight or right|

For the sixth image, the model could not tell this is a Children crossing sign at its top five soft max probabilities.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
Visualization of the first layer captures lines and edges of the images, and visualization of the second layer (not sure in the notebook) captures the patch pattern of the images.

