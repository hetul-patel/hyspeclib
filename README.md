Developed as a part of B.Tech Dissertation.

Code Manual : <https://drive.google.com/open?id=1hLnw2yNUpvKDbErnlKUZbUSA2DjESjyA>

Thesis soft copy : <https://drive.google.com/open?id=1PQTapftFkTBi9-NZ1ixLuPSGxkn5h-ZB>

Hyspeclib is a helper library written in python language on top of Tensorflow
backend for analysis of hyperspectral images and perform classification task
using supervised deep learning algorithm.

Hyspeclib includes helper functions for performing preprocessing on raw
images such as noisy bands identification and removal. It provides utility for
extracting spectral signatures and assigning class labels to it from ROI
images selected using ENVI for supervised learning task.
Training data can be visualised in 2D and quality of training dataset can be
improved by excluding outliers using hyspeclib modules. Two dimensionality
techniques can be implemented on any hyperspectral image for selecting
optimal number of bands. Band reduction not only reduces the size of image
but also reduces the processing time and improves the accuracy. It is also
possible to measure class to class separability using selected band
combination with JM-Distance algorithm provided here.
Selecting large number of training points for each class ( for e.g crops )
manually using softwares is time consuming. Using data augmentation utility
provided by hyspeclib, large number of high quality training samples can be
selected automatically from the image itself provided few manually selected
samples.

Deep learning requires a well tuned model in terms of optimal number of
layers, optimal number of nodes in each layer, proper learning rate and many
other hyper parameters. Hyspeclib provides a very easy way to design a
network with desired parameters, desired number of layer and nodes for
experimentation to achieve best classification accuracy based on images.
Other than classification, validation is also important aspect in any
supervised learning algorithm. Hyspeclib can calculate overall accuracy,
average class accuracy, confusion matrix, kappa coefficient for training and
blind site (testing data).

Hyspeclib is designed run on both GPU based and CPU based systems with
python and required packages installed. For images having larger size than
RAM (Computer memory), images are processed in smaller blocks so that
entire algorithm can run on entire image without memory issues.
Colourful classification mask can be generated and saved for hyperspectral
image to visualise predicted class for given region of image.

## Citation

@INPROCEEDINGS{8897897,  
author={H. {Patel} and N. {Bhagia} and T. {Vyas} and B. {Bhattacharya} and K. {Dave}},  
booktitle={IGARSS 2019 - 2019 IEEE International Geoscience and Remote Sensing Symposium},   
title={Crop Identification and Discrimination Using AVIRIS-NG Hyperspectral Data Based on Deep Learning Techniques},   
year={2019},  
volume={},  
number={},  
pages={3728-3731},}
