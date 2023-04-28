# Galaxy Classification using Machine Learning
**Note**: Some figures below can not be fully seen with GitHub's dark theme. Please click on them for additional details.
## Abstract
With the new James-Webb telescope, science & technology allow us to probe deeper into space than ever before. As we survey and collect heaps of data from space, it is important to efficiently and accuratly analyze such data. Particularly, space data consists of telescopic images that can be classified and analyzed using modern machine learning techniques. In this project, we use PCA, CNNs, and transfer-learning to classify different types of galaxies.

## Introduction
Every galaxy has a "morphology," or a shape, that scientists think is determined by how the galaxy originally formed. Astronomers and cosmologists have split all galaxies into ten different major classes based on the shape of these galaxies. Better understanding of the distribution of galaxies across these classes could provide valuable insights into the formation of the universe.

The big challenge around automating this process is that the "shape" of a galaxy is somewhat subjective. Indeed, when volunteers classified images of galaxies dependent upon shape, there were results with a large diversity of opinion from human classifiers. If we can't get humans to agree on a "correct" classification of a galaxy, how are we to expect a machine to be able to do so?

To tackle this, we first explore a classical approach, PCA coupled with logistic regression, to serve as a benchmark for our models. Then, we explore Convolutional Neural Networks (CNNs). CNNs are some of the best-suited methods for image and video data with their grid-like architecture. Given that we use a dataset of galaxy images, it is only natural we use CNNs for most of our analysis. We choose to first build a simple CNN from the ground up. Then, we leverage transfer-learning and explore state-of-the-art image classification models like Resnet50 and Inceptionv3.

There are certain limitations to the extent to which we can train our models. First, CNNs generally rely on having immense datasets. While our current dataset is large, it is a limiting factor to the test accuracies we can achieve. CNNs also have a strong tendency to overfit, which would be another pitfall to avoid. A second limitation is that the pre-trained models we use are trained on ImageNet, which does not contain galaxy-like structures. This limits the learning that is transferred to our use case.

## Setup
**Note**: To use our code, you must download the .h5 file data from [here](https://astro.utoronto.ca/~hleung/shared/Galaxy10/Galaxy10_DECals.h5) to your working directory.
### Dataset
We use the Galaxy10 dataset contains 17,736 color pictures of galaxies, each 256 x 256 pixels with 3 color channels, totaling about 2.7GB of data. This dataset was collected by a variety of researchers from across the world \[1]. The distribution of the galaxies in the dataset is as follow:

![](https://i.imgur.com/Q3axAJ9.png)

![](https://i.imgur.com/2xobWRz.jpg)

### Model Setups
#### PCA with Logistic Regression
First, we create a 80-20 training test split to optimize the principal components of the PCA and test performance.

We test principal components from 1 to 100 and generated a scree plot to determine the optimal principal components that capture most of the variance in our data:

![](https://i.imgur.com/6I6GzzY.png)

According the plot, the first 20 principal components capture most of the information and is therfore our parameter choice.

#### Simple Convolutional Neural Network
For our convolutional neural network, the structure that we settled on had two sets of convolutional layers, each followed by ReLU activations and then a max pooling layer, followed by three linear layers of decreasing size (15552, 576, 64, 10). Each linear layer had ReLU activation, except for the output layer which used LogSoftmax. When the model started overfitting, we included a dropout as well to help mitigate those effects.

We notice that our dataset is skewed, in that some labels have far more images than others. To augment our dataset, we notice that the classification of a galaxy does not depend on its orientation - a galaxy flipped horizontally still looks (and is) the same class of galaxy. Therefore, we produced all the flips and rotations of the galaxies, and used some of them depending on how lacking the corresponding class was. Before augmenting, the distribution of classes was this:
![](https://i.imgur.com/GjFAwO7.png)


After augmenting, the distribution of classes is this:
![](https://i.imgur.com/OVI06He.png)


In addition, we use some techniques to deal with overfitting:
- L2 weight decay ($\alpha = 0.2$)
- Dropout (0.25 after each fully-connected layer)
- Adding Gaussian noise to the training data

Gaussian noise is done by randomly sampling a Gaussian distribution $\sim\mathcal{N}(0, \beta)$, where $\beta$ is the "noise standard deviation". This value is simply some fraction of the standard deviation of the dataset; in our example it is $\beta = \frac{1}{5} \sigma$. A demonstration is shown below:
![](https://i.imgur.com/gTUPgcE.png)


#### Transfer-learning with CNNs

While we explore two different transfer-learning models (Resnet50 & Inceptionv3), we use the same set of parameters and optimizers.

With these models, we do a 90-10 train test split and augment our images using several transformations to improve performance:
- Random Resized Crop
- Random Horizontal Flip
- Normalization

As with the CNNs, we used a Stochastic Gradient Descent optimizer with an initial learning rate of 0.001 and momentum of 0.9. Additionally, we also used a learning rate schedular with a schedule step of 5 with gamma 0.1. A learning rate scheduler essentially decreases the learning rate after a certain number of epoch steps to hone into the cost function's minimum. Finally, we used a batch size of 32 trained over 20 epochs.

These hyperparameters and optimizers were chosen because they gave us the best results (compared to adam, no scheduler, more/less epochs).

*Resnet50*

Resnet50 is a high-perfoming CNN that is residual in the sense that it has shortcut connections that skips certain layers in order to avoid the vanishing gradient problem. It was trained on ImageNet, a dataset with over 1 million images.

According to datagen.tech \[2], Resnet50 has the following architecture:
- A 7×7 kernel convolution alongside 64 other kernels with a 2-sized stride.
- A max pooling layer with a 2-sized stride.
- 9 more layers—3×3,64 kernel convolution, another with 1×1,64 kernels, and a third with 1×1,256 kernels. These 3 layers are repeated 3 times. 
- 12 more layers with 1×1,128 kernels, 3×3,128 kernels, and 1×1,512 kernels, iterated 4 times.
- 18 more layers with 1×1,256 cores, and 2 cores 3×3,256 and 1×1,1024, iterated 6 times.
- 9 more layers with 1×1,512 cores, 3×3,512 cores, and 1×1,2048 cores iterated 3 times.


*Inceptionv3*

Inceptionv3 is another popular image classification model trained on ImageNet by Google. The model comprises of symmetric and asymmetric building blocks such as convolutions, pooling, concatenations, dropouts, and fully connected layers. Batch normalization is widely used, and Softmax is used for loss computation. The basic architecture is shown below \[3].
![](https://i.imgur.com/4gxu9vU.png)

For both transfer-learning models, an additional fully-connected layer with 10 outputs was added to the end as our dataset classifies 10 galaxies. The networks were then trained using Northeastern's Dicovery Cluster to leverage GPUs.

## Results
### PCA with Logistic Regression
PCA gave us a training accuracy score of 31% and a test accuracy score of 19%. It best classified unbarred loose spiral galaxies, but wasn't able to correctly classify a singal cigar smooth galaxy, as seen in the training and validation matrices below.
#### Training Confusion Matrix
![](https://i.imgur.com/T0JMtUl.png)

#### Validation Confusion Matrix
![](https://i.imgur.com/ecjTd4o.png)

### Simple CNN
As expected, the CNN outperformed the PCA and Logistic Regression model by quite a margin. The maximum accuracy achieved by the CNN was about 46.0%. 
##### Train and Test Loss Curves
![](https://i.imgur.com/Ju6Bh4a.png)

### CNNs with Transfer Learning
As expected, Resnet50 & Inceptionv3 far outperformed the other models.

#### ResNet50
Resnet50 gave us a maximum accuracy of about 75.1%.
##### Training Loss Curve
![](https://i.imgur.com/7EC9cXF.png)
##### Test Loss Curves
![](https://i.imgur.com/2h4psTi.png)
It is interesting that the test loss and the accuracy both increase at some point. This is not a case of overfitting, rather a shift in the model's confidence of correctly and incorrectly classified images and is not found in the Inceptionv3 model.
##### Validation Confusion Matrix
![](https://i.imgur.com/Gy2GwIB.png)
#### Inceptionv3
Inceptionv3 gave us a maximum accuracy of about 76.2%.
##### Training Loss Curve
![](https://i.imgur.com/ssDUad2.png)
##### Testing Loss Curves
![](https://i.imgur.com/tP1SgAX.png)
##### Validation Confusion Matrix
![](https://i.imgur.com/Qj8Od4v.png)

Both neural networks do a decent job at classifying the galaxies. Like the PCA, they poorly classify the Cigar Smooth galaxies. This is to be expected as the Cigar Smooth galaxies are least represented in the dataset. 

Interestingly, the Spiral galaxies are often incorrectly classified with one another. This is likely due to the structural similarities betweent the three types of galaxies and the subtlties in their differences.

#### Incorrectly Classified Galaxies by Resnet50
![](https://i.imgur.com/sY3j7TT.png)

Examining the above misclassified images, we realize that a lot of the images are even difficult for humans to classify. This hints to the fact that a lot of the dataset was labeled by volunteers. To add-on, we see several completely black images due to the image transformations, that make it impossible to make any kind of correct classification.

## Discussion
Overall, we received good results but there is room for improvement.

As expected, PCA with logistic regression gave us poor results compared to state-of-the-art CNNs. This is because image classification is highly nonlinear process and thus a linear approximation using logistic regression simply did not suffice.

The best result we got with the neural networks had a testing accuracy of 76%. Comparatively, original researchers of the dataset created models that were able to obtain 99% accuracy on the images \[4]. However, they had access to a much larger version of the dataset (314,000), which was a limitation of our approach as mentioned. Even with almost 18,000 images we didn't have enough data to upscale our model and have it converge without extreme overfitting. 


The main issue we think with the CNN was that it started overfitting, and we had to implement somewhat extreme dropdown and L2 regularization to prevent severe overfitting. Even then, the accuracy was only around 30%, so we also think our model might be too simple.

One potential remedy for this would have been the observation that we can rotate and flip the galaxies any way we wanlaxy. By doing this, we would have been able to multiply our dataset eight-fold for free. The risk involved is again overfitting, and given that we were already getting overfitting problems, this may not be the wisest decision.

This technique also could have helped balance out the dataset across all the classes. Since some of the classes of galaxies had very few images compared to others, those ones could have had disproportionately more of their flips sampled in the final dataset to balance out the label distribution in the augmented training set. This would help out all of our models, especially the simpler ones.

One other technique we could have used to reduce the overfitting problem would be adding gaussian noise to the images. Adding noise, especially if it changes epoch to epoch, would reduce the overfitting problem, as it becomes impossible for the model to memorize the dataset when it changes epoch to epoch.

We were satisfied with the results from transfer learning. The only drawback we had was that the FCNs were originally trained on IMAGENET, which contains pictures of everyday objects, but not galaxies. Transferring the learned patterns of cars and planes to various shapes of galaxies is perhaps difficult than the size of our dataset allowed for, so maybe using a FCN that was originally trained on more similar data would yield better results. All in all, we are happy with transfer learning's results.

## Conclusion and Future Work
We tested several models on galaxy dataset that gave us varying performances. While not meeting the expectations we originally had, these models are satisfactory for the size of our dataset. 

In future, we wish to experiment further with deeper CNNs, and with other training models. The reference paper\[4] was able to achieve 99% accuracy on a subset of the images using Bayesian convolution neural networks, so we would try and use that model too. Also, for transfer learning, we might try with other pretrained models, perhaps ones already trained on images of the sky.


## References:
\[1]: https://github.com/henrysky/Galaxy10
\[2] https://datagen.tech/guides/computer-vision/resnet-50/
\[3] https://cloud.google.com/tpu/docs/inception-v3-advanced
\[4]: https://arxiv.org/abs/2102.08414 
https://arxiv.org/abs/2102.08414 
