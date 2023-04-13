# Galaxy Classification using Machine Learning

## Abstract:
We used PCA, CNNs, and ultimately transfer learning from a pre-trained CNN to predict the class of galaxies from an image of the galaxy. 

## Introduction:
Every galaxy has a "morphology," or a shape, that scientists think is determined by how the galaxy originally formed. Astronomers and cosmologists have split all galaxies into ten different major classes based on the shape of these galaxies. Better understanding of the distribution of galaxies across these classes could provide valuable insights into the formation of the universe. 

The big challenge around automating this process is that the "shape" of a galaxy is somewhat subjective. Indeed, when volunteers classified images of galaxies dependent upon shape, there were results with a large diversity of opinion from human classifiers. If we can't get humans to agree on a "correct" classification of a galaxy, how are we to expect a machine to be able to do so?

Nevertheless, we trained three models against the dataset of human-classified images. In other applications, several techniques, especially convolutional neural networks, have proved to yield useful results in classifying images of objects, so we hope to apply them here. However, they also rely on having a large dataset, which could potentially be a problem. They also have a strong tendency to overfit, which would be another pitfall to avoid.


## Setup:
Our dataset contained 17,736 color pictures of galaxies, each 256 x 256 pixels, totaling about 2.7GB of data. This dataset was collected by a variety of researchers from across the world. The following shows a distribution of the dataset:
![](https://i.imgur.com/Q3axAJ9.png)

![](https://i.imgur.com/2xobWRz.jpg)




We first ran PCA with Logistic Regression, using a grid search to optimize the number of components to pull out. Then we used a convolutional neural network as planned, tuning with the number of convolutional and pooling layers, as well as the number and size of the fully-connected layers, and some dropout. Lastly, we used a different pre-trained fully convolutional network and performed transfer learning on the dataset. All of the neural networks ran on the Northeastern discovery cluster, on a GPU.

For our convolutional neural network, the structure that we settled on had two sets of convolutional layers, each followed by ReLU activations and then a max pooling layer, followed by three linear layers of decreasing size (15552, 576, 64, 10). Each linear layer had ReLU activation, except for the output layer which used LogSoftmax. When the model started overfitting, we included a dropout as well to help mitigate those effects.









## Results:
### PCA with Logistic Regression
As a benchmark, we decided to try a PCA with 31 components, determined through validation, and running a logistic regression model on our dataset. As expected, we didn't get great results: our training error was about 33% and our validation error was about 18%.
##### Training Confusion Matrix
![](https://i.imgur.com/ycrugKg.png)
##### Validation Confusion Matrix
![](https://i.imgur.com/4xaXacr.png)

### Transfer Learning
Next, we decided to leverage transfer learning to use state of the art image classification models to classify the galaxies. As expected this group of models far outperformed the other models.

#### ResNet50
Gave us an accuracy of about 73%.
##### Training Loss Curve
![](https://i.imgur.com/dO0Plkk.png)
##### Testing Loss Curves
![](https://i.imgur.com/hurQD2y.png)
It is interesting that the test loss and the accuracy both increase at some point. This is not a case of overfitting, rather a shift in the model's confidence of correctly and incorrectly classified images.
##### Validation Confusion Matrix
![](https://i.imgur.com/U2TrUeL.png)

#### Inception
Gave us an accuracy of about 76%.
##### Training Loss Curve
![](https://i.imgur.com/qEY1Ylo.png)
##### Testing Loss Curves
![](https://i.imgur.com/L8FVlwc.png)
##### Validation Confusion Matrix
![](https://i.imgur.com/u1KF1BU.png)

We see that the Cigar Shaped galaxies were most commonly misidentified. This is to be somewhat expected as the training dataset had a disproportionate amount of labeled images for the Cigar shaped galaxies.

##### Incorrectly Classified Galaxies by Resnet50
![](https://i.imgur.com/sY3j7TT.png)



## Discussion:
Our results were great but there is room for improvement. The best result we got had a testing accuracy of 76%, whereas when the original researchers created models against this data, they were able to obtain 99% accuracy on images that human classifiers were confident on.[1] We don't ever expect PCA with Logistic Regression to perform on the level of the neural networks. 

The main issue we think with the CNN was that it started overfitting, and we had to implement somewhat extreme dropdown and L2 regularization to prevent severe overfitting. Even then, the accuracy was only around 30%, so we also think our model was too simple, but even with almost 18,000 images we didn't have enough data to upscale our model and have it converge without extreme overfitting. That is one advantage the authors of the paper had over us: their dataset had 314,000 galaxies. 

One potential remedy for this would have been the observation that we can rotate and flip the galaxies any way we wanlaxy. By doing this, we would have been able to multiply our dataset eight-fold for free. The risk involved is again overfitting, and given that we were already getting overfitting problems, this may not be the wisest decision.

This technique also could have helped balance out the dataset across all the classes. Since some of the classes of galaxies had very few images compared to others, those ones could have had disproportionately more of their flips sampled in the final dataset to balance out the label distribution in the augmented training set. This would help out all of our models, especially the simpler ones.

One other technique we could have used to reduce the overfitting problem would be adding gaussian noise to the images. Adding noise, especially if it changes epoch to epoch, would reduce the overfitting problem, as it becomes impossible for the model to memorize the dataset when it changes epoch to epoch.

We were satisfied with the results from transfer learning. The only drawback we had was that the FCNs were originally trained on IMAGENET, which contains pictures of everyday objects, but not galaxies. Transferring the learned patterns of cars and planes to various shapes of galaxies is perhaps difficult than the size of our dataset allowed for, so maybe using a FCN that was originally trained on more similar data would yield better results. All in all, we are happy with transfer learning's results.

## Conclusion
We tested several models on galaxy dataset that gave us varying performances.

## References:
[1]: https://arxiv.org/abs/2102.08414 
https://arxiv.org/abs/2102.08414 
