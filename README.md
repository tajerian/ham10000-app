# a new machine-learning based diagnostic tool for differentiation of dermatoscopic skin cancer images.
We developed an application to enable researchers to assess our skin cancer classification model and load their images to this app to get the results instantly.  Please note that this application has no clinical or diagnostic value and is for research purposes only.<br>
This computer vision program classify dermatoscopic images of skin cancer lesions. We have used the HAM10000 MNIST dataset to train my model. The model's backbone is EfficientNetB1 and a Global average pooling 2D layer is added on top.
<br>
Here are the results: accuracy on test set 84.3%
## Dataset description

This population contains information about 10015 participants with the mean age of 51.86±16.96 which was only reported for 9958 of cases.
The biological sex of the 5406 (54.1%) of participants were Male, 4552 (45.5%) were Female, and for 57 (0.6%) of them the gender was unknown.
The skin lesion images were taken from various parts of the body. The back with 2192(21.9%) images, the lower extremities with 2077(20.7 %) images,
the trunk with 1404(14.0%) images, the upper extremities with 1118(11.2%) images, and the abdomen with 1022(10.2) images were the 5 most involved parts of the body;
other ports prevalence in described in the Figure bellow.
The diagnosis for each lesion was confirmed via a specific route, 5340(53.3%) of lesions were confirmed through histopathologic examinations,
3704(37.0%) of lesions were confirmed by follow-up examinations, 902(9.0%) of lesions were confirmed by expert consensus, and the ground truth for 69(0.7%) of lesions was confirmation by in-vivo confocal microscopy.<br>
![Dataset description](https://github.com/tajerian/ham10000-app/blob/main/datset.png?raw=true)

## lesion classes

The following are the seven categories:
1. Actinic Keratosis [akiec]: Types of squamous cell carcinoma that are noninvasive and can be treated locally without surgery (327 images are available in the data set).
2. Basal Cell Carcinoma [bcc]: A type of epithelial skin cancer that seldom spreads but, if left untreated, can be fatal. (514 images are available in the data set).
3. Benign Keratosis-like Lesions [bkl]: Seborrheic keratosis, lichen-planus-like keratosis, and solar lentigo, correlate to a seborrheic keratosis or a sun lentigo with regression and inflammation, are all examples of “benign keratosis” (1099 images are available in the data set).
4. Dermatofibroma [df]: Skin lesions that are either benign growth or an inflammatory response to minor trauma (115 images are available in the data set).
5. Melanoma [mel]: Melanoma is a cancerous tumor that develops from melanocytes and can take many different forms. If caught early enough, it can be treated with a basic surgical procedure (1113 images are available in the data set).
6. Melanocytic Nevi [nv]: Skin lesions are benign neoplasms of melanocytes and appear in a variety of shapes and sizes. From a dermatoscopic standpoint, the variants may differ dramatically (6705 images are available in the data set).
7. Vascular Lesions [vasc]: Cherry antifoams, angiokeratomas, and pyogenic granulomas are examples of benign or malignant angiomas. (142 images are available in the data set).
![frequency description](https://github.com/tajerian/ham10000-app/blob/main/frequency.png?raw=true)
<br>

## Model architecture for Transfer Learning

It is generally not a good idea to train a very large Deep Neural Network from scratch as training such large models with at least 200 to 300 hidden layers requires a huge amount of resources and time not everyone has, instead using existing pretrained models that accomplishes a similar task is a much reasonable idea. Transfer learning is a research problem in machine learning that focuses on storing knowledge in process of solving one problem and applying the gained accumulated knowledge to accelerate the learning in new different but related tasks. Transfer learning nets are trained on large datasets and their model parameters of each layer could manually set to be frozen so that they won’t change during retraining.<br>The Efficient nets are a family of neural networks with the baseline model constructed with Neural Architecture Search technique. The EfficientNET-B1, a variant of baseline model EfficientNET-B0 which is created through compound scaling, is the backbone to our model. We deleted the top layer of EfficientNET-B1 then a Global average pooling 2D layer and a softmax layer with 7 nodes added on top. The model architecture is shown in the Figure bellow.
![model-arch description](https://github.com/tajerian/ham10000-app/blob/main/results/model_arch.png?raw=true)

## Image augmentation
In order to artificially increase the instances a data augmentation technique used to generating new sample images. This technique was consisting of a random width shift from -20% to +20% of image width and a random height shift from -20% to +20% of image height and a random max 0.2-degree shear angle in counter-clockwise direction to rectify the perception angle and also random horizontal and vertical flip was applied. Empty pixels then were filled by the nearest pixel. To better understand this procedure, we randomly applied data augmentation to 25 images for 3 times and the results are shown in the Figure bellow. Also this video provided in supplementary materials illustrates 600 frames of random augmentations for 9 lesions with frame rate of 10 FPS.
![data augmentation](https://tajerian.info/ftp/image-aug.gif)

## ML model performance
The Machine Learning model was trained and tested on Google Colaboratory environment with an Intel(R) Xeon(R) 2.30GHz CPU processor and 13GB of RAM and NVIDIA Tesla T4 CUDA enabled GPU processor with CUDA 11.2 which has designed for high-performance computing, deep learning training and inference, machine learning, and data analytics. The model was created with Python 3.8.6, and TensorFlow 2.11, Scikit-Learn 1.0.2, and Numpy as dependencies.<br>
We used an Adaptive Momentum (Adam) optimizer on Categorical Cross Entropy loss function with a dynamic learning rate (LR) starting from 0.001. For fine tuning in order to make the optimizer converge faster and get closer to the global minimum of the loss function, the learning was set high in early epochs and by getting closer to the global optimum the learning rates decreased to take tiny steps toward the global optimum, also we used the ReduceLROnPlateau callback to reduce the LR even more if the validation loss did not improve after 3 epochs. The metrics and LR for each epoch is described in the Table bellow.
|epoch|loss|accuracy|val_loss|val_accuracy|LR|
| :---:|:--:| :------:| :------:| :----------:|-:|
|0|0.8674|0.6979|0.7513|0.7250|1.0e-03|
|1|0.7089|0.7367|0.6836|0.7485|1.0e-03|
|2|0.6479|0.7608|0.6517|0.7600|1.0e-03|
|3|0.6077|0.7777|0.6297|0.7630|1.0e-03|
|4|0.5808|0.7844|0.6094|0.7760|1.0e-03|
|5|0.5550|0.7976|0.5935|0.7840|1.0e-03|
|6|0.5368|0.8055|0.5849|0.7850|1.0e-03|
|7|0.5186|0.8125|0.5781|0.7860|1.0e-03|
|8|0.5035|0.8180|0.5672|0.7855|1.0e-03|
|9|0.4886|0.8243|0.5623|0.7900|1.0e-03|
|10|0.4785|0.8292|0.5548|0.7940|1.0e-03|
|11|0.4668|0.8316|0.5463|0.7995|1.0e-03|
|12|0.4581|0.8386|0.5463|0.8025|1.0e-03|
|13|0.4370|0.8485|0.5367|0.8045|1.0e-04|
|14|0.4341|0.8509|0.5355|0.8050|1.0e-04|
|15|0.4593|0.8292|0.4886|0.8225|1.0e-04|
|16|0.2851|0.9018|0.4450|0.8390|1.0e-05|
|17|0.2522|0.9169|0.4387|0.8420|1.0e-06|
|18|0.2470|0.9177|0.4387|0.8430|1.0e-08|
<br><br><br><br>
The model’s Precisions and Recalls and F1 Scores is described and compared in Figure based on each class. As demonstrated in Figure, the model has the best performance in detecting melanocytic nevi lesions with the F1 score of 0.93. This performance difference between the classes is mainly due to the highly imbalance classes of the dataset. As the model get trained with lots of melanocytic nevi images which was about 5364 images (comparing to 92 dermatofibroma images), inevitably the model learns more patterns to detect this specific class and higher performance on this class.
![model-arch description](https://github.com/tajerian/ham10000-app/blob/main/metric.png?raw=true)
