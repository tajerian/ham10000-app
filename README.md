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
