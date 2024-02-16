# Classification Exam FISI-3650 

Welcome to the classification examination for the entrepreneurship course at Universidad de Los Andes. Your task is to develop a machine-learning model to identify rice varieties using image data.

The challenge consists of a directory named **Rice_Image_Dataset** that comprises training and testing data:
- The training data is contained in the **Train** folder, this folder contains a folder with images for each class.
- The testing data can be found in the **Test** folder; which consists of numerous images labeled with their respective class. The class of each image will be denoted by the initial string of the label.

<p align="center">
  <img src="./Rice_Image_Dataset/Test/Arborio (10001).jpg" />
</p>

## Context 
------
Computer vision is one of the fields that benefit greatly from machine learning, given that programming a series of criteria to classify an image manually is quite challenging. Real-world images exhibit high variability due to factors such as lighting conditions, perspective, camera settings, signal shapes, and others. Building a model from scratch and understanding why it works is a crucial skill for a machine learning engineer. Hence, we will assess your understanding of the subject by examining the quality of your code and challenging your comprehension of your own code. 

There are multiple methods to work with computer vision, such as [neural networks](https://www.investopedia.com/terms/n/neuralnetwork.asp#:~:text=A%20neural%20network%20is%20a,organic%20or%20artificial%20in%20nature), [k-means clustering](https://en.wikipedia.org/wiki/K-means_clustering) and [support vector machines](https://en.wikipedia.org/wiki/Support-vector_machine). We strongly recommend using neural networks for this challenge due to their demonstrated superior performance. However, we always welcome innovative approaches that show good results.

## Deliverable
------
Your deliverable is a Python script that must be called **model.py**; this file must have a class named Model that has a **predict(file_path)** method. The class will contain your pre-trained model and must load an image from any size specified in the **file_path** and return an integer indicating the predicted class for that image. 

Your model can be written in PyTorch, TensorFlow, scikit-learn, or can be entirely your own creation. Any other libraries will not be considered as your model will be tested automatically. 

## Evaluation
-----
We will evaluate the accuracy of your code against a larger test dataset. The accuracy will be determined by how many classes were correctly identified relative to the total number of files. The **Test** dataset used for the evaluation comprises 25000 images compared to the one provided in the repository, which contains 10000 images. Any code that takes more than 27 minutes to evaluate the entire dataset will be terminated, and the accuracy will be computed based on the images predicted within the 27-minute window.

This repository includes an **eval.py** script that will provide an accuracy score (higher is better). You can use it to test your solutions against the labeled examples. We will use this same script to evaluate your solution. Therefore it's imperative that you test your code, without modifying the eval scripts, so you can be certain it'll run on the evaluation stage. 

### **We will grade solution based on the mean accuracy of the entire submissions, the minimum accuracy needed to be accepted in the course is 65%.** 

Hints
------
- We have computed results with accuracies higher than 90% on the **Train** set. You can aim to get results with this score. 
- The highest accuracy we have reached for the **Test** set is 83.372%.
- Normalize all the images before parsing them into your model; images can have different shapes.
- Use the three color channels to parse info into your model.
- The training dataset is well-balanced and equally sized, and the images are highly detailed. We suggest you spend more time on the model than preprocessing the dataset. 
- We have seen that the denser the model, the better; however, balance it not to spend too much time on the evaluation. Remember **27 mins** is the max. 
- Check that your code runs on the eval script; if it doesn't run, we will not be able to give you a grade. Include the libraries you need to run the code. 

[Application form](https://forms.office.com/Pages/ShareFormPage.aspx?id=fAS9-kj_KkmLu4-YufucyuvPP8FxoDxPtQnJHZ3zr3NUMzdSOEs5UVBQMDBTSkJDNDdRMVNQTEpHNy4u&sharetoken=MQ2AdCM7zIO8yA5Hw98a)
-------
Use the link to pre-register for the course; we will have more communications for those who sign up.
