# Lung-Cancer-Detection-using-CNN
The aim of this project is to classify lung images into 3 categories: Normal,Lung Adenocarcinomas and Lung Squamous Cell Carcinomas using Convolutional Neural Network.<br>
## **Project and code Idea:**
https://www.geeksforgeeks.org/deep-learning/lung-cancer-detection-using-convolutional-neural-network-cnn/ <br>
## **Dataset:**
https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images<br>
The dataset consists of images which were generated from an original sample of HIPAA compliant and validated sources, consisting of 750 total images of lung tissue (250 benign lung tissue, 250 lung adenocarcinomas, and 250 lung squamous cell carcinomas) and augmented to 5,000 images belonging to each of the three categories. <br>
## **Training, Testing and Validation sets**:<br>
First split:<br>
•	Train = 80%<br>
•	Val = 20%<br>
Second split:<br>
•	Test = 10% of 80% = 8%<br>
•	Final Train = 72%<br>
Final Distribution:<br>
• Train → 72%<br>
•	Validation → 20%<br>
•	Test → 8%<br>
## **Results**:<br>
<img width="940" height="588" alt="image" src="https://github.com/user-attachments/assets/99fa9333-4ee7-46c7-b41c-c679a4b11956" /> <br>
From the plot it can be seen that both training and validation accuracy have reached above 90% in 8 epochs.<br>
**Classification metrics for validation set:** <br>
<img width="692" height="259" alt="image" src="https://github.com/user-attachments/assets/05de3146-b655-4ccd-8a3a-a9db4889de44" /><br>
**Classification metrics for test set:** <br>
<img width="940" height="288" alt="image" src="https://github.com/user-attachments/assets/fca97cde-4be5-4a4a-8175-c06b70a2b6f9" /> <br>
Both the classification metrics show good F1 score for all the classes.






