# cmpe462hw1
This repository includes the source code for CMPE 462 HW1. </br>
## Install requirements
`pip install -r requirements.txt` </br>
## perceptron.py
This code was implemented for training a perceptron for large and small datasets given in the data folder. In order to run the code, run `python perceptron.py <size(small|large)>`. </br>
## logistic.py
This code was implemented for training a logistic regression model for the dataset named "Rice_Cammeo_Osmancik.arff" in the data folder. The model can be trained with gradient descent and stochastic descent algorithms. Also, l2 norm regularization can be applied. The functions called in lines 257, 258 and 259 were implemented for the 3rd, 4th and 5th questions of the related part. Uncomment the related line if you want to test by yourself. In order to run the code, run `python logistic.py`.</br>
## bayes.py
This code was implemented for training a naive bayes classifier for Breast Cancer Wisconsin Diagnostic dataset in the data folder. In order to run the code, run `python bayes.py`.
