# import necessary libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

x=['a1','a2','a3','a4','a5','a6','a7','a8','a9','a10','a11','a12','a13','a14','a15','a16','a17','a18','a19','a20','a21']
data=pd.read_csv('/opt/intel/compilers_and_libraries_2018.2.199/linux/daal/examples/data/batch/naivebayes_train_dense.csv',names=x)
## add column labels to the data fetched from csv file
data
#dataset.info()


## collect Data in X and y
X_train=data.iloc[:, :-1]
y_train=data.iloc[:, -1]

# train/test split function
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)


# Price prediction of Test.csv Using KNN for Prediction
x=['a1','a2','a3','a4','a5','a6','a7','a8','a9','a10','a11','a12','a13','a14','a15','a16','a17','a18','a19','a20','a21']
data_test=pd.read_csv('/opt/intel/compilers_and_libraries_2018.2.199/linux/daal/examples/data/batch/naivebayes_test_dense.csv',names=x)
#data_test.head()
datatest=data_test.iloc[:, :-1]
Ground_truth=data_test.iloc[:,-1]

# Applying KNN on Test data to predict price range
predicted=gnb.predict(datatest)

#print("Ground truth\t", "NaiveBayes classification results")
#for x1, y1 in zip(Ground_truth, predicted):     # side by side display of expected and predicted value
#    print(x1, y1, sep='\t\t\t')


from sklearn.metrics import f1_score
f1_score(Ground_truth, predicted, average='weighted')
