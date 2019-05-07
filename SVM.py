# import necessary libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


data=pd.read_csv('/home/abhi/Desktop/Intel-HPC/ml_algos/ML_casestudy/cancer_cells/train.csv')

#data
#dataset.info()
## collect Data in X and y
X = data[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
y= np.asarray(data['Class'])

# train/test split function
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)

from sklearn import svm
clf = svm.SVC(kernel='rbf',gamma='auto')
clf.fit(X_train, y_train)


# Price prediction of Test.csv Using SVM for Prediction

data_set=pd.read_csv('/home/abhi/Desktop/Intel-HPC/ml_algos/ML_casestudy/cancer_cells/test.csv')

#data_test.head()
datatest = data_set[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]

Ground_truth=np.asarray(data_set['Class'])

# Applying SVM on Test data to predict price range
y_hat=clf.predict(datatest)

#print("Ground truth\t", "SVM classification results")
#for x1, y1 in zip(Ground_truth, y_hat):     # side by side display of expected and predicted value
#    print(x1,y1, sep='\t\t\t')


from sklearn.metrics import f1_score
f1_score(Ground_truth, y_hat, average='weighted')
