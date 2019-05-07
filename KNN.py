## K_nearest_neighbours_database-DAAL

# import necessary libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

x=['a1','a2','a3','a4','a5','a6']
dataset=pd.read_csv('/opt/intel/compilers_and_libraries_2018.2.199/linux/daal/examples/data/batch/k_nearest_neighbors_train.csv',names=x)
#dataset.head()
#dataset.info()


## collect Data in X and y

X_train=dataset.iloc[:, :-1]
y_train=dataset.iloc[:, -1]
# train/test split function
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)


# Creating & Training KNN Model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train,y_train)



# Price prediction of Test.csv Using KNN for Prediction
x=['a1','a2','a3','a4','a5','a6']
data_test=pd.read_csv('/opt/intel/compilers_and_libraries_2018.2.199/linux/daal/examples/data/batch/k_nearest_neighbors_train.csv',names=x)
#data_test.head()

datatest=data_test.iloc[:, :-1]
Ground_truth=data_test.iloc[:,-1]

# Applying KNN on Test data to predict price range
predicted_price=knn.predict(datatest)

#for x, y in zip(Ground_truth, predicted_price):     # side by side display of expected and predicted value
#    print(x, y, sep='\t\t')


from sklearn.metrics import f1_score
f1_score(Ground_truth, predicted_price, average='weighted')
