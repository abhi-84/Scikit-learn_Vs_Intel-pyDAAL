import os
import sys
import pandas as pd

from daal.algorithms.svm import training, prediction
from daal.algorithms import kernel_function, classifier
from daal.data_management import (
    DataSourceIface, FileDataSource, HomogenNumericTable, MergedNumericTable, NumericTableIface
)

#utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
#if utils_folder not in sys.path:
#    sys.path.insert(0, utils_folder)
from utils import printNumericTables

# split original single database file into 'test.csv' and 'train.csv' files and remove labels
# read CSV file and convert "Class" label into 2 classes: 1 & -1. two_class_SVM in pyDAAL accept output in -1 & 1 only 

x=pd.read_csv('/home/abhi/Desktop/Intel-HPC/ml_algos/ML_casestudy/cancer_cells/train.csv')
x['Class'].replace(
        to_replace=2,
        value=1,
        inplace=True)
x['Class'].replace(
        to_replace=4,
        value=-1,
        inplace=True)
x.drop(['ID'],axis=1,inplace=True)
x.to_csv('/home/abhi/Desktop/Intel-HPC/ml_algos/ML_casestudy/cancer_cells/pyDAAL/train.csv',index=False, header=None)

x=pd.read_csv('/home/abhi/Desktop/Intel-HPC/ml_algos/ML_casestudy/cancer_cells/test.csv')
x['Class'].replace(
        to_replace=2,
        value=1,
        inplace=True)
x['Class'].replace(
        to_replace=4,
        value=-1,
        inplace=True)
x.drop(['ID'],axis=1,inplace=True)
x.to_csv('/home/abhi/Desktop/Intel-HPC/ml_algos/ML_casestudy/cancer_cells/pyDAAL/test.csv',index=False, header=None)

DATA_PREFIX = os.path.join('/home/abhi/Desktop/Intel-HPC/ml_algos/ML_casestudy/', 'cancer_cells', 'pyDAAL')

trainDatasetFileName = os.path.join(DATA_PREFIX, 'train.csv')
testDatasetFileName = os.path.join(DATA_PREFIX, 'test.csv')

nFeatures = 9

# Parameters for the SVM kernel function
kernel = kernel_function.linear.Batch()

# Model object for the SVM algorithm
trainingResult = None
predictionResult = None
testGroundTruth = None


def trainModel():
    global trainingResult

    # Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
    trainDataSource = FileDataSource(
        trainDatasetFileName, DataSourceIface.notAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )

    # Create Numeric Tables for training data and labels
    trainData = HomogenNumericTable(nFeatures, 0, NumericTableIface.doNotAllocate)
    trainGroundTruth = HomogenNumericTable(1, 0, NumericTableIface.doNotAllocate)
    mergedData = MergedNumericTable(trainData, trainGroundTruth)

    # Retrieve the data from the input file
    trainDataSource.loadDataBlock(mergedData)

    # Create an algorithm object to train the SVM model
    algorithm = training.Batch()

    algorithm.parameter.kernel = kernel
    algorithm.parameter.cacheSize = 600000000

    # Pass a training data set and dependent values to the algorithm
    algorithm.input.set(classifier.training.data, trainData)
    algorithm.input.set(classifier.training.labels, trainGroundTruth)

    # Build the SVM model
    trainingResult = algorithm.compute()


def testModel():
    global predictionResult, testGroundTruth

    # Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from a .csv file
    testDataSource = FileDataSource(
        testDatasetFileName, DataSourceIface.notAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )

    # Create Numeric Tables for testing data and labels
    testData = HomogenNumericTable(nFeatures, 0, NumericTableIface.doNotAllocate)
    testGroundTruth = HomogenNumericTable(1, 0, NumericTableIface.doNotAllocate)
    mergedData = MergedNumericTable(testData, testGroundTruth)

    # Retrieve the data from input file
    testDataSource.loadDataBlock(mergedData)

    # Create an algorithm object to predict SVM values
    algorithm = prediction.Batch()

    algorithm.parameter.kernel = kernel

    # Pass a testing data set and the trained model to the algorithm
    algorithm.input.setTable(classifier.prediction.data, testData)
    algorithm.input.setModel(classifier.prediction.model, trainingResult.get(classifier.training.model))

    # Predict SVM values
    algorithm.compute()

    # Retrieve the algorithm results
    predictionResult = algorithm.getResult()


def printResults():

    printNumericTables(
        testGroundTruth, predictionResult.get(classifier.prediction.prediction),
        "Ground truth\t", "Classification results",
        "SVM classification results (first 20 observations):", flt64=False
    )

if __name__ == "__main__":

    trainModel()
    %timeit testModel()
    printResults()
