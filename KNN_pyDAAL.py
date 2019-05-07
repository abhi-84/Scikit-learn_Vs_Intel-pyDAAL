# file: kdtree_knn_dense_batch.py

import os
import sys

from daal.algorithms.kdtree_knn_classification import training, prediction
from daal.algorithms import classifier
from daal.data_management import (
    DataSourceIface, FileDataSource, HomogenNumericTable, MergedNumericTable, NumericTableIface
)

#utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
#if utils_folder not in sys.path:
#    sys.path.insert(0, utils_folder)
from utils import printNumericTables

DAAL_PREFIX = os.path.join('/opt/intel/compilers_and_libraries_2018.2.199/linux/daal/examples/', 'data')

# Input data set parameters
trainDatasetFileName = os.path.join(DAAL_PREFIX, 'batch', 'k_nearest_neighbors_train.csv')
testDatasetFileName = os.path.join(DAAL_PREFIX, 'batch', 'k_nearest_neighbors_test.csv')

nFeatures = 5

trainingResult = None
predictionResult = None


def trainModel():
    global trainingResult

    # Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
    trainDataSource = FileDataSource(
        trainDatasetFileName, DataSourceIface.notAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )

    # Create Numeric Tables for training data and dependent variables
    trainData = HomogenNumericTable(nFeatures, 0, NumericTableIface.doNotAllocate)
    trainGroundTruth = HomogenNumericTable(1, 0, NumericTableIface.doNotAllocate)
    mergedData = MergedNumericTable(trainData, trainGroundTruth)

    # Retrieve the data from input file
    trainDataSource.loadDataBlock(mergedData)

    # Create an algorithm object to train the KD-tree based kNN model
    algorithm = training.Batch()

    # Pass a training data set and dependent values to the algorithm
    algorithm.input.set(classifier.training.data, trainData)
    algorithm.input.set(classifier.training.labels, trainGroundTruth)

    # Train the KD-tree based kNN model
    trainingResult = algorithm.compute()


def testModel():
    global trainingResult, predictionResult

    # Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from a .csv file
    testDataSource = FileDataSource(
        testDatasetFileName, DataSourceIface.doAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )

    # Create Numeric Tables for testing data and ground truth values
    testData = HomogenNumericTable(nFeatures, 0, NumericTableIface.doNotAllocate)
    testGroundTruth = HomogenNumericTable(1, 0, NumericTableIface.doNotAllocate)
    mergedData = MergedNumericTable(testData, testGroundTruth)

    # Load the data from the data file
    testDataSource.loadDataBlock(mergedData)

    # Create algorithm objects for KD-tree based kNN prediction with the default method
    algorithm = prediction.Batch()

    # Pass the testing data set and trained model to the algorithm
    algorithm.input.setTable(classifier.prediction.data,  testData)
    algorithm.input.setModel(classifier.prediction.model, trainingResult.get(classifier.training.model))

    # Compute prediction results
    predictionResult = algorithm.compute()
    printNumericTables(
        testGroundTruth, predictionResult.get(classifier.prediction.prediction),
        "Ground truth", "Classification results",
        "KD-tree based kNN classification results (first 20 observations):", 20, flt64=False
    )

if __name__ == "__main__":

    trainModel()
    %timeit testModel()
