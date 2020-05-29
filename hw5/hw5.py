# -*- coding: utf-8 -*-
from scipy.io import arff
import pandas
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import neural_network
import numpy
import sys

pandas.options.mode.chained_assignment = None

# read data from source file
def readIrisData(filePath) -> pandas.DataFrame:
    #load data from arff file
    data = arff.loadarff(filePath)
    
    #put data into a dataframe for easier user and manipulation
    dataFrame = pandas.DataFrame(data[0])
    
    #strip whitespace and unwanted chars from class values
    dataFrame['class'] = dataFrame['class'].str.strip()
    dataFrame['class'] = dataFrame['class'].map(lambda x: str(x)[:-1])
    dataFrame['class'] = dataFrame['class'].map(lambda x: str(x)[2:])
    
    #convert class names to ints
    dataFrame['class'][dataFrame['class'] == 'Iris-setosa'] = 1
    dataFrame['class'][dataFrame['class'] == 'Iris-versicolor'] = 2
    dataFrame['class'][dataFrame['class'] == 'Iris-virginica'] = 3
    dataFrame['class'] = pandas.to_numeric(dataFrame['class'])
    
    #return dataframe for later use
    return dataFrame

def classifyWithNeuralNetwork(xTrain, xTest, yTrain, yTest, synthdf):
    print()
    print("Neural Network Classification: ")
    #train and test with iris data
    nn = neural_network.MLPClassifier(random_state=2, max_iter=10000)
    print(nn.fit(xTrain, yTrain))
    print("Neural Network score: " + str(nn.score(xTest, yTest)))
    
    #predict with generated data
    synthMatrix = synthdf.as_matrix()
    predicted = nn.predict(synthMatrix)
    
    #plot predicted classes
    plotScatter(synthMatrix[:,0], synthMatrix[:,1], predicted, 'sepallength', 'sepalwidth')
    plotScatter(synthMatrix[:,0], synthMatrix[:,2], predicted, 'sepallength', 'petallength')
    plotScatter(synthMatrix[:,0], synthMatrix[:,3], predicted, 'sepallength', 'petalwidth')
    plotScatter(synthMatrix[:,1], synthMatrix[:,2], predicted,  'sepalwidth', 'petallength')
    plotScatter(synthMatrix[:,1], synthMatrix[:,3], predicted,  'sepalwidth', 'petalwidth')
    plotScatter(synthMatrix[:,2], synthMatrix[:,3], predicted,  'petallength', 'petalwidth')
    print()
    
    
    
def classifyWithSVM(xTrain, xTest, yTrain, yTest, synthdf):
    print()
    print("SVM Classification: ")
    #train and test with iris data
    svc = svm.SVC(random_state=2)
    print(svc.fit(xTrain, yTrain))
    print("SVM Score: " + str(svc.score(xTest, yTest)))
    
    #predict with generated data
    synthMatrix = synthdf.as_matrix()
    predicted = svc.predict(synthMatrix)
    
    #plot predicted classes
    plotScatter(synthMatrix[:,0], synthMatrix[:,1], predicted, 'sepallength', 'sepalwidth')
    plotScatter(synthMatrix[:,0], synthMatrix[:,2], predicted, 'sepallength', 'petallength')
    plotScatter(synthMatrix[:,0], synthMatrix[:,3], predicted, 'sepallength', 'petalwidth')
    plotScatter(synthMatrix[:,1], synthMatrix[:,2], predicted,  'sepalwidth', 'petallength')
    plotScatter(synthMatrix[:,1], synthMatrix[:,3], predicted,  'sepalwidth', 'petalwidth')
    plotScatter(synthMatrix[:,2], synthMatrix[:,3], predicted,  'petallength', 'petalwidth')
    print()
    
def plotScatter(x, y, classes, feature1, feature2):
    plot.scatter(x, y, c=classes)
    plot.xlabel(feature1)
    plot.ylabel(feature2)
    plot.show()
    
def irisWithoutClass(df) -> pandas.DataFrame:
    return df.drop(columns=['class'])
    
#generate 300 observations for a given mean vector and covariance matrix
def generateSyntheticData(mean, cov) -> pandas.DataFrame:
    synth = numpy.random.multivariate_normal(mean, cov, 300)
    synthDF = pandas.DataFrame(synth)
    print("Generated Synthetic Data: ")
    print(synthDF.describe())
    return synthDF

#actual script for homework
#read in dataframe of iris dataset
df = readIrisData(sys.argv[1])

#split test train sets
xTrain, xTest, yTrain, yTest = train_test_split(irisWithoutClass(df).as_matrix(), 
                                                    df['class'].as_matrix(), 
                                                    test_size = 0.2, random_state=2)
#get mean and cov of iris data to generate with
mean = irisWithoutClass(df).mean()
cov = irisWithoutClass(df).cov()

#generate data
synthdf = generateSyntheticData(mean, cov)

#problem 4.a
classifyWithNeuralNetwork(xTrain, xTest, yTrain, yTest, synthdf)

#problem 4.b
classifyWithSVM(xTrain, xTest, yTrain, yTest, synthdf)



