# -*- coding: utf-8 -*-
import pandas
from scipy.io import arff
import matplotlib.pyplot as plot
import numpy
import sys
import math

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
    
    #return dataframe for later use
    return dataFrame

# return a dataframe of min values for each feature
def getMin(dataframe) -> pandas.DataFrame:
    minDF = pandas.DataFrame(columns = ('features', 'min'))
    minDF.set_index('features')
    for feature in dataframe:
        if feature != 'class':
            min = dataframe.loc[0, feature]
            for index in range (1, len(dataframe)):
                if dataframe.loc[index, feature] < min:
                    min = dataframe.loc[index, feature]
            minDF.loc[feature] = [feature, min]
    return minDF

# return a dataframe of max values for each feature
def getMax(dataframe) -> pandas.DataFrame:
    maxDF = pandas.DataFrame(columns = ('features', 'max'))
    maxDF.set_index('features')
    for feature in dataframe:
        if feature != 'class':
            max = dataframe.loc[0, feature]
            for index in range (1, len(dataframe)):
                if dataframe.loc[index, feature] > max:
                    max = dataframe.loc[index, feature]
            maxDF.loc[feature] = [feature, max]
    return maxDF

#return a dataframe of mean values for each feature
def getMean(dataframe) -> pandas.DataFrame:
    meanDF = pandas.DataFrame(columns = ('features', 'mean'))
    meanDF.set_index('features')
    for feature in dataframe:
        if feature != 'class':
            mean = dataframe.loc[0, feature]
            for index in range(1, len(dataframe)):
                mean += int(dataframe.loc[index, feature])
            mean = mean/len(dataframe)
            meanDF.loc[feature] = [feature, mean]
    return meanDF

#return dataframe of the mean with p values trimmed from sorted dataFrame
def getTrimmedMean(dataframe, p) -> pandas.DataFrame:
    trimmed = pandas.DataFrame(columns = ('features', 'mean'))
    trimmed.set_index('features')
    for feature in dataframe:
        if feature != 'class':
            sorted = dataframe.sort_values(by=[feature])
            sorted = sorted[p:-p]
            sorted = sorted.reset_index(drop=True)
            mean = sorted.loc[0, feature]
            for index in range(1, len(sorted)):
                mean += int(sorted.loc[index, feature])
            mean = mean/len(sorted)
            trimmed.loc[feature] = [feature, mean]
    return trimmed

#return dataframe of standard deviation for dataframe by feature
def getStanDev(dataframe, meanDF) -> pandas.DataFrame:
    stanDevDF = pandas.DataFrame(columns = ('features', 'standard deviation'))
    stanDevDF.set_index('features')
    for feature in dataframe:
        if feature != 'class':
            stanDev = (dataframe.loc[0, feature] - meanDF.loc[feature, 'mean'])**2
            for index in range(1, len(dataframe)):
                stanDev += (dataframe.loc[index, feature] - meanDF.loc[feature, 'mean'])**2
            stanDev = math.sqrt(stanDev/len(dataframe))
            stanDevDF.loc[feature] = [feature, stanDev]
    return stanDevDF

#return dataframe of skewness for dataframe by feature
def getSkew(dataframe, meanDF, stanDevDF) -> pandas.DataFrame:
    skewDF = pandas.DataFrame(columns = ('features', 'skewness'))
    skewDF.set_index('features')
    for feature in dataframe:
        if feature != 'class':
            skewness = (dataframe.loc[0, feature] - meanDF.loc[feature, 'mean'])**3
            for index in range(1, len(dataframe)):
                skewness += (dataframe.loc[index, feature] - meanDF.loc[feature, 'mean'])**3
            skewness = (skewness/(stanDevDF.loc[feature, 'standard deviation']**3))
            skewDF.loc[feature] = [feature, skewness]
    return skewDF

#return dataframe of kurtosis for dataframe by feature
def getKurtosis(dataframe, meanDF, stanDevDF) -> pandas.DataFrame:
    kurtosisDF = pandas.DataFrame(columns = ('features', 'kurtosis'))
    kurtosisDF.set_index('features')
    for feature in dataframe:
        if feature != 'class':
            kurtosis = (dataframe.loc[0, feature] - meanDF.loc[feature, 'mean'])**4
            for index in range(1, len(dataframe)):
                kurtosis += (dataframe.loc[index, feature] - meanDF.loc[feature, 'mean'])**4
            kurtosis = (kurtosis/(stanDevDF.loc[feature, 'standard deviation']**4))
            kurtosisDF.loc[feature] = [feature, kurtosis]
    return kurtosisDF 

#put together all stats into one dataframe
def problem1IndividualTable(df, p) -> pandas.DataFrame:
    minDF = getMin(df)
    maxDF = getMax(df)
    meanDF = getMean(df)
    trimmedDF = getTrimmedMean(df, p)
    stanDevDF = getStanDev(df, meanDF)
    skewDF = getSkew(df, meanDF, stanDevDF)
    kurtosisDF = getKurtosis(df, meanDF, stanDevDF)
    allStats = pandas.DataFrame(columns = ('features', 'min', 'max', 'mean', 'trimmed mean',
                                       'standard deviation', 'skewness', 'kurtosis'))
    allStats['features'] = minDF['features']
    allStats['min'] = minDF['min']
    allStats['max'] = maxDF['max']
    allStats['mean'] = meanDF['mean']
    allStats['trimmed mean'] = trimmedDF['mean']
    allStats['standard deviation'] = stanDevDF['standard deviation']
    allStats['skewness'] = skewDF['skewness']
    allStats['kurtosis'] = kurtosisDF['kurtosis']
    
    return allStats

#print analysis for all iris data and by class
def problem1(df):
    allStats = problem1IndividualTable(df, 3)
    print('statistics for full iris dataset')
    print(allStats)
    
    #split dataframe by class
    setosa = df[df['class']=='Iris-setosa']
    setosa = setosa.reset_index(drop=True)
    versicolor = df[df['class']=='Iris-versicolor']
    versicolor = versicolor.reset_index(drop=True)
    virginica = df[df['class']=='Iris-virginica']
    virginica = virginica.reset_index(drop=True)
    
    #perform problem 1 analysis on each class dataframe
    setosaStats = problem1IndividualTable(setosa, 1)
    print('statistics for setosa dataset')
    print(setosaStats)
    versicolorStats = problem1IndividualTable(versicolor, 1)
    print('statistics for versicolor dataset')
    print(versicolorStats)
    virginicaStats = problem1IndividualTable(virginica, 1)
    print('statistics for virginica dataset')
    print(virginicaStats)
    return
    
#create covariance matrix for given class dataframe
def covarianceMatrix(df):
    dfMatrix = df.drop(['class'], axis=1).as_matrix()
    
    nFraction = (1/len(df))
    #create ones array
    ones = numpy.ones(len(df))
    
    X = dfMatrix - numpy.multiply(numpy.dot(numpy.dot(ones, ones.T),dfMatrix),nFraction)
    
    Y = numpy.dot(X.T, X)
    
    covMatrix = numpy.multiply(Y, nFraction)
    
    return covMatrix

def problem2(df):
    #split dataframe by class
    setosa = df[df['class']=='Iris-setosa']
    setosa = setosa.reset_index(drop=True)
    versicolor = df[df['class']=='Iris-versicolor']
    versicolor = versicolor.reset_index(drop=True)
    virginica = df[df['class']=='Iris-virginica']
    virginica = virginica.reset_index(drop=True)
    
    #get mean vertex (take mean column of dataframe)
    meanDF = getMean(df)
    meanVertex = meanDF['mean']
    
    #create within scatter matrix from meanVertex
    
    
    
    return

#generate 100 additional observations for each class in iris dataset
def generateSyntheticData(meanDF, covDF) -> pandas.DataFrame:
    synth = numpy.random.multivariate_normal(meanDF['mean'], covDF, 100)
    synthDF = pandas.DataFrame({'sepallength':synth[:,0], 'sepalwidth':synth[:,1],
                                'petallength':synth[:,2], 'petalwidth':synth[:,3]})
    print(synthDF.describe())
    return synthDF

#make an indiviudal scatter plot for two given features on a given dataframe
def plotScatter(df, dfSynth, feature1, feature2, title):
    plot.scatter(df[feature1], df[feature2], color='r')
    plot.scatter(dfSynth[feature1], dfSynth[feature2], color='b')
    plot.xlabel(feature1)
    plot.ylabel(feature2)
    plot.title(title)
    plot.show()
    return

#print plot for problem 3
def problem3(df):
    # perform problem 3 analysis on each class
    setosa = df[df['class']=='Iris-setosa']
    setosa = setosa.reset_index(drop=True)
    setosaMean = getMean(setosa)
    setosaCov = covarianceMatrix(setosa)
    setosaSynth = generateSyntheticData(setosaMean, setosaCov)
    plotScatter(setosa, setosaSynth, 'petalwidth', 'petallength', 'setosa')
    plotScatter(setosa, setosaSynth, 'petalwidth', 'sepallength', 'setosa')
    plotScatter(setosa, setosaSynth, 'petalwidth', 'sepalwidth', 'setosa')
    plotScatter(setosa, setosaSynth, 'petallength', 'sepallength', 'setosa')
    plotScatter(setosa, setosaSynth, 'petallength', 'sepalwidth', 'setosa')
    plotScatter(setosa, setosaSynth, 'sepallength', 'sepalwidth', 'setosa')
    
    versicolor = df[df['class']=='Iris-versicolor']
    versicolor = versicolor.reset_index(drop=True)
    versicolorMean = getMean(versicolor)
    versicolorCov = covarianceMatrix(versicolor)
    versicolorSynth = generateSyntheticData(versicolorMean, versicolorCov)
    plotScatter(versicolor, versicolorSynth, 'petalwidth', 'petallength', 'versicolor')
    plotScatter(versicolor, versicolorSynth, 'petalwidth', 'sepallength', 'versicolor')
    plotScatter(versicolor, versicolorSynth, 'petalwidth', 'sepalwidth', 'versicolor')
    plotScatter(versicolor, versicolorSynth, 'petallength', 'sepallength', 'versicolor')
    plotScatter(versicolor, versicolorSynth, 'petallength', 'sepalwidth', 'versicolor')
    plotScatter(versicolor, versicolorSynth, 'sepallength', 'sepalwidth', 'versicolor')
    
    virginica = df[df['class']=='Iris-virginica']
    virginica = virginica.reset_index(drop=True)
    virginicaMean = getMean(virginica)
    virginicaCov = covarianceMatrix(virginica)
    virginicaSynth = generateSyntheticData(virginicaMean, virginicaCov)
    plotScatter(virginica, virginicaSynth, 'petalwidth', 'petallength', 'virginica')
    plotScatter(virginica, virginicaSynth, 'petalwidth', 'sepallength', 'virginica')
    plotScatter(virginica, virginicaSynth, 'petalwidth', 'sepalwidth', 'virginica')
    plotScatter(virginica, virginicaSynth, 'petallength', 'sepallength', 'virginica')
    plotScatter(virginica, virginicaSynth, 'petallength', 'sepalwidth', 'virginica')
    plotScatter(virginica, virginicaSynth, 'sepallength', 'sepalwidth', 'virginica')
    return
    
#actual script for homework
#read in dataframe of iris dataset
df = readIrisData(sys.argv[1])

#problem 1
problem1(df)

#problem 2
problem2(df)

#problem 3
problem3(df)