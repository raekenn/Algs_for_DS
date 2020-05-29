# -*- coding: utf-8 -*-
import pandas
from scipy import stats
import numpy as np
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import decomposition
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import mixture
from sklearn import discriminant_analysis
from sklearn import svm
from sklearn import neural_network
import sys

#method to clean iris data set
def cleanIrisData(df) -> pandas.DataFrame:   
    #split dataframe by class
    setosa = df[df['class']==1]
    setosa = setosa.reset_index(drop=True)
    versicolor = df[df['class']==2]
    versicolor = versicolor.reset_index(drop=True)
    virginica = df[df['class']==3]
    virginica = virginica.reset_index(drop=True)
    
    #get mean vector for each class
    setosaMean = setosa.mean()
    versicolorMean = versicolor.mean()
    virginicaMean = virginica.mean()
    
    #replace NaN in each class with mean for each feature of each class
    setosa = replaceNaNWithMean(setosa, setosaMean)
    versicolor = replaceNaNWithMean(versicolor, versicolorMean)
    virginica = replaceNaNWithMean(virginica, virginicaMean)
    
    #recombine classes into dataframe and return
    return pandas.concat([setosa, versicolor, virginica], ignore_index=True)
   
#method to replae all Nan in a dataframe with the means of each feature
def replaceNaNWithMean(df, meanVec) -> pandas.DataFrame:
    for feature in df:
        if feature != 'class':
            mean = meanVec.loc[feature]
        for index in range(0, len(df)):
            if pandas.isnull(df.loc[index, feature]):
                df.loc[index, feature] = mean
    return df

#method to transform iris data into normalized columns
def dataTransformation(df) -> pandas.DataFrame:
    #remove class column before transforming features
    classCol = df['class']
    df = irisWithoutClass(df)
    
    #normalize, go from centimeter space to 0-1 space
    x = df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pandas.DataFrame(x_scaled, columns = df.columns)
    
    #add class back into dataframe before returning
    df['class'] = classCol
    
    return df

#method to generate 2 new columns bases on the existing iris features
def generateFeatures(df) ->  pandas.DataFrame:
    #generate columns for the sepal and petal areas (length*width)
    df['sepal area'] = df.iloc[:,0] * df.iloc[:,1]
    df['petal area'] = df.iloc[:,2] * df.iloc[:,3]
    return df

#method to remove outliers from data set
def featurePreProcessing(df) -> pandas.DataFrame:
    #split dataframe by class
    setosa = df[df['class']==1]
    setosa = setosa.reset_index(drop=True)
    versicolor = df[df['class']==2]
    versicolor = versicolor.reset_index(drop=True)
    virginica = df[df['class']==3]
    virginica = virginica.reset_index(drop=True)
    
    #remove outliers on each class
    setosa = removeOutliers(irisWithoutClass(setosa), 'setosa')
    versicolor = removeOutliers(irisWithoutClass(versicolor), 'versicolor')
    virginica = removeOutliers(irisWithoutClass(virginica), 'virginica')
    
    #add class column back to class datsets
    setosa['class'] = 1
    versicolor['class'] = 2
    virginica['class'] =3
    
    #recombine classes into dataframe and return
    return pandas.concat([setosa, versicolor, virginica], ignore_index=True)

#method to perform outlier removal by class
def removeOutliers(df, className) -> pandas.DataFrame:
    #obtain z scores for each item in iris dataset
    #this will be a 7x150 matrix of scores
    z = np.abs(stats.zscore(df))
    
    #display items with z score greater than 3 present
    #this will be any item with a value outside of 3 standard deviations from
    #   the mean of that columns
    print("Outlier rows/columns for {}: ".format(className))
    print(df.iloc[np.where(z>3)])
    
    #remove these rows from dataframe and return dataframe
    new_iris = df[(z < 3).all(axis=1)]
    return new_iris.reset_index(drop=True)

#method to rank the correlation of each feature to the class
def rankFeatures(df):
    #translate dataframe to matrix without class
    df = df.drop(columns=['New Feature 1', 'New Feature 2'])
    X = irisWithoutClass(df)
    
    #create class array
    Y = df['class']
    
    #run chi squared on 6 "non new" features and return top 2
    selector = SelectKBest(score_func=chi2, k=2).fit(X, Y)
    names = X.columns.values[selector.get_support()] 
    scores = pandas.DataFrame(selector.scores_[selector.get_support()], index=names)

    print('Feature Ranking scores: ')
    print(scores)

#method to perform principal component analysis and return iris dataset reduced
#to 2 features
def performPCA(df) -> pandas.DataFrame:
    #remove class list and fit iris features to PCA model
    classes = df['class']
    X = irisWithoutClass(df)
    pca = decomposition.PCA(n_components=2)
    pca.fit(X)
    
    pca_iris = pandas.DataFrame(pca.transform(X))
    pca_iris['class'] = classes
    
    return pca_iris
    
def classifyByEachTechnique(df):
    #split int test train
    xTrain, xTest, yTrain, yTest = train_test_split(irisWithoutClass(df).as_matrix(), 
                                                    df['class'].as_matrix(), 
                                                    test_size = 0.2, random_state = 2)
    
    #expectation maximization
    classifyWithEM(xTrain, xTest, yTrain, yTest)
    print()
    
    #linear descriminant analysis
    classifyWithLinearDiscriminant(xTrain, xTest, yTrain, yTest)
    print()
    
    #neural network
    classifyWithNeuralNetwork(xTrain, xTest, yTrain, yTest)
    print()
    
    #svm
    classifyWithSVM(xTrain, xTest, yTrain, yTest)
    print()
    
def classifyWithEM(xTrain, xTest, yTrain, yTest):
    
    em = mixture.GaussianMixture(n_components=3, random_state=2)
    print(em.fit(xTrain, yTrain))
    print("Expectation Maximization Score: " + str(em.score(xTest, yTest)))
    print("Expectation Maximization Report: ")
    print(classification_report(yTest, em.predict(xTest), labels=[1, 2, 3]))
    
def classifyWithLinearDiscriminant(xTrain, xTest, yTrain, yTest):
    
    lda = discriminant_analysis.LinearDiscriminantAnalysis()
    print(lda.fit(xTrain, yTrain))
    print("Linear Discriminant Analysis Score: " + str(lda.score(xTest, yTest)))
    print("Linear Discriminant Analysis Report: ")
    print(classification_report(yTest, lda.predict(xTest), labels=[1, 2, 3]))
    
def classifyWithNeuralNetwork(xTrain, xTest, yTrain, yTest):
    
    nn = neural_network.MLPClassifier(random_state = 2, max_iter=10000)
    print(nn.fit(xTrain, yTrain))
    print("Neural Network Score: " + str(nn.score(xTest, yTest)))
    print("Neural Network Report: ")
    print(classification_report(yTest, nn.predict(xTest), labels=[1, 2, 3]))
    
def classifyWithSVM(xTrain, xTest, yTrain, yTest):
    
    svc = svm.SVC(random_state = 2)
    print(svc.fit(xTrain, yTrain))
    print("Support Vector Machine Score: " + str(svc.score(xTest, yTest)))
    print("Support Vector Machine Report: ")
    print(classification_report(yTest, svc.predict(xTest), labels=[1, 2, 3]))
    
def irisWithoutClass(df) -> pandas.DataFrame:
    return df.drop(columns=['class'])

#set printing to output file provided in arguments  
orig_stdout = sys.stdout
f = open(sys.argv[3], 'w')
sys.stdout = f

#read updated csv data to pandas dataframe
df = pandas.read_csv(sys.argv[1])
print("Input Dirty Data: ")
print(df.describe())

#spacing
print('\n' * 2)

df = cleanIrisData(df)
print("Cleaned Data: ")
print(df.describe())

#spacing
print('\n' * 2)

df = dataTransformation(df)
print("Data Normalized: ")
print(df.describe())

#spacing
print('\n' * 2)

df = generateFeatures(df)
print("Features Generated: ")
print(df.describe())

#spacing
print('\n' * 2)

df = featurePreProcessing(df)
print("Outliers Removed: ")
print(df.describe())

#spacing
print('\n' * 2)

rankFeatures(df)

#spacing
print('\n' * 2)

df = performPCA(df)
print("PCA Transformed Data: ")
print(df.describe())

#spacing
print('\n' * 2)

classifyByEachTechnique(df)

#spacing
print('\n' * 2)

#read "original" csv data to pandas dataframe
df = pandas.read_csv(sys.argv[2])
print("Input Original Data: ")
print(df.describe())

#classify based on original data
#spacing
print('\n' * 2)

classifyByEachTechnique(df)


#close output file stream
sys.stdout = orig_stdout
f.close()