# -*- coding: utf-8 -*-
import pandas
from scipy.io import arff
import matplotlib.pyplot as plot
import numpy
import sys
import math

# from assignment instructions: 1. Implement an algorithm to read in the Iris dataset
def readIrisData(filePath) -> pandas.DataFrame:
    #load data from arff file
    data = arff.loadarff(filePath)
    
    #put data into a dataframe for easier user and manipulation
    dataFrame = pandas.DataFrame(data[0])
    
    #strip whitespace and unwanted chars from class values
    dataFrame['class'] = dataFrame['class'].str.strip()
    dataFrame['class'] = dataFrame['class'].map(lambda x: str(x)[:-1])
    dataFrame['class'] = dataFrame['class'].map(lambda x: str(x)[2:])
    
    #load data to output file
    dataFrame.to_csv(sys.argv[2])
    
    #return dataframe for later use
    return dataFrame
    
# from assignment instructions: 2. Implement an algorithm to visually see two 
# sets of features and the class they belong to.
def visualizeFeature(dataFrame, feature, inputNumber):
    #set figure for pyplot
    figure = plot.figure()
    
    #set histogram bins to be 0-8 to tighten frame
    bins = numpy.linspace(0, 8, 50)
    
    #split dataframe by class
    setosa = dataFrame[dataFrame['class']=='Iris-setosa']
    versicolor = dataFrame[dataFrame['class']=='Iris-versicolor']
    virginica = dataFrame[dataFrame['class']=='Iris-virginica']
    
    #plot each subset onto a histogram
    plot.hist(setosa[feature], bins, label='setosa')
    plot.hist(versicolor[feature], bins, label='versicolor')
    plot.hist(virginica[feature], bins, label='virginica')
    
    #add labeling
    plot.title(feature + ' by Class')
    plot.xlabel(feature)
    plot.ylabel('Count')
    plot.legend()
    plot.show()

    #save plot to file
    figure.savefig(sys.argv[inputNumber], bbox_inches='tight')
    return;

# from assignment instructions: 3. Implement your sorting algorithm from Homework 1 Problem 5. 
# sorting algorithm chosen was a merge sort with buckets of size lg(n) to do insertion sort on
def mergeSort(df):
    #want buckets of lg( whole dataFrame size) to do insertionSort on
    #hardcoded 150 in for whole dataframe size, will get buckets of 7
    if (len(df) > math.log2(150)):
        #split dataframe in two and do merge sort on each half
        middle = int((len(df))/2)
        firstHalf = df[:middle]
        firstHalf = firstHalf.reset_index(drop=True)
        secondHalf = df[middle:]
        secondHalf = secondHalf.reset_index(drop=True)
        mergeSort(firstHalf)
        mergeSort(secondHalf)
        
        #move values from temporary half dataframes back into df
        firstCounter = secondCounter = fullCounter = 0
        while firstCounter < len(firstHalf) and secondCounter < len(secondHalf):
            #if value in first half less than value in second half, add first half back into df
            if firstHalf.loc[firstCounter, 'petallength'] < secondHalf.loc[secondCounter, 'petallength']:
                df.iloc[fullCounter] = firstHalf.iloc[firstCounter]
                #move to next value in first half
                firstCounter = firstCounter +1
            else: #add second half value back into df
                df.iloc[fullCounter] = secondHalf.iloc[secondCounter]
                secondCounter = secondCounter +1
            #move to next position in full df
            fullCounter = fullCounter + 1
            
        #move remaining values from first half into dataframe
        while firstCounter < len(firstHalf): 
            df.loc[fullCounter] = firstHalf.loc[firstCounter] 
            firstCounter+=1
            fullCounter+=1
            
        #move remaining values from second half into dataframe
        while secondCounter < len(secondHalf): 
            df.loc[fullCounter] = secondHalf.loc[secondCounter] 
            secondCounter+=1
            fullCounter+=1
    else:
        #when buckets are small enough, do insertion sort on dataframe
        for index in range(1, len(df)):
            #set the key to the current item
            key = df.iloc[index]
            #compare with previous value
            compareIndex = (index - 1)
            #while compareindex is within the array bounds (greater than -1)
            #and petal legnth of compareIndex is greater than key's petal length
            while (compareIndex >= 0 and df.loc[compareIndex, 'petallength'] > key['petallength']):
                #move row at compareIndex down one
                df.iloc[(compareIndex + 1)]=df.iloc[compareIndex]
                #set compareIndex one less (higher)
                compareIndex = compareIndex - 1
                #set row at compareIndex +1 to key value (key petallength is greater than all pervious values)
                df.iloc[(compareIndex + 1)] = key
    return
        

# actual scriptting to run full instructions
# perform (1) from PA1 assignment
df = readIrisData(sys.argv[1])

# perform (2) from PA1 assignment
visualizeFeature(df, 'petallength', 3)
visualizeFeature(df, 'petalwidth', 4)
visualizeFeature(df, 'sepallength', 5)
visualizeFeature(df, 'sepalwidth', 6)

# perform (3) from PA1 assignment
mergeSort(df)
df.to_csv(sys.argv[7])

#signal to console that script is complete
print("completed programming assignment 1 run")