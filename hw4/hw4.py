# -*- coding: utf-8 -*-
from scipy.io import arff
import pandas
from scipy.stats import norm
import numpy
import sys

# read data from source file
def readIrisData(filePath) -> pandas.DataFrame:
    #load data from arff file
    data = arff.loadarff(filePath)
    
    #put data into a dataframe for easier user and manipulation
    dataFrame = pandas.DataFrame(data[0])
    
    #remove not needed class feature
    dataFrame = dataFrame.drop(columns=['class'])
    
    #return dataframe for later use
    return dataFrame

# probability that a point came from a Guassian with given parameters
# note that the covariance must be diagonal for this to work
def prob(val, mu, sig, lam):
  p = lam
  for i in range(len(val)):
    p *= norm.pdf(val[i], mu[i], sig[i][i])
  return p


# assign every data point to its most likely cluster
def expectation(dataFrame, parameters, k):
   columns = list(dataFrame)
   columns.remove('label')
   for index in range(0, len(dataFrame)): 
        #initialize empty arrays
        p_cluster = numpy.empty(k)
        p = numpy.empty(len(columns))
        #get value for each feature (not including label feature at end)
        for i in range(0, len(columns)):
            p[i] = dataFrame.iloc[[index], [i]].values[0]
        #create a probability for each cluster in k
        for cluster in range(1, k):    
            p_cluster[cluster-1] = prob(p, list(parameters['mu{}'.format(cluster)]), list(parameters['sig{}'.format(cluster)]), parameters['lambda'][cluster-1] )
        dataFrame['label'][index] = numpy.argmax(p_cluster)+1
   return dataFrame


# update estimates of lambda, mu and sigma
def maximization(dataFrame, parameters, k):
  columns = list(dataFrame)
  columns.remove('label')
  for i in range(0, k-1):
      points_assigned_to_clusteri = dataFrame[dataFrame['label'] == (i+1)]
      percent_assigned_to_clusteri = len(points_assigned_to_clusteri) / float(len(dataFrame))
      parameters['lambda'][i] = percent_assigned_to_clusteri
      #fill in mu and sig for each feature
      for j in range(0, len(columns)):
          parameters['mu{}'.format(i+1)][j] = points_assigned_to_clusteri.iloc[:, j].mean()
          sig = numpy.zeros(len(columns))
          sig[j] = points_assigned_to_clusteri.iloc[:, j].std()
          parameters['sig{}'.format(i+1)][j] = sig
  return parameters

# get the distance between points
# used for determining if params have converged
def distance(old_params, new_params, k):
  dist = 0
  for param in range(1, k):
    for i in range(len(old_params)):
      dist += (old_params['mu{}'.format(param)][i] - new_params['mu{}'.format(param)][i]) ** 2
  return dist ** 0.5
  
def expectation_max(df, guess, k, **kwargs):
    # loop until parameters converge
    shift = sys.maxsize
    #default episolon is .01 (convergance means new value must be this close to this value)
    epsilon = kwargs.get('epsilon', 0.01)
    #default iterations is "infinity" for when we truly want to wait for convergance
    iterations = kwargs.get('iterations', float('inf'))
    iters = 0
    df_copy = df.copy()
    #seed numpy.random
    numpy.random.seed(1)
    # randomly assign points to their initial clusters
    df_copy['label'] = map(lambda x: x+1, numpy.random.choice(k, len(df)))
    params = pandas.DataFrame(guess)
    
    while shift > epsilon and iters < iterations:
        iters += 1
        # E-step
        updated_labels = expectation(df_copy.copy(), params, k)
        
        # M-step
        updated_parameters = maximization(updated_labels, params.copy(), k)
        
        # see if our estimates of mu have changed
        # could incorporate all params, or overall log-likelihood
        shift = distance(params, updated_parameters, k)
        
        # logging
        print("iteration {}, shift {}".format(iters, shift))
        print(updated_parameters)
        
        # update labels and params for the next iteration
        df_copy = updated_labels
        params = updated_parameters
        
#generate 300 observations for a given mean vector and covariance matrix
def generateSyntheticData(mean, cov) -> pandas.DataFrame:
    synth = numpy.random.multivariate_normal(mean, cov, 300)
    synthDF = pandas.DataFrame(synth)
    print(synthDF.describe())
    return synthDF
        
#actual run for homework
#problem 1
#given matrix of data for problem 1
p1df = pandas.DataFrame(data = [[1 ,2 ], [4, 2], [1, 3], [4, 3]])
p1df['label'] = 1

#given initial guesses from expectation maximazation slides
p1guess = { 'mu1': [1,4],
          'sig1': [ [1, 0], [0, 1]],
          'mu2': [2,2],
          'sig2': [ [1, 0], [0, 1] ],
          'lambda': [0.5, 0.5]
        }

#perform expectation max on p1 data
expectation_max(p1df, p1guess, 2, iterations=5)

#problem 2
#read in dataframe of iris dataset
p2df = readIrisData(sys.argv[1])
p2df['label'] = 1

#make initial guesses
p2guess = { 'mu1': [1,3, 5, 4],
          'sig1': [ [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
          'mu2': [4,4, 4, 4],
          'sig2': [ [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
          'mu3': [1,1,1,1],
          'sig3':[ [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
          'lambda': [0.2, 0.6, 0.2, 0]
        }

#perform expectation max
expectation_max(p2df, p2guess, 3)

#problem 3
#given mean vector and covariance matrix
mean = [4.5, 2.2, 3.3]
cov = [[0.5, 0.1, 0.05], 
       [0.1, 0.25, 0.1], 
       [0.05, 0.1, 0.4]]

#generate data
p3df = generateSyntheticData(mean, cov)
p3df['label'] = 1

p3guess = { 'mu1': [1,1, 1],
          'sig1': [ [1, 0, 0], [0, 1, 0], [0, 0, 1]],
          'mu2': [4,4, 4],
          'sig2': [ [1, 0, 0], [0, 1, 0], [0, 0, 1] ],
          'lambda': [0.4, 0.5, 0.1]
        }

expectation_max(p3df, p3guess, 2)