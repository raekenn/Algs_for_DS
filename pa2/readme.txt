Rachael Kenney
Algorithms for Data Science
Programming Assignment 2
Due 8/12/2019

The source code in this assignment was developed using Python in the Spyder IDE and can be run in iPython console or from command line.

To run this code from command line, enter: python pa2.py [iris_for_cleansing.csv input] [update_iris.csv input] [pa2_output.txt output]
[iris_for_cleansing.csv input] - the "dirty" iris dataset in a .csv file location
[update_iris.csv input] - the iris dataset with additional features in a .csv file location
[pa2_output.txt output] - the text file location you want the output to be sent to

outside libraries used:
- pandas: for dataframe structure
- sys: to read command line arguments
- sklearn.preprocessing: for normalization
- scipy.stats: obtain z score of data for outlier removal
- numpy: for array/matrix manipulation
- sklearn.decomposition: for PCA
- sklearn.feature_selection: for feature ranking
- sklearn.model_selection: for test train split for data modeling
- sklearn.mixture: guassian mixture for expectation maximization
- sklearn.discriminant_analysis: for linear discriminant model
- sklearn.svm: for support vector machine model
- sklearn.neural_network: for neural network model
- sklearn.metrics: for classification report to review model performance
