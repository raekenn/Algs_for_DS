The source code in this assignment was developed using Python in the Spyder IDE and can be run in iPython console or from command line.

To run this code from command line, enter: python pa1.py [iris.arff input] [iris dataset csv] [petallength visual pdf] [petalwidth visual pdf] [sepallength visual pdf] [sepalwidth visual pdf] [sort output csv]
[iris.arff input] - the iris dataset in a .arff file location
[iris dataset csv] - output file location to show .arff file can be read into a dataframe
[petallength visual pdf] - output file location of petal length histogram for each class in .pdf format
[petalwidth visual pdf] - output file location of petal width histogram for each class in .pdf format
[sepallength visual pdf] - output file location of sepal length histogram for each class in .pdf format
[sepalwidth visual pdf] - output file location of sepal width histogram for each class in .pdf format
[sort output csv] - output file location of sorted dataframe

outside libraries used:
- pandas: for dataframe creation and output files
- sys: to read command line arguments
- math: for logarithm function
- scipy: for arff reader
- matplotlib: for visualization
- numpy: to create bins for histograms
