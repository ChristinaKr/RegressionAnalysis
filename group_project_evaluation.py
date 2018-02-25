import csv
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt

from exploratory_data_analysis import exp_data_ana
from LinearModels import linear_models
from knn_regression import knn_regression
from rbf_and_bayesian_analysis import rbf_bayesian
from polynomial_regression import polynomial_regression

def main(
        ifname, delimiter=';', columns=[0,1,2,3,4,5,6,7,8,9,10,11], has_header=True,
        test_fraction=0.10):
        
    training_data_as_array, test_data_as_array, field_names = split_data(
            ifname, delimiter=';', has_header=has_header, columns=[0,1,2,3,4,5,6,7,8,9,10,11], seed=42)
    
    exp_data_ana(training_data_as_array,field_names)
    print "finished exploratory data analysis"
    
    linear_models(training_data_as_array,test_data_as_array, test_fraction)
    print "finished simple linear regression"
    
    knn_regression(training_data_as_array,test_data_as_array)
    print "finished kNN regression"
    
    rbf_bayesian(training_data_as_array, test_data_as_array, field_names)
    print "finished RBF and Bayesian model"
    
    Train_Data, Test_Data, field_names = split_data(ifname, delimiter, has_header, [0,1,2,3,4,5,6,7,8,9,10])
    Train_Targets, Test_Targets, field_names = split_data(ifname, delimiter, has_header, [11])
    
    polynomial_regression(Train_Data,Test_Data,Train_Targets,Test_Targets)
    print "finished polynomial"
    
    plt.show()

def split_data(ifname, delimiter=None, has_header=False, columns=None, seed=42):
    """
    Imports a tab/comma/semi-colon/... separated data file as an array of 
    floating point numbers. If the import file has a header then this should
    be specified, and the field names will be returned as the second argument.

    parameters
    ----------
    ifname -- filename/path of data file.
    delimiter -- delimiter of data values
    has_header -- does the data-file have a header line
    columns -- a list of integers specifying which columns of the file to import
        (counting from 0)

    returns
    -------
    data_as_array -- the data as a numpy.array object  
    field_names -- if file has header, then this is a list of strings of the
      the field names imported. Otherwise, it is a None object.
    """
    fraction = 0.15
    np.random.seed(seed)
    test_rows = np.unique(np.array(np.random.uniform(size = int(fraction*1599))*1599).astype(int))
    #print(test_rows)
    
    if delimiter is None:
        delimiter = '\t'
    with open(ifname, 'r') as ifile:
        datareader = csv.reader(ifile, delimiter=delimiter)
        # if the data has a header line we want to avoid trying to import it.
        if has_header:
            field_names = next(datareader)
        # create empty lists to store each row of data
        training_data = []
        test_data = []
        count = 0
        for row in datareader:
            # for each row of data only take the columns we are interested in
            if not columns is None:
                row = [row[c] for c in columns]
            # now store in our data lists
            if(count in test_rows):
                test_data.append(row)
            else:
                training_data.append(row)
            count+=1
    #print("There are %d training entries" % len(training_data))
    #print("There are %d test entries" % len(test_data))
    #print("Each row has %d elements" % len(training_data[0]))
    # convert the data (list object) into a numpy array.
    training_data_as_array = np.array(training_data).astype(float)
    test_data_as_array = np.array(test_data).astype(float)
    # return the two data sets to caller
    return training_data_as_array, test_data_as_array, field_names

if __name__ == '__main__':
    """
    To run this script on just synthetic data use:

        python regression_external_data.py

    You can pass the data-file name as the first argument when you call
    your script from the command line. E.g. use:

        python regression_external_data.py datafile.tsv

    If you pass a second argument it will be taken as the delimiter, e.g.
    for comma separated values:

        python regression_external_data.py comma_separated_data.csv ","

    for semi-colon separated values:

        python regression_external_data.py comma_separated_data.csv ";"

    If your data has more than 2 columns you must specify which columns
    you wish to plot as a comma separated pair of values, e.g.

        python regression_external_data.py comma_separated_data.csv ";" 8,9

    For the wine quality data you will need to specify which columns to pass.
    """
    import sys
    if len(sys.argv) == 1:
        main() # calls the main function with no arguments
    elif len(sys.argv) == 2:
        # assumes that the first argument is the input filename/path
        main(ifname=sys.argv[1])
    elif len(sys.argv) == 3:
        # assumes that the second argument is the data delimiter
        main(ifname=sys.argv[1], delimiter=sys.argv[2])
    elif len(sys.argv) == 4:
        # assumes that the third argument is the list of columns to import
        columns = list(map(int, sys.argv[3].split(","))) 
        main(ifname=sys.argv[1], delimiter=sys.argv[2], columns=columns)
