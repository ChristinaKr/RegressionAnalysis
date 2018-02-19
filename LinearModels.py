import csv
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import scipy as sp

from LinearModelsMethods import plot_train_test_errors
from LinearModelsMethods import simple_evaluation_linear_model
from LinearModelsMethods import exploratory_plots

#from gradientDescent import gradientDescent
#from gradientDescent import computeCost

def main(
        ifname, delimiter=None, columns=None, has_header=True,
        test_fraction=0.25):
    """
    To be called when the script is run. This function creates, fits and plots
    synthetic data, and then fits and plots imported data (if a filename is
    provided). In both cases, data is 2 dimensional real valued data and is fit
    with maximum likelihood 2d gaussian.

    parameters
    ----------
    ifname -- filename/path of data file. 
    delimiter -- delimiter of data values
    has_header -- does the data-file have a header line
    columns -- a list of integers specifying which columns of the file to import
        (counting from 0)    
    """
    # if no file name is provided then use synthetic data
    data, field_names = import_data(
            ifname, delimiter=delimiter, has_header=has_header, columns=columns)
    exploratory_plots(data, field_names)
    N = data.shape[0]
    inputs = data[:,[0,1,2,3,4,5,6,7,8,9,10]]
    targets = data[:,11]

    # let's inspect the data a little more
    fixed_acid = inputs[:,0]
    vol_acid = inputs[:,1]
    citric_acid = inputs[:,2]
    resid_sugar = inputs[:,3]
    chlor = inputs[:,4]
    f_SO2 = inputs[:,5]
    t_SO2 = inputs[:,6]
    density = inputs[:,7]
    pH = inputs[:,8]
    sulph = inputs[:,9]
    alco = inputs[:,10]

    print("np.mean(fixed_acid) = %r" % (np.mean(fixed_acid),))
    print("np.std(fixed_acid) = %r" % (np.std(fixed_acid),))
    print("np.mean(vol_acid) = %r" % (np.mean(vol_acid),))
    print("np.std(vol_acid) = %r" % (np.std(vol_acid),))
    print("np.mean(citric_acid) = %r" % (np.mean(citric_acid),))
    print("np.std(citric_acid) = %r" % (np.std(citric_acid),))

    # normalise inputs (meaning radial basis functions are more helpful)
    # normalising even for linear can be quicker computationally as scales and units are different for different attributes
    inputs[:,0] = (fixed_acid - np.mean(fixed_acid))/np.std(fixed_acid)
    inputs[:,1] = (vol_acid - np.mean(vol_acid))/np.std(vol_acid)
    inputs[:,2] = (citric_acid - np.mean(citric_acid))/np.std(citric_acid)
    inputs[:,3] = (resid_sugar - np.mean(resid_sugar))/np.std(resid_sugar)
    inputs[:,4] = (chlor - np.mean(chlor))/np.std(chlor)
    inputs[:,5] = (f_SO2 - np.mean(f_SO2))/np.std(f_SO2)
    inputs[:,6] = (t_SO2 - np.mean(t_SO2))/np.std(t_SO2)
    inputs[:,7] = (density - np.mean(density))/np.std(density)
    inputs[:,8] = (pH - np.mean(pH))/np.std(pH)
    inputs[:,9] = (sulph - np.mean(sulph))/np.std(sulph)
    inputs[:,10] = (alco - np.mean(alco))/np.std(alco)

    train_errors, test_errors, fig,ax = evaluate_linear_approx(
        inputs, targets, test_fraction)
    
    fig.suptitle('Train and Test Errors over different values of lambda (the regularisation parameter)')
    
    plt.show()
    
    # factor in the reg param based on the above plot
    # running a number of times, the train and test errors are minimised between 10**-7 and 10**-8
    train_error_reg, test_error_reg = simple_evaluation_linear_model(
        inputs, targets, test_fraction=test_fraction, reg_param=10**-7)
    print("Linear Regression with regularisation parameter of 10^-7:")
    print("\t(train_error, test_error) = %r" % ((train_error_reg, test_error_reg),))
    train_error_reg, test_error_reg = simple_evaluation_linear_model(
        inputs, targets, test_fraction=test_fraction, reg_param=10**-8)
    print("Linear Regression with regularisation parameter of 10^-8:")
    print("\t(train_error, test_error) = %r" % ((train_error_reg, test_error_reg),))
    
    # gradient descent

    # set X (training data) and y (target variable)
    X2 = inputs
    y2 = targets
    ones = np.ones((1599,1))
    X2 = np.hstack((ones,X2))
    #print (np.matrix(data[:,11])).T
    
    # convert to matrices and initialize theta
    X2 = np.matrix(X2)
    y2 = (np.matrix(data[:,11])).T
    m = np.shape(X2)
    theta = np.zeros((1,12))
    theta = np.matrix(theta)
    #theta = theta.T
    #print theta
    alpha = 0.1
    iters = 200

    # perform linear regression on the data set
    theta, cost2 = gradientDescent(X2, y2, theta, alpha, iters)
    
    # plot
    fig, ax = plt.subplots()
    ax.plot(np.arange(iters), cost2, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Cost vs Iterations')
    
    # predict    
    predict_wine(7.4,0.7,0,1.9,0.076,11,34,0.9978,3.51,0.56,9.4, theta)
    
    plt.show()

def evaluate_linear_approx(inputs, targets, test_fraction):
    # the linear performance
    #train_error, test_error = simple_evaluation_linear_model(inputs, targets, test_fraction=test_fraction)
    reg_params = np.logspace(-20,-4, 25)
    train_errors = []
    test_errors = []
    for reg_param in reg_params:
        print("Evaluating reg_para " + str(reg_param))
        train_error, test_error = simple_evaluation_linear_model(
            inputs, targets, test_fraction=test_fraction, reg_param=reg_param)
        train_errors.append(train_error)
        test_errors.append(test_error)
    
    fig , ax = plot_train_test_errors(
        "$\lambda$", reg_params, train_errors, test_errors)
    # we also want to plot a straight line showing the linear performance
    xlim = ax.get_xlim()
    #ax.plot(xlim, test_error_linear*np.ones(2), 'g:')
    ax.set_xscale('log')
    
    print("Linear Regression:")
    print("\t(train_error, test_error) = %r" % ((train_error, test_error),))
    return train_errors, test_errors, fig, ax

def ml_weights(inputmtx, targets):
    """
    This method returns the weights that give the best linear fit between
    the processed inputs and the targets.

    """
    Phi = np.matrix(inputmtx)
    targets = np.matrix(targets).reshape((len(targets),1))
    weights = linalg.inv(Phi.transpose()*Phi)*Phi.transpose()*targets
    return np.array(weights).flatten()

def regularised_ml_weights(
        inputmtx, targets, reg_param):
    """
    This method returns the weights that give the best linear fit between
    the processed inputs and the targets penalised by some regularisation term
    (reg_param)
    """
    Phi = np.matrix(inputmtx)
    targets = np.matrix(targets).reshape((len(targets),1))
    I = np.identity(Phi.shape[1])
    weights = linalg.inv(reg_param*I + Phi.transpose()*Phi)*Phi.transpose()*targets
    return np.array(weights).flatten()

def predict_wine(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,theta):
    x_i = sp.matrix([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11])
    x_n = (x_i - np.mean(x_i))/np.std(x_i) # normalise
    x = sp.hstack((sp.matrix([1]), x_n))
    quality = (x * theta.T)
    print ("Quality of wine with fixed acidity {}, volitile acidity {}, citric acid {}, residual sugar {}, chlorides {}, free sulfur dioxide {}, total sulfur dioxide {}, density {}, pH {}, sulphates {} and alcohol {} has a predicted quality of {}".format(x1, x2, x3,x4,x5,x6,x7,x8,x9,x10,x11,quality))

def gradientDescent(X, y, theta2, alpha, iters):
    
    #theta2 = theta2.T  
    #print "theta2: ", theta2
    
    temp = np.matrix(np.zeros(theta2.shape))
    parameters = int(theta2.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta2.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta2[0,j] - ((alpha / len(X)) * np.sum(term))

        theta2 = temp
        cost[i] = computeCost(X, y, theta2)

    return theta2, cost

def computeCost(X, y, theta2): 
    #theta2 = theta2.T   
    inner = np.power(((X * theta2.T) - y), 2)
    return np.sum(inner) / (2 * len(X))
    
def import_data(ifname, delimiter=None, has_header=False, columns=None):
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
    if delimiter is None:
        delimiter = '\t'
    with open(ifname, 'r') as ifile:
        datareader = csv.reader(ifile, delimiter=delimiter)
        # if the data has a header line we want to avoid trying to import it.
        # instead we'll print it to screen
        if has_header:
            field_names = next(datareader)
            print("Importing data with field_names:\n\t" + ",".join(field_names))
        else:
            # if there is no header then the field names is a dummy variable
            field_names = None
        # create an empty list to store each row of data
        data = []
        for row in datareader:
            print("row = %r" % (row,))
            # for each row of data only take the columns we are interested in
            if not columns is None:
                row = [row[c] for c in columns]
            # now store in our data list
            data.append(row)
        print("There are %d entries" % len(data))
        print("Each row has %d elements" % len(data[0]))
    # convert the data (list object) into a numpy array.
    data_as_array = np.array(data).astype(float)
    if not columns is None and not field_names is None:
        # thin the associated field names if needed
        field_names = [field_names[c] for c in columns]
    # return this array to caller (and field_names if provided)
    return data_as_array, field_names


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

