import csv
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as linalg
import math
from split_train_test import split_data

def import_data(ifname):
    """
    Imports data with file-name/-path ifname as a numpy array.
    """
    with open(ifname, 'r') as ifile:
        datareader = csv.reader(ifile, delimiter=';')
        # we want to avoid importing the header line.
        # instead we'll print it to screen
        field_names = next(datareader)
        # create an empty list to store each row of data
        data = []
        for row in datareader:
            # for each row of data 
            # convert each element (from string) to float type
            row_of_floats = list(map(float,row))
            # now store in our data list
            data.append(row_of_floats)
        # convert the data (list object) into a numpy array.
        data_as_array = np.array(data)
        # return this array to caller
        return data_as_array, field_names
    
def predictive_distribution(designmtx, beta, mN, SN):
    """
    Calculates the predictive distribution a linear model. This amounts to a
    mean and variance for each input point.

    parameters
    ----------
    designmtx - 2d (N x K) array of inputs (data-matrix or design-matrix) where
        N is the number of data-points and each row is that point's
        representation
    beta - the known noise precision
    mN - posterior mean of the weights (vector) 1d-array (or array-like)
        of length K
    SN - the posterior covariance matrix for the weights 2d (K x K)-array 

    returns
    -------
    ys - a vector of mean predictions, one for each input datapoint
    sigma2Ns - a vector of variances, one for each input data-point 
    """
    N, K = designmtx.shape
    Phi = np.matrix(designmtx)
    mN = np.matrix(mN).reshape((K,1))
    SN = np.matrix(SN)
    ys = Phi*mN
    # create an array of the right size with the uniform term
    sigma2Ns = np.ones(N)/beta
    for n in range(N):
        # now calculate and add in the data dependent term
        phi_n = Phi[n,:].transpose()
        sigma2Ns[n] += phi_n.transpose()*SN*phi_n
    return np.array(ys).flatten(), np.array(sigma2Ns)    
    
def construct_rbf_feature_mapping(centres, scale):
    """
    parameters
    ----------
    centres - a DxM matrix (numpy array) where D is the dimension of the space
        and each row is the central position of an rbf basis function.
        For D=1 can pass an M-vector (numpy array).
    scale - a float determining the width of the distribution. Equivalent role
        to the standard deviation in the Gaussian distribution.

    returns
    -------
    feature_mapping - a function which takes an NxD data matrix and returns
        the design matrix (NxM matrix of features)
    """
    #  to enable python's broadcasting capability we need the centres
    # array as a 1xDxM array
    if len(centres.shape) == 1:
        centres = centres.reshape((1,1,centres.size))
    else:
        centres = centres.reshape((1,centres.shape[1],centres.shape[0]))
    # the denominator
    denom = 2*scale**2
    # now create a function based on these basis functions
    def feature_mapping(datamtx):
        #  to enable python's broadcasting capability we need the datamtx array
        # as a NxDx1 array
        if len(datamtx.shape) == 1:
            # if the datamtx is just an array of scalars, turn this into
            # a Nx1x1 array
            datamtx = datamtx.reshape((datamtx.size,1,1))
        else:
            # if datamtx is NxD array, then we reshape matrix as a
            # NxDx1 array
            datamtx = datamtx.reshape((datamtx.shape[0], datamtx.shape[1], 1))
        return np.exp(-np.sum((datamtx - centres)**2,1)/denom)
    # return the created function
    return feature_mapping


def calculate_weights_posterior(designmtx, targets, beta, m0, S0):
    """
    Calculates the posterior distribution (multivariate gaussian) for weights
    in a linear model.

    parameters
    ----------
    designmtx - 2d (N x K) array of inputs (data-matrix or design-matrix) where
        N is the number of data-points and each row is that point's
        representation
    targets - 1d (N)-array of target values
    beta - the known noise precision
    m0 - prior mean (vector) 1d-array (or array-like) of length K
    S0 - the prior covariance matrix 2d-array

    returns
    -------
    mN - the posterior mean (vector)
    SN - the posterior covariance matrix 
    """
    N, K = designmtx.shape
    Phi = np.matrix(designmtx)
    t = np.matrix(targets).reshape((N,1))
    m0 = np.matrix(m0).reshape((K,1))
    S0_inv = np.matrix(np.linalg.inv(S0))
    SN = np.linalg.inv(S0_inv + beta*Phi.transpose()*Phi)
    mN = SN*(S0_inv*m0 + beta*Phi.transpose()*t)
    return np.array(mN).flatten(), np.array(SN)

def construct_feature_mapping_approx(feature_mapping, weights):
    """
    This function creates and returns a prediction function based on a
    feature mapping and some weights.

    The returned prediction function takes a set of input values and returns
    the predicted output for each.
    """
    # here is a function that is created on the fly from the input feature
    # mapping and weights
    def prediction_function(xs):
        designmtx = np.matrix(feature_mapping(xs))
        return linear_model_predict(designmtx, weights)
    # we return the function reference (handle) itself. This can be used like
    # any other function
    return prediction_function

def linear_model_predict(designmtx, weights):
    ys = np.matrix(designmtx)*np.matrix(weights).reshape((len(weights),1))
    return np.array(ys).flatten()


def plot_data_and_approximation(
        ys, inputs, targets, linewidth=3, xlim=None,
        **kwargs):
    """
    Plot a function, some associated regression data and an approximation
    in a given range

    parameters
    ----------
    predict_func - the approximating function
    inputs - the input data
    targets - the targets

    <for optional arguments see plot_function_and_data>

    returns
    -------
    fig - the figure object for the plot
    ax - the axes object for the plot
    lines - a list of the line objects on the plot
    """
    if xlim is None:
        xlim = (0,1)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    #ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(0.5, 8.5) 
    
    line, = ax.plot(inputs, targets, 'bo')
    line, = ax.plot(inputs, ys, 'r-', linewidth=linewidth)
    return fig, ax, [line]
           

def evaluate_linear_approx(inputs, targets, test_fraction):
    # the linear performance
    train_error, test_error = simple_evaluation_linear_model(
        inputs, targets, test_fraction=test_fraction)
    print("Linear Regression:")
    print("\t(train_error, test_error) = %r" % ((train_error, test_error),))
    return train_error, test_error
    
def simple_evaluation_linear_model(
        inputs, targets, test_fraction=None, reg_param=None):
    """
    Will split inputs and targets into train and test parts, then fit a linear
    model to the training part, and test on the both parts.

    Inputs can be a data matrix (or design matrix), targets should
    be real valued.

    parameters
    ----------
    inputs - the input design matrix (any feature mapping should already be
        applied)
    targets - the targets as a vector
    reg_param (optional) - the regularisation strength. If provided, then
        regularised least squares fitting is uses with this regularisation
        strength. Otherwise, (non-regularised) least squares is used.

    returns
    -------
    train_error - the training error for the approximation
    test_error - the test error for the approximation
    """
    # get the train test split
    train_part, test_part = train_and_test_split(
        inputs.shape[0], test_fraction=test_fraction)
    # break the data into train and test parts
    train_inputs, train_targets, test_inputs, test_targets = \
        train_and_test_partition(inputs, targets, train_part, test_part)
    # now train and evaluate the error on both sets
    train_error, test_error = train_and_test(
        train_inputs, train_targets, test_inputs, test_targets,
        reg_param=reg_param)
    return train_error, test_error
    
def train_and_test(
        train_inputs, train_targets, test_inputs, test_targets, reg_param=None):
    """
    Will fit a linear model with either least squares, or regularised least 
    squares to the training data, then evaluate on both test and training data

    parameters
    ----------
    train_inputs - the input design matrix for training
    train_targets - the training targets as a vector
    test_inputs - the input design matrix for testing
    test_targets - the test targets as a vector
    reg_param (optional) - the regularisation strength. If provided, then
        regularised maximum likelihood fitting is uses with this regularisation
        strength. Otherwise, (non-regularised) least squares is used.

    returns
    -------
    train_error - the training error for the approximation
    test_error - the test error for the approximation
    """
    # Find the optimal weights (depends on regularisation)
    if reg_param is None:
        # use simple least squares approach
        weights = ml_weights(
            train_inputs, train_targets)
    else:
        # use regularised least squares approach
        weights = regularised_ml_weights(
          train_inputs, train_targets,  reg_param)
    # predictions are linear functions of the inputs, we evaluate those here
    train_predicts = linear_model_predict(train_inputs, weights)
    test_predicts = linear_model_predict(test_inputs, weights)
    # evaluate the error between the predictions and true targets on both sets
    train_error = root_mean_squared_error(train_targets, train_predicts)
    test_error = root_mean_squared_error(test_targets, test_predicts)
    return train_error, test_error

def cv_evaluation_linear_model(
        inputs, targets, folds, reg_param=None):
    """
    Will split inputs and targets into train and test parts, then fit a linear
    model to the training part, and test on the both parts.

    Inputs can be a data matrix (or design matrix), targets should
    be real valued.

    parameters
    ----------
    inputs - the input design matrix (any feature mapping should already be
        applied)
    targets - the targets as a vector
    num_folds - the number of folds
    reg_param (optional) - the regularisation strength. If provided, then
        regularised least squares fitting is uses with this regularisation
        strength. Otherwise, (non-regularised) least squares is used.

    returns
    -------
    train_errors - the training errors for the approximation
    test_errors - the test errors for the approximation
    """
    # get the number of datapoints
    N = inputs.shape[0]
    # get th number of folds
    num_folds = len(folds)
    train_errors = np.empty(num_folds)
    test_errors = np.empty(num_folds)
    for f,fold in enumerate(folds):
        # f is the fold id, fold is the train-test split
        train_part, test_part = fold
        
        # break the data into train and test sets
        train_inputs, train_targets, test_inputs, test_targets = \
            train_and_test_partition(inputs, targets, train_part, test_part)
        # now train and evaluate the error on both sets
        train_error, test_error = train_and_test(
            train_inputs, train_targets, test_inputs, test_targets,
            reg_param=reg_param)
        #print("train_error = %r" % (train_error,))
        #print("test_error = %r" % (test_error,))
        train_errors[f] = train_error
        test_errors[f] = test_error
    return train_errors, test_errors


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

def ml_weights(inputmtx, targets):
    """
    This method returns the weights that give the best linear fit between
    the processed inputs and the targets.
    """
    Phi = np.matrix(inputmtx)
    targets = np.matrix(targets).reshape((len(targets),1))
    weights = linalg.inv(Phi.transpose()*Phi)*Phi.transpose()*targets
    return np.array(weights).flatten()

def train_and_test_split(N, test_fraction=None):
    """
    Randomly generates a train/test split for data of size N.

    parameters
    ----------
    N - the dataset size
    test_fraction - a fraction (between 0 and 1) specifying the proportion of
        the data to use as test data.
    """
    np.random.seed(7)
    if test_fraction is None:
        test_fraction = 0.5
    p = [test_fraction,(1-test_fraction)]
    train_part = np.random.choice([False,True],size=N, p=p)
    test_part = np.invert(train_part)
    return train_part, test_part

def train_and_test_partition(inputs, targets, train_part, test_part):
    """
    Splits a data matrix (or design matrix) and associated targets into train
    and test parts.

    parameters
    ----------
    inputs - a 2d numpy array whose rows are the datapoints, or can be a design
        matric, where rows are the feature vectors for data points.
    targets - a 1d numpy array whose elements are the targets.
    train_part - A list (or 1d array) of N booleans, where N is the number of
        data points. If the ith element is true then the ith data point will be
        added to the training data.
    test_part - (like train_part) but specifying the test points.

    returns
    -------     
    train_inputs - the training input matrix
    train_targets - the training targets
    test_inputs - the test input matrix
    test_targets - the test targtets
    """
    # get the indices of the train and test portion
    train_inputs = inputs[train_part,:]
    test_inputs = inputs[test_part,:]
    train_targets = targets[train_part]
    test_targets = targets[test_part]
    return train_inputs, train_targets, test_inputs, test_targets

def root_mean_squared_error(y_true, y_pred):
    """
    Evaluate how closely predicted values (y_pred) match the true values
    (y_true, also known as targets)

    Parameters
    ----------
    y_true - the true targets
    y_pred - the predicted targets

    Returns
    -------
    mse - The root mean squared error between true and predicted target
    """
    N = len(y_true)
    # be careful, square must be done element-wise (hence conversion
    # to np.array)
    mse = np.sum((np.array(y_true).flatten() - np.array(y_pred).flatten())**2)/N
    return np.sqrt(mse)

def plot_train_test_errors(
        control_var, experiment_sequence, train_errors, test_errors):
    """
    Plot the train and test errors for a sequence of experiments.

    parameters
    ----------
    control_var - the name of the control variable, e.g. degree (for polynomial)
        degree.
    experiment_sequence - a list of values applied to the control variable.
    """
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    train_line, = ax.plot(experiment_sequence, train_errors,'b-')
    test_line, = ax.plot(experiment_sequence, test_errors, 'r-')
    ax.set_xlabel(control_var)
    ax.set_ylabel("$E_{RMS}$")
    ax.legend([train_line, test_line], ["train", "test"])
    return fig, ax

def parameter_search_with_centres_rbf(inputs, targets, folds):
    """
    """
    N = inputs.shape[0]    
    
    centres_set = []
    centres_fraction = np.linspace(0.05, 0.4, 10)
    
    centres_variance = []
    for i, sample_fraction in enumerate(centres_fraction):
        p = (1-sample_fraction,sample_fraction)
        np.random.seed(42)
        centres = inputs[np.random.choice([False,True], size=N, p=p),:]
        centres_set.append(centres)
        centres=np.array(centres)
        centres_variance.append(np.var(centres))
    
    centres_variance = np.array(centres_variance)
    print("centres_variance = %r"%(centres_variance,))
    centres_set = np.array(centres_set)
    
    scales = np.logspace(0,5, 15) # of the basis functions
    reg_params = np.logspace(-12,-1, 15) # choices of regularisation strength
    reg_params = np.insert(reg_params, 0, 0)
    # create empty 2d arrays to store the train and test errors
    train_errors_means = np.empty((centres_set.size, scales.size,reg_params.size))
    test_errors_means = np.empty((centres_set.size, scales.size,reg_params.size))
    train_stdev_errors = np.empty((centres_set.size, scales.size,reg_params.size))
    test_stdev_errors = np.empty((centres_set.size, scales.size,reg_params.size))
    
    num_folds = len(folds)
    
    #iterate over centres
    for i,centres in enumerate(centres_set): 
        # iterate over the scales
        for j,scale in enumerate(scales):
            # i is the index, scale is the corresponding scale
            # we must recreate the feature mapping each time for different scales
            feature_mapping = construct_rbf_feature_mapping(centres,scale)
            designmtx = feature_mapping(inputs)
            # partition the design matrix and targets into train and test   
            
            # iteratre over the regularisation parameters
            for k, reg_param in enumerate(reg_params):
                # train and test the data
                  
                # k is the index of reg_param, reg_param is the regularisation parameter
                # cross validate with this regularisation parameter
                train_errors, test_errors = cv_evaluation_linear_model(
                designmtx, targets, folds, reg_param=reg_param)
                
                # we're interested in the average (mean) training and testing errors
                train_mean_error = np.mean(train_errors)
                test_mean_error = np.mean(test_errors)
                train_stdev_error = np.std(train_errors)
                test_stdev_error = np.std(test_errors)
                
                # store the train and test errors in our 2d arrays
                train_errors_means[i,j,k] = train_mean_error
                test_errors_means[i,j,k] = test_mean_error
        
                # store the results
                train_stdev_errors[i,j,k] = train_stdev_error
                test_stdev_errors[i,j,k] = test_stdev_error
    
    # we have a 2d array of train and test errors, we want to know the (i,j,k)
    # index of the best value
    best_i = math.floor(np.argmin(test_errors_means)/(test_errors_means.shape[1]*test_errors_means.shape[2]))
    best_j = math.floor(np.argmin(test_errors_means[best_i, :, :])/(test_errors_means[best_i, :, :]).shape[1])
    best_k = np.argmin(test_errors_means[best_i, best_j,:])
    
    no_reg_best_centres_i = math.floor(np.argmin(test_errors_means[:,:,0])/(test_errors_means[:,:,0].shape[1]))
    no_reg_best_scales_j = np.argmin(test_errors_means[no_reg_best_centres_i,:,0])
    print("Best joint choice of parameters:")
    print(
        "\tscale %.2g ; lambda = %.2g ; centres proportion %.2g" % (scales[best_j],reg_params[best_k], centres_fraction[best_i]))
    print("best number of features = %r"%(len(centres_set[best_i])))
    
    print("with no regularisation: best centres = %.2g"%(centres_fraction[no_reg_best_centres_i],))
    print("with no regularisation: best scale = %.2g"%(scales[no_reg_best_scales_j],))
   
    lower_train = train_errors_means - train_stdev_errors/np.sqrt(num_folds)
    upper_train = train_errors_means + train_stdev_errors/np.sqrt(num_folds)
    # test error bars
    lower_test = test_errors_means - test_stdev_errors/np.sqrt(num_folds)
    upper_test = test_errors_means + test_stdev_errors/np.sqrt(num_folds)
    
    # now we can plot the error for different scales using the best
    # regulariation and centres choice
    fig , ax = plot_train_test_errors(
        "scale", scales, train_errors_means[best_i,:,best_k], test_errors_means[best_i,:,best_k])
    ax.set_xscale('log')
    ax.fill_between(scales, lower_train[best_i,:,best_k], upper_train[best_i,:,best_k], alpha=0.2, color='b')
    ax.fill_between(scales, lower_test[best_i,:,best_k], upper_test[best_i,:,best_k], alpha=0.2, color='r')  
    ax.set_ylim([0.5,0.85])
    fig.suptitle('Errors for different scale choices')
    plt.axvline(x=scales[best_j])
   
    # ...and the error for different regularisation choices given the best
    # scale choice 
    fig , ax = plot_train_test_errors(
        "$\lambda$", reg_params, train_errors_means[best_i,best_j,:], test_errors_means[best_i,best_j,:])
    ax.fill_between(reg_params, lower_train[best_i,best_j,:], upper_train[best_i,best_j,:], alpha=0.2, color='b')
    ax.fill_between(reg_params, lower_test[best_i,best_j,:], upper_test[best_i,best_j,:], alpha=0.2, color='r')
    ax.set_xscale('log')
    ax.set_ylim([0.5,0.85])
    fig.suptitle('Error for different regularisation choices')
    plt.axvline(x=reg_params[best_k])
    
    #Plot error for varying number of features given best regularisation and scales choice
    fig , ax = plot_train_test_errors(
        "p", centres_fraction, train_errors_means[:,best_j,best_k], test_errors_means[:,best_j,best_k])
    fig.suptitle('Errors over varying number of features')
    ax.fill_between(centres_fraction, lower_train[:,best_j,best_k], upper_train[:,best_j,best_k], alpha=0.2, color='b')
    ax.fill_between(centres_fraction, lower_test[:,best_j,best_k], upper_test[:,best_j,best_k], alpha=0.2, color='r')
    ax.set_ylim([0.5,0.85])
    return centres_fraction[best_i], scales[best_j], reg_params[best_k], centres_set[best_i], scales[no_reg_best_scales_j],centres_set[no_reg_best_centres_i]
    
def exploratory_plots(data, field_names=None):
    # the number of dimensions in the data
    dim = data.shape[1]
    # create an empty figure object
    fig = plt.figure()
    # create a grid of four axes
    plot_id = 1
    for i in range(dim):
        for j in range(dim):
            ax = fig.add_subplot(dim,dim,plot_id)
            # if it is a plot on the diagonal we histogram the data
            if i == j:
                ax.hist(data[:,i])
            # otherwise we scatter plot the data
            else:
                ax.plot(data[:,i],data[:,j], 'o', markersize=1)
            # we're only interested in the patterns in the data, so there is no
            # need for numeric values at this stage
            ax.set_xticks([])
            ax.set_yticks([])
            # if we have field names, then label the axes
            if not field_names is None:
                if i == (dim-1):
                    ax.set_xlabel(field_names[j])
                if j == 0:
                    ax.set_ylabel(field_names[i])
            # increment the plot_id
            plot_id += 1
    plt.tight_layout()  

def create_folds(N, num_folds):
    """
    Defines the cross-validation splits for N data-points into num_folds folds.
    Returns a list of folds, where each fold is a train-test split of the data.
    Achieves this by partitioning the data into num_folds (almost) equal
    subsets, where in the ith fold, the ith subset will be assigned to testing,
    with the remaining subsets assigned to training.

    parameters
    ----------
    N - the number of datapoints
    num_folds - the number of folds

    returns
    -------
    folds - a sequence of num_folds folds, each fold is a train and test array
        indicating (with a boolean array) whether a datapoint belongs to the
        training or testing part of the fold.
        Each fold is a (train_part, test_part) pair where:

        train_part - a boolean vector of length N, where if ith element is
            True if the ith data-point belongs to the training set, and False if
            otherwise.
        test_part - a boolean vector of length N, where if ith element is
            True if the ith data-point belongs to the testing set, and False if
            otherwise.
    """
    # if the number of datapoints is not divisible by folds then some parts
    # will be larger than others (by 1 data-point). min_part is the smallest
    # size of a part (uses integer division operator //)
    min_part = N//num_folds
    # rem is the number of parts that will be 1 larger
    rem = N % num_folds
    # create an empty array which will specify which part a datapoint belongs to 
    parts = np.empty(N, dtype=int)
    start = 0
    for part_id in range(num_folds):
        # calculate size of the part
        n_part = min_part
        if part_id < rem:
            n_part += 1
        # now assign the part id to a block of the parts array
        parts[start:start+n_part] = part_id*np.ones(n_part)
        start += n_part
    # now randomly reorder the parts array (so that each datapoint is assigned
    # a random part.
    np.random.shuffle(parts)
    # we now want to turn the parts array, into a sequence of train-test folds
    folds = []
    for f in range(num_folds):
        train = (parts != f)
        test = (parts == f)
        folds.append((train,test))
    return folds
    
def gaussian(x, mu, sigma2):
     return (np.sqrt(2*math.pi*sigma2)**-1)*math.e**-((x - mu)**2/(2*sigma2))
    
def joint_log_probability_error(targets, mean_approx, sigma2Ns):
    joint_log_error = 0
    for n in range(0, targets.shape[0]):
        joint_log_error += np.log(gaussian(targets[n], mean_approx[n], sigma2Ns[n]))
    joint_log_error = -joint_log_error
    return joint_log_error
    

def main(ifname):
    data, field_names = import_data(ifname)
          
    test_fraction = 0.1
    # Import data and split into train and test parts
    training_data, test_data, field_names = split_data(ifname, delimiter=';', has_header=True,seed=42, fraction=test_fraction)
    training_targets = training_data[:,11] #Quality of the wine
    test_targets = test_data[:,11]
    training_inputs = training_data[:, [0,1,2,3,4,5,6,7,8,9,10]]
    test_inputs = test_data[:, [0,1,2,3,4,5,6,7,8,9,10]]
     
    data1 = data[:, [0,1,2,3,11]]
    data2 = data[:, [4,5,6,7,11]]
    data3 = data[:, [8,9,10,11]]
    
    exploratory_plots(data1, ['f_acid','v_acid','c_acid', 'r_sug','q'])
    exploratory_plots(data2, ['cl','f_sul','t_sul', 'd','q'])
    exploratory_plots(data3, ['ph','sulp','alc','q'])
    
    targets = data[:,11] #Quality of the wine

    inputs = data[:, [0,1,2,3,4,5,6,7,8,9,10]]
    
    training_fixed_acidity_inputs = training_inputs[:,0]
    training_volatile_acidity_inputs = training_inputs[:,1]
    training_citric_acid_inputs = training_inputs[:,2]
    training_residual_sugar_inputs = training_inputs[:,3]
    training_chlorides_inputs = training_inputs[:,4]
    training_free_sulfur_dioxide_inputs = training_inputs[:,5]
    training_total_sulfur_dioxide_inputs = training_inputs[:,6]
    training_density_inputs = training_inputs[:,7]
    training_pH_inputs = training_inputs[:,8]
    training_sulphates_inputs = training_inputs[:,9]
    training_alcohol_inputs = training_inputs[:,10]
    
    test_fixed_acidity_inputs = test_inputs[:,0]
    test_volatile_acidity_inputs = test_inputs[:,1]
    test_citric_acid_inputs = test_inputs[:,2]
    test_residual_sugar_inputs = test_inputs[:,3]
    test_chlorides_inputs = test_inputs[:,4]
    test_free_sulfur_dioxide_inputs = test_inputs[:,5]
    test_total_sulfur_dioxide_inputs = test_inputs[:,6]
    test_density_inputs = test_inputs[:,7]
    test_pH_inputs = test_inputs[:,8]
    test_sulphates_inputs = test_inputs[:,9]
    test_alcohol_inputs = test_inputs[:,10]    
    
    # normalise inputs (meaning radial basis functions are more helpful)
    training_inputs[:,0] = (training_fixed_acidity_inputs - np.mean(training_fixed_acidity_inputs))/np.std(training_fixed_acidity_inputs)
    training_inputs[:,1] = (training_volatile_acidity_inputs - np.mean(training_volatile_acidity_inputs))/np.std(training_volatile_acidity_inputs)
    training_inputs[:,2] = (training_citric_acid_inputs - np.mean(training_citric_acid_inputs))/np.std(training_citric_acid_inputs)
    training_inputs[:,3] = (training_residual_sugar_inputs - np.mean(training_residual_sugar_inputs))/np.std(training_residual_sugar_inputs)
    training_inputs[:,4] = (training_chlorides_inputs - np.mean(training_chlorides_inputs))/np.std(training_chlorides_inputs)
    training_inputs[:,5] = (training_free_sulfur_dioxide_inputs - np.mean(training_free_sulfur_dioxide_inputs))/np.std(training_free_sulfur_dioxide_inputs)
    training_inputs[:,6] = (training_total_sulfur_dioxide_inputs - np.mean(training_total_sulfur_dioxide_inputs))/np.std(training_total_sulfur_dioxide_inputs)
    training_inputs[:,7] = (training_density_inputs - np.mean(training_density_inputs))/np.std(training_density_inputs)
    training_inputs[:,8] = (training_pH_inputs - np.mean(training_pH_inputs))/np.std(training_pH_inputs)
    training_inputs[:,9] = (training_sulphates_inputs - np.mean(training_sulphates_inputs))/np.std(training_sulphates_inputs)
    training_inputs[:,10] = (training_alcohol_inputs - np.mean(training_alcohol_inputs))/np.std(training_alcohol_inputs)
   
    test_inputs[:,0] = (test_fixed_acidity_inputs - np.mean(test_fixed_acidity_inputs))/np.std(test_fixed_acidity_inputs)
    test_inputs[:,1] = (test_volatile_acidity_inputs - np.mean(test_volatile_acidity_inputs))/np.std(test_volatile_acidity_inputs)
    test_inputs[:,2] = (test_citric_acid_inputs - np.mean(test_citric_acid_inputs))/np.std(test_citric_acid_inputs)
    test_inputs[:,3] = (test_residual_sugar_inputs - np.mean(test_residual_sugar_inputs))/np.std(test_residual_sugar_inputs)
    test_inputs[:,4] = (test_chlorides_inputs - np.mean(test_chlorides_inputs))/np.std(test_chlorides_inputs)
    test_inputs[:,5] = (test_free_sulfur_dioxide_inputs - np.mean(test_free_sulfur_dioxide_inputs))/np.std(test_free_sulfur_dioxide_inputs)
    test_inputs[:,6] = (test_total_sulfur_dioxide_inputs - np.mean(test_total_sulfur_dioxide_inputs))/np.std(test_total_sulfur_dioxide_inputs)
    test_inputs[:,7] = (test_density_inputs - np.mean(test_density_inputs))/np.std(test_density_inputs)
    test_inputs[:,8] = (test_pH_inputs - np.mean(test_pH_inputs))/np.std(test_pH_inputs)
    test_inputs[:,9] = (test_sulphates_inputs - np.mean(test_sulphates_inputs))/np.std(test_sulphates_inputs)
    test_inputs[:,10] = (test_alcohol_inputs - np.mean(test_alcohol_inputs))/np.std(test_alcohol_inputs)
   
    #### RBF model #####
    
    # get the cross-validation folds
    num_folds = 5
    N = training_data.shape[0]
    folds = create_folds(N, num_folds)
   
    # Find the paramaters for number of centres/features, scale and reg param
    # that best fits training data: 
    best_feature_proportion, best_scale, best_regparam, centres, best_scale_no_reg, best_centres_no_reg = \
        parameter_search_with_centres_rbf(training_inputs, training_targets, folds) 
    
    # Now evaluate error on test data using optimized parameter 
    # produced by training data:
    
    # with regularisation
    feature_mapping = construct_rbf_feature_mapping(centres, best_scale)
    designmtx = feature_mapping(training_inputs)
    weights = regularised_ml_weights(designmtx, training_targets,  best_regparam)
    
    test_designmtx = feature_mapping(test_inputs)
    test_predicts = linear_model_predict(test_designmtx, weights)
    
    # Evaluate the error between the predictions and true targets 
    test_error = root_mean_squared_error(test_targets, test_predicts)
    print("Final: rbf test error with regularisation= %r"%(test_error,))
    
    #Now find error without regularisation:
    feature_mapping = construct_rbf_feature_mapping(best_centres_no_reg, best_scale_no_reg)
    designmtx = feature_mapping(training_inputs)
    weights = ml_weights(designmtx, training_targets)
    
    test_designmtx = feature_mapping(test_inputs)
    test_predicts = linear_model_predict(test_designmtx, weights)
    
    # Evaluate the error between the predictions and true targets 
    test_error = root_mean_squared_error(test_targets, test_predicts)
    print("Final: rbf test error without regularisation= %r"%(test_error,))
    
    #### BAYESIAN analysis ####
    
    # the number of features is the width of the design matrix
    M = designmtx.shape[1]
    # Define a prior mean
    m0 = np.zeros(M)

    # Try different values of alpha for the prior covariance matrix
    alphas = np.logspace(-1,8, 10)  

    #Try different iterations on the noise precision
    betas = np.logspace(-1,2,10)

    bayesian_test_errors=np.empty((alphas.size, betas.size))
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
       
            # Covariance prior
            S0 = alpha * np.identity(M)
            # find the posterior over weights 
            mN, SN = calculate_weights_posterior(designmtx, training_targets, beta, m0, S0)
            # Now find the error
            ys, sigma2Ns = predictive_distribution(test_designmtx, beta, mN, SN)    
            log_error = joint_log_probability_error(test_targets, ys, sigma2Ns)
        
            bayesian_test_errors[i,j] = log_error
    
    # Index of variation with smallest test error 
    best_i = math.floor(np.argmin(bayesian_test_errors)/bayesian_test_errors.shape[1])
    best_j = np.argmin(bayesian_test_errors[best_i,:])
    
    print("best alpha = %r" % (alphas[best_i],))
    print("best beta = %r"%(betas[best_j],))
    print("1/np.var(training_targets) = %r"%(np.var(training_targets),))
    print("bayesian test log error with optimal alpha and beta = %r"%(np.min(bayesian_test_errors),))
   
    #Plot the error for different values of alpha and beta
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alphas, bayesian_test_errors[:,best_j], 'r-')
    ax.set_xlabel("alpha")
    ax.set_ylabel("$Error(alpha)$")
    fig.suptitle("Error while varying alpha using joint log probability ")
    ax.set_xscale('log')
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(betas, bayesian_test_errors[best_i,:], 'r-')
    ax.set_xscale('log')
    ax.set_xlabel("Beta")
    plt.axvline(x=1/np.var(training_targets))
    ax.set_ylabel("$Error(beta)$")
    fig.suptitle("Error while varying beta using joint log probability")
    
    # Now we have best alpha and beta, find the rms error of the model predictions
    # for the test data:
     
    # Covariance prior
    S0 = alphas[best_i] * np.identity(M)
    # find the posterior over weights 
    mN, SN = calculate_weights_posterior(designmtx, training_targets, betas[best_j], m0, S0)
    # Now find the error
    mean_approx = construct_feature_mapping_approx(feature_mapping, mN)
    test_predic = mean_approx(test_inputs)  
    b_test_error = root_mean_squared_error(test_targets, test_predic)
    print("RMS error for bayesian model with best choices of alpha and beta = %r"%(b_test_error,))
    
    plt.show()
        
        
if __name__ == '__main__':
    import sys

    main(sys.argv[1])
