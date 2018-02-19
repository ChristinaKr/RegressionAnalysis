import csv
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as linalg
import math

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
        # NOTE: I couldn't work out a neat way of doing this without a for-loop
        # NOTE: but if anyone can please share the answer.
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
        predict_func, inputs, targets, linewidth=3, xlim=None,
        **kwargs):
    """
    Plot a function, some associated regression data and an approximation
    in a given range

    parameters
    ----------
    predict_func - the approximating function
    inputs - the input data
    targets - the targets
    true_func - the true function
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
    xs = np.linspace(xlim[0], xlim[1], 101)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(0.5, 8.5) 
    
    line, = ax.plot(inputs, targets, 'bo')
    ys = predict_func(xs)
    line, = ax.plot(xs, ys, 'r-', linewidth=linewidth)
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
  
def evaluate_rbf_for_various_reg_params(
        inputs, targets, test_fraction, test_error_linear):
    """
    """

    # for rbf feature mappings
    # for the centres of the basis functions choose 10% of the data
    N = inputs.shape[0]
    centres = inputs[np.random.choice([False,True], size=N, p=[0.9,0.1]),:]
    print("centres.shape = %r" % (centres.shape,))
    scale = 1 # of the basis functions
    feature_mapping = construct_rbf_feature_mapping(centres,scale)
    designmtx = feature_mapping(inputs)
    train_part, test_part = train_and_test_split(N, test_fraction=test_fraction)
    train_designmtx, train_targets, test_designmtx, test_targets = \
        train_and_test_partition(
            designmtx, targets, train_part, test_part)
    # output the shapes of the train and test parts for debugging
    print("train_designmtx.shape = %r" % (train_designmtx.shape,))
    print("test_designmtx.shape = %r" % (test_designmtx.shape,))
    print("train_targets.shape = %r" % (train_targets.shape,))
    print("test_targets.shape = %r" % (test_targets.shape,))
    # the rbf feature mapping performance
    reg_params = np.logspace(-15,-4, 11)
    train_errors = []
    test_errors = []
    for reg_param in reg_params:
        print("Evaluating reg_para " + str(reg_param))
        train_error, test_error = simple_evaluation_linear_model(
            designmtx, targets, test_fraction=test_fraction, reg_param=reg_param)
        train_errors.append(train_error)
        test_errors.append(test_error)

    fig , ax = plot_train_test_errors(
        "$\lambda$", reg_params, train_errors, test_errors)
    # we also want to plot a straight line showing the linear performance
    xlim = ax.get_xlim()
    ax.plot(xlim, test_error_linear*np.ones(2), 'g:')
    ax.set_xscale('log')

def parameter_search_rbf(inputs, targets, test_fraction):
    """
    """
    N = inputs.shape[0]
    # run all experiments on the same train-test split of the data 
    train_part, test_part = train_and_test_split(N, test_fraction=test_fraction)
    # for the centres of the basis functions sample 10% of the data
    sample_fraction = 0.1
    p = (1-sample_fraction,sample_fraction)
    centres = inputs[np.random.choice([False,True], size=N, p=p),:]
    print("centres.shape = %r" % (centres.shape,))
    scales = np.logspace(0,5, 20) # of the basis functions
    reg_params = np.logspace(-15,-6, 20) # choices of regularisation strength
    # create empty 2d arrays to store the train and test errors
    train_errors = np.empty((scales.size,reg_params.size))
    test_errors = np.empty((scales.size,reg_params.size))
    # iterate over the scales
    for i,scale in enumerate(scales):
        # i is the index, scale is the corresponding scale
        # we must recreate the feature mapping each time for different scales
        feature_mapping = construct_rbf_feature_mapping(centres,scale)
        designmtx = feature_mapping(inputs)
        # partition the design matrix and targets into train and test
        train_designmtx, train_targets, test_designmtx, test_targets = \
            train_and_test_partition(
                designmtx, targets, train_part, test_part)
        # iteratre over the regularisation parameters
        for j, reg_param in enumerate(reg_params):
            # j is the index, reg_param is the corresponding regularisation
            # parameter
            # train and test the data
            train_error, test_error = train_and_test(
                train_designmtx, train_targets, test_designmtx, test_targets,
                reg_param=reg_param)
            # store the train and test errors in our 2d arrays
            train_errors[i,j] = train_error
            test_errors[i,j] = test_error
    
    # we have a 2d array of train and test errors, we want to know the (i,j)
    # index of the best value
    best_i = math.floor(np.argmin(test_errors)/test_errors.shape[0])
    
    #best_i = np.argmin(np.argmin(test_errors,axis=1))
    best_j = np.argmin(test_errors[i,:])
    print("Best joint choice of parameters:")
    print(
        "\tscale %.2g and lambda = %.2g" % (scales[best_i],reg_params[best_j]))
    # now we can plot the error for different scales using the best
    # regulariation choice
    fig , ax = plot_train_test_errors(
        "scale", scales, train_errors[:,best_j], test_errors[:,best_j])
    ax.set_xscale('log')
    fig.suptitle('Error for different scales using the best regulariation choice')
    plt.axvline(x=scales[best_i])
    # ...and the error for different regularisation choices given the best
    # scale choice 
    fig , ax = plot_train_test_errors(
        "$\lambda$", reg_params, train_errors[best_i,:], test_errors[best_i,:])
    ax.set_xscale('log')
    fig.suptitle('Error for different regularisation choices given the best scale choice')
    plt.axvline(x=reg_params[best_j])

    return scales[best_i], reg_params[best_j], centres 
  
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
  
def main(ifname):
    data, field_names = import_data(ifname)
    if type(data) == np.ndarray:
        print("Data array loaded: there are %d rows" % data.shape[0])
        print ("first row:", data[0,:])
    
    data1 = data[:, [0,1,2,3,11]]
    data2 = data[:, [4,5,6,7,11]]
    data3 = data[:, [8,9,10,11]]

    
    exploratory_plots(data1, ['facid','vacid','cacid', 'rsug','q'])
    exploratory_plots(data2, ['cl','fsul','tsul', 'd','q'])
    exploratory_plots(data3, ['ph','sulp','alc','q'])
    
    
    targets = data[:,11] #Quality of the wine
    #data_Matrix = np.delete(data,11,1)
    inputs = data[:, [0,1,2,3,4,5,6,7,8,9,10]]
    
    N = data.shape[0]
    
    fixed_acidity_inputs = inputs[:,0]
    volatile_acidity_inputs = inputs[:,1]
    citric_acid_inputs = inputs[:,2]
    residual_sugar_inputs = inputs[:,3]
    chlorides_inputs = inputs[:,4]
    free_sulfur_dioxide_inputs = inputs[:,5]
    total_sulfur_dioxide_inputs = inputs[:,6]
    density_inputs = inputs[:,7]
    pH_inputs = inputs[:,8]
    sulphates_inputs = inputs[:,9]
    alcohol_inputs = inputs[:,10]
    
    print("np.mean(fixed_acidity_inputs) = %r" % (np.mean(fixed_acidity_inputs),))
    print("np.std(volatile_acidity_inputs) = %r" % (np.std(volatile_acidity_inputs),))
    print("np.mean(citric_acid_inputs) = %r" % (np.mean(citric_acid_inputs),))
    print("np.std(residual_sugar_inputs) = %r" % (np.std(residual_sugar_inputs),))
    print("np.mean(chlorides_inputs) = %r" % (np.mean(chlorides_inputs),))
    print("np.std(free_sulfur_dioxide_inputs) = %r" % (np.std(free_sulfur_dioxide_inputs),))
    print("np.std(total_sulfur_dioxide_inputs) = %r" % (np.std(total_sulfur_dioxide_inputs),))
    print("np.std(density_inputs) = %r" % (np.std(density_inputs),))
    print("np.std(pH_inputs) = %r" % (np.std(pH_inputs),))
    print("np.std(sulphates_inputs) = %r" % (np.std(sulphates_inputs),))
    print("np.std(alcohol_inputs) = %r" % (np.std(alcohol_inputs),))
    
    # normalise inputs (meaning radial basis functions are more helpful)
    inputs[:,0] = (fixed_acidity_inputs - np.mean(fixed_acidity_inputs))/np.std(fixed_acidity_inputs)
    inputs[:,1] = (volatile_acidity_inputs - np.mean(volatile_acidity_inputs))/np.std(volatile_acidity_inputs)
    inputs[:,2] = (citric_acid_inputs - np.mean(citric_acid_inputs))/np.std(citric_acid_inputs)
    inputs[:,3] = (residual_sugar_inputs - np.mean(residual_sugar_inputs))/np.std(residual_sugar_inputs)
    inputs[:,4] = (chlorides_inputs - np.mean(chlorides_inputs))/np.std(chlorides_inputs)
    inputs[:,5] = (free_sulfur_dioxide_inputs - np.mean(free_sulfur_dioxide_inputs))/np.std(free_sulfur_dioxide_inputs)
    inputs[:,6] = (total_sulfur_dioxide_inputs - np.mean(total_sulfur_dioxide_inputs))/np.std(total_sulfur_dioxide_inputs)
    inputs[:,7] = (density_inputs - np.mean(density_inputs))/np.std(density_inputs)
    inputs[:,8] = (pH_inputs - np.mean(pH_inputs))/np.std(pH_inputs)
    inputs[:,9] = (sulphates_inputs - np.mean(sulphates_inputs))/np.std(sulphates_inputs)
    inputs[:,10] = (alcohol_inputs - np.mean(alcohol_inputs))/np.std(alcohol_inputs)
    
    test_fraction = 0.25
    
    train_error_linear, test_error_linear = evaluate_linear_approx(inputs, targets, test_fraction)
    evaluate_rbf_for_various_reg_params(inputs, targets, test_fraction, test_error_linear)
    #Find the best scales and regularisation parameter 
    best_scale, best_regparam, centres =  parameter_search_rbf(inputs, targets, test_fraction)
    
    plt.show()
    
    

    # create the feature mapping
    feature_mapping = construct_rbf_feature_mapping(centres, best_scale)  
    
    designmtx = feature_mapping(inputs)
    
    # the number of features is the width of this matrix
    M = designmtx.shape[1]
    # define a prior mean and covariance matrix
    m0 = np.zeros(M)
    alpha = 10
    S0 = alpha * np.identity(M)
    # define the noise precision of our data
    beta = 10000000
    # find the posterior over weights 
    mN, SN = calculate_weights_posterior(designmtx, targets, beta, m0, S0)
    # the posterior mean (also the MAP) gives the central prediction
    mean_approx = construct_feature_mapping_approx(feature_mapping, mN)#prediciton function
    
    # Input variables:
    # i =
    # 0 - fixed_acidity; 1-volatile_acidity; 2 - citric_acid; 
    # 3 - residual_sugar; 4 - chlorides; 5 -  free_sulfur;
    # 6 - total_sulfur_dioxide; 7 - density; 8 - pH;
    # 9 - sulphates; 10 - alcohol  
    i = 0
    one_variable_input = inputs[:,i]
    field_name = field_names[i]
    
    fig, ax, lines = plot_data_and_approximation(
        mean_approx, one_variable_input, targets, xlim=[np.min(one_variable_input)-2, np.max(one_variable_input)+2])
    ax.set_xlabel(field_name);
    ax.set_ylabel("Quality");
        
    
    #now for the predictive distribuiton
    new_inputs = []
    for i in range (0, M+1):
        new_variable = np.mean(inputs[:,i])
        new_inputs.append(new_variable)
    new_inputs = np.array(new_inputs)
    print(new_inputs)
    
    new_designmtx = feature_mapping(new_inputs)
    ys, sigma2Ns = predictive_distribution(new_designmtx, beta, mN, SN)
    print("sigma2Ns = %r" % (sigma2Ns,))
    fig, ax, lines = plot_data_and_approximation(mean_approx, one_variable_input, targets, xlim=[np.min(new_inputs),np.max(new_inputs)])
    ax.plot(new_inputs, ys, 'r', linewidth=3)
    ax.set_xlabel(field_name);
    ax.set_ylabel("Quality");
    lower = ys-np.sqrt(sigma2Ns)
    upper = ys+np.sqrt(sigma2Ns)
    print("lower.shape = %r" % (lower.shape,))
    print("upper.shape = %r" % (upper.shape,))
    ax.fill_between(new_inputs, lower, upper, alpha=0.2, color='r')
    plt.show()
        
        
if __name__ == '__main__':
    import sys
    # this allows you to pass the file name as the first argument when you call
    # your script from the command line
    # so to run this script use:
    # python old_faithful.py old_faithful.tsv
    try:
        main(sys.argv[1])
    except IndexError:
        print(
            "[ERROR] Please give the data-file location as the first argument.")

