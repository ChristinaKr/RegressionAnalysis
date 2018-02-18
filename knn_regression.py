import csv
import numpy as np
import matplotlib.pyplot as plt

def import_data(ifname):
    """
    Imports data with file-name/-path ifname as a numpy array.
    """
    with open(ifname, 'r') as ifile:
        datareader = csv.reader(ifile, delimiter=';')
        # we want to avoid importing the header line.
        # instead we'll print it to screen
        header = next(datareader)
        #print("Importing data with fields:\n\t" + ",".join(header))
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
        return data_as_array


def construct_knn_approx(train_inputs, train_targets, k):  
    """
    For 1 dimensional training data, it produces a function:reals-> reals
    that outputs the mean training value in the k-Neighbourhood of any input.
    """
    # Create Euclidean distance.
    distance = lambda x,y: (x-y)**2
    print("Size of data: %d" % train_inputs.size)
    print("Shape of data inputs:", np.shape(train_inputs))
    train_inputs = np.resize(train_inputs, (1,train_inputs.size))
    print("Shape of data inputs after resizing:", np.shape(train_inputs))
    inputs = [7.5, 0.7, 0.1, 2, 0.03, 10, 30, 0.99, 3.46, 1.56, 9.3]
    def prediction_function(inputs):
        print("Hello")
        inputs = np.array([7.5, 0.7, 0.1, 2, 0.03, 10, 30, 0.99, 3.46, 1.56, 9.3])
        inputs = inputs.reshape((inputs.size,1))
        print("inputs:", inputs)
        print(np.shape(inputs))
        distances = distance(train_inputs, inputs)
        print(np.shape(distances))
        predicts = np.empty(inputs.size)
        print("predicts empty", predicts)
        for i, neighbourhood in enumerate(np.argpartition(distances, k)[:,:k]):
            # the neighbourhood is the indices of the closest inputs to xs[i]
            # the prediction is the mean of the targets for this neighbourhood
            predicts[i] = np.mean(train_targets[neighbourhood])
            print(predicts[i])
        print("Predicts: xx", predicts)
        return predicts
    # We return a handle to the locally defined function
    return prediction_function(inputs)
    
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
def main(ifname):
    data = import_data(ifname)
    if type(data) == np.ndarray:
        print("Data array loaded: there are %d rows" % data.shape[0])
        print ("first row:", data[0,:])
    targets = data[:,11]
    construct_knn_approx(data, targets, 3)
#    fig = plt.figure()
#    ax = fig.add_subplot(1,1,1)
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

