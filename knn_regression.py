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
    # Create Euclidean distance
    distance = lambda x,y: (x-y)**2
    train_inputs = train_inputs.transpose() # 12 x 1599
    # Take all the independent variables, exclude the independent one (quality)
    data = train_inputs[:11,:] # 11 x 1599
    # 11 x-values of invented data point of which quality should be predicted 
    Xx = np.array([4, 0.7, 0.1, 3, 0.07, 13, 40, 0.9964, 3, 0.56, 10])

    def prediction_function(Xx):
        # Reshape arrays of x-values into 11 x 1 column vector
        Xx = Xx.reshape((Xx.size,1))
        # Calculates distance between data points and the invented data point to predict (11 x 1599)
        distances = distance(data, Xx)
        
        # Sums up all distances per column (axis = 0), so that there's only 1 distance left per data point
        distanceSum = [np.sum(distances, axis = 0)]
        distanceSum = np.array(distanceSum)
        distanceSum = distanceSum.reshape((1, distanceSum.size)) # 1 x 1599
        
        # All quality values of the data points 
        ys = train_inputs[11,:].reshape(1, len(data[1,:]))
        print(np.shape(ys))
        
        # Append the quality values to the distance values (making a 2 x 1599 array "distanceQuality")
        distanceQuality = np.append(distanceSum, ys, axis = 0)
        
        # Sort array with regard to first row (distanceSum)
        distanceQuality = distanceQuality.transpose() # 1599 x 2
        distanceQSorted = distanceQuality[distanceQuality[:,0].argsort()]
        
        # Average over k-nearest neighbours
        predicts = np.round(np.mean(distanceQSorted[:k,1]),0)
        print(predicts)
        return predicts
    # We return a handle to the locally defined function
    return prediction_function(Xx)
    

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

