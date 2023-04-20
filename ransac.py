import numpy as np
import cv2
import random
import scipy.stats

# Matplotlib
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt


def calculate_N(p, s, e=.95):
    assert p > 0. and p < 1., "p must be between 0. and 1."
    assert e > 0. and e < 1., "p must be between 0. and 1."
    return np.log(1. - float(p)) / np.log(1. - (1. - e ** float(s)))


def RANSAC(data, gen_model, loss_model, consensus_model, s, N=1000, debug=False):
    """ Parameters:
            data: array(N,2)
            gen_model(data)-->model: function(array(N,2))-->any; returns the model given an array of data.
            loss_model(model,data)-->loss: function(any,array(N,2))-->[float, ...]; returns the loss of each data point based on the given model.
            concensus_model(loss)-->concensus_indexing: function(float)-->[int, ...]; returns a list of indexes of data points which fit the model.
            s: (int); random sample size.
            N: (int); the number of test iterations.
            debug: (bool); whether or not to print various debug information.

        Returns:
            best_model: any; Returns the model returned by gen_model which has the largest consensus set and smallest mean loss.
            best_consensus: [int, ...]; The coresponding list of concensus values. """

    best_model = None  # Stores the min error, Starts at infinity
    best_consensus = []  # Stores the best match list, starts at none
    best_loss = np.inf
    if debug: print("N: {}".format(N))
    for n in range(int(N)):
        M = random.sample(data, s)  # Get a random sample of data
        model = gen_model(M)  # Use the model on the data to get a translation or whatever else
        loss = loss_model(model, data)  # Get the error based on the model and all other points
        consensus_index = consensus_model(loss)  # Get the index all the data values that agree with the model
        consensus = np.array(data)[consensus_index]  # Get the values from the index
        if debug: print(consensus_index)
        mean_loss = np.mean(loss)
        if len(consensus) > len(best_consensus) or \
                (len(consensus) == len(
                    best_consensus) and mean_loss < best_loss):  # Reset best distance and best data if this distance is better than all previous
            best_model = model
            best_consensus = consensus
            best_loss = mean_loss
            if debug: print("Best Inliers: {}".format(len(consensus)))

    return best_model, best_consensus


def compute_linear(x, y, thresh, s=2, p=.98, debug=False):
    """ Parameters:
            x, y: list or array of data.
            thresh: The distance between the model and a data point for it to qualify for the consensus set.
            s: int; The sample size of the data for each model to fit.
            p: float (0,1); A probability for the fit of the final model being the best possible given parameters.
            debug: bool; Whether or not to print various outputs.

        Returns:
            (m, b): tuple (float, float); The model of the form y = m*x + b
            concensus_set: list; The indexes of consensus samples in the output model. """

    s = int(s)  # s must be an integer
    assert s > 2, "Sample size must be greater than 2 for linear fit."
    data = np.array([np.array(x) ** -1., y]).T
    if debug: print(data)

    # The model is a translation from one point to one other
    def translation_model(data):
        data = np.array(data)
        if debug: print("Gen Model In: {}".format(data))
        m, b, _, _, _ = scipy.stats.linregress(data[:, 0], data[:, 1])
        if debug: print("Gen Model Out: y={}*x+{}".format(m, b))
        return m, b

    def apply_model(m, b, x, y):
        """ Returns the distance from the line to the model. """
        if debug: print("Apply Model In: {}={}*{}+{}".format(y, m, x, b))
        predicted_y = m * x + b
        diff = np.abs(y - predicted_y)
        if debug: print("Apply Model Out: {}".format(diff))
        return diff

    # Loss is based on standard deviation
    def loss_model(model, data):
        if debug: print("Loss Model In: {}, {}".format(model, data))
        m, b = model
        out = apply_model(m, b, data[:, 0], data[:, 1])  # Get the distance after applying the model to keypoints
        if debug: print("Loss Model Out: {}".format(out))
        return out

    # Any loss less than the threshold is a member of the concensus set
    def concensus_model(loss):
        if debug: print("Consensus Model In: {}".format(loss))
        out = np.where(np.array(loss) < thresh)[0]
        if debug: print("Consensus Model Out: {}".format(out))
        return out

    # Get the number of iterations
    N = calculate_N(p, s)
    if debug: print("N: {}".format(N))

    model, consensus = RANSAC(data, translation_model, loss_model, concensus_model, s, N)
    m, b = model
    if debug: print("y={}*x+{}; consensus={}".format(m, b, len(consensus)))
    return model, list(consensus)


if __name__ == "__main__":
    x = [1237, 723, 1822, 1753, 1874, 2453]
    y = [361, 658, 1022, 1984, 2435, 2877]
    bars_length = len(x)
    model, _ = compute_linear(x=x, y=y, thresh=0.5, p=.8, s=3)
    # Unpack the model, get trend line
    m, b = model
    trend_x = np.linspace(np.min(bars_length), np.max(bars_length), 100)
    trend_y = m * trend_x + b

    # Plot the data
    plt.scatter(x, y)
    plt.plot(trend_x, trend_y)
    plt.savefig("bars.png")
