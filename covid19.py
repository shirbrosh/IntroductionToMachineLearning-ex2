import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linear_model import fit_linear_regression,predict


def the_plots(data, y_hat):
    """
    This function plots the graphs requested in question 21
    :param data: the dataset
    :param y_hat: the prediction vector
    :return:
    """
    # plot for the log detected graph
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title('”log detected” as a function of ”day num"')
    plt.scatter(data['day_num'],data['log_detected'], s=4)
    plt.plot(data['day_num'],y_hat)
    ax.set_xlabel('day_num')
    ax.set_ylabel('log_detected')
    fig.show()

    # plot for the detected graph
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title('”detected” as a function of ”day num"')
    plt.scatter(data['day_num'],data['detected'],s=4)
    plt.plot(data['day_num'],np.exp(y_hat))
    ax.set_xlabel('day_num')
    ax.set_ylabel('detected')
    fig.show()


def analyze_covid19():
    """
    This function contains the requested levels in the Exponential Regression section
    of the exercise
    """
    data = pd.read_csv('covid19_israel.csv')
    data['log_detected'] = np.log(data['detected'])
    X = data['day_num']
    X = X.values.reshape((1, X.shape[0]))
    y = data['log_detected']
    w, sig = fit_linear_regression(X, y)
    y_hat = predict(X,w)
    the_plots(data, y_hat)
