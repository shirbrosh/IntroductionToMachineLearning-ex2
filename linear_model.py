import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def fit_linear_regression(X, y):
    """
    This function receives a design matrix ‘X‘ and a response vector ‘y‘ and returns
    the coefficients vector ‘w‘ and the singular values of X
    :param X: a design matrix
    :param y: a response vector
    :return: the coefficients vector ‘w‘ and the singular values of X
    """
    X_ones = np.concatenate((np.ones((1, X.shape[1])), X), axis=0)
    X_ones_daggerT = np.linalg.pinv(X_ones.T)
    w = X_ones_daggerT @ y
    singular_values = np.linalg.svd(X, compute_uv=False)
    return w, singular_values


def predict(X, w):
    """
    This function receives a design matrix ‘X‘ and a coefficients vector ‘w‘ and
    returns a predicted value by the model.
    :param X: a design matrix
    :param w: a coefficients vector
    :return: a numpy array with the predicted value by the model.
    """
    X_ones = np.concatenate((np.ones((1, X.shape[1])), X), axis=0)
    return (X_ones.T) @ w


def mse(y, y_hat):
    """
    This function receives a response vector and a predict vector and returns the MSE
    over the received samples
    :param y: a response vector
    :param y_hat: a predict vector
    :return: the MSE over the received samples
    """
    m = y.shape[0]
    return (1 / m) * np.sum(np.power(y_hat - y, 2))


def load_data(path):
    """
    This function receives a path to a csv file loads the dataset and performs all
    the needed preprocessing so to get a valid design matrix
    :param path: a path to a csv file
    :return: a valid design matrix and its label vector
    """
    data = pd.read_csv(path)

    # remove rows with empty values:
    data = data.dropna(axis=0)

    # remove rows with invalid values:
    data = data.drop(
        data[(data.price <= 0) | (data.bedrooms <= 0) | (data.bedrooms >= 20) | (
                data.floors <= 0) | (data.bathrooms <= 0) | (data.sqft_lot <= 0) | (
                     data.sqft_lot15 <= 0) | (
                     data.sqft_living <= 0) | (
                     data.condition <= 0)].index)

    # convert the date feature to contain only the year
    data['date'] = (pd.to_datetime(data['date']).dt.strftime("%Y")).astype(float)

    # handle categorical features:
    data = pd.concat([data, pd.get_dummies(data["zipcode"], prefix='zipcode')],
                     axis=1)

    y = data['price']

    # drop the features that do not contribute
    data = data.drop(["id", "price", "zipcode"], axis=1)
    return data, y


def plot_singular_values(sing_vals):
    """
    This function receives a collection of singular values and plots them in
    descending order
    :param sing_vals: a collection of singular values
    """
    sing_vals[::-1].sort()
    # x= np.arange(1,sing_vals.shape[0]+1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title('Singular values in descending')
    plt.plot(sing_vals, '-ok')
    plt.yscale('log')
    ax.set_xlabel('index number')
    ax.set_ylabel('singular value’s value')
    fig.show()


def q_15():
    """
    This function uses the function plot_singular_values to plot the singulae values
    for the data in the 'kc_house_data.csv' after being process by the load_data function.
    """
    data, y = load_data('kc_house_data.csv')
    X = data.T
    w, sig = fit_linear_regression(X, y)
    plot_singular_values(sig)


def q_16():
    """
    This function fit a model and test it over the data in the 'kc_house_data.csv'
    file, it also creates a plot of MSE over the test set as a function of p%
    """
    X, y = load_data('kc_house_data.csv')

    # splits the data, 1/4- for the test data and 3/4 for the train data
    split = int(X.shape[0] * 0.25)

    testX = X[:split]
    testy = y[:split]

    trainX = X[split:]
    trainy = y[split:]

    err = []
    for p in range(1, 101):
        # split the training data to the first p%
        split_p = int(trainX.shape[0] * (p / 100))
        trainX_p = trainX[:split_p]
        trainy_p = trainy[:split_p]

        w, sig = fit_linear_regression(trainX_p.T, trainy_p)
        y_hat = predict(testX.T, w)
        err.append(mse(testy, y_hat))

    # plot the mse
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title('MSE over the test set as a function of p%')
    plt.plot(np.array(err), '-ok')
    plt.yscale('log')
    ax.set_xlabel('p%')
    ax.set_ylabel('MSE')
    fig.show()


def feature_evaluation(X, y):
    """
    This function, given the design matrix and response vector, plots for every non-
    categorical feature, a graph (scatter plot) of the feature values and the response
    values. It then also computes and shows on the graph the Pearson Correlation
    between the feature and the response
    :param X: a design matrix
    :param y: a response vector
    """
    y = y.values.reshape((1, y.shape[0]))
    i = 0
    for label in X.items():
        # break after plotting 18 features- without plotting zipcodes features
        if i == 18:
            break

        # Pearson Correlation:
        feature = X.values.T[i]
        p = np.cov(y, feature)[0][1] / np.sqrt(np.var(y) * np.var(feature))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.title(
            'price VS. ' + str(label[0]) + " ,with Pearson Correlation of " + str(p))
        plt.scatter(y, feature)
        ax.set_ylabel(label[0])
        ax.set_xlabel('price')
        plt.xticks(rotation='vertical')
        fig.show()
        i += 1
