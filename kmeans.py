import numpy as np
import matplotlib.pyplot as plt


def kmeans(X, k, t):
    """
    :param X: numpy array of size (m, d) containing the test samples
    :param k: the number of clusters
    :param t: the number of iterations to run
    :return: a column vector of length m, where C(i) ∈ {1, . . . , k} is the identity of the cluster in which x_i has been assigned.
    """
    m, d = X.shape
    # Initialize the centroids randomly
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    for _ in range(t):
        # Compute the distances between each sample and the centroids
        distances = np.sqrt(np.sum((X[:, np.newaxis] - centroids) ** 2, axis=-1))
        # Assign each sample to the closest centroid
        clusters = np.argmin(distances, axis=1)
        # Update the centroids to be the mean of the samples in each cluster
        new_centroids = np.array([X[clusters == c].mean(axis=0) for c in range(k)])
        # Check for convergence
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    clusters += 1
    return clusters.reshape((m, 1))


def simple_test():
    # load sample data (this is just an example code, don't forget the other part)
    data = np.load('mnist_all.npz')
    X = np.concatenate((data['train0'], data['train1']))
    m, d = X.shape

    # run K-means
    c = kmeans(X, k=10, t=10)

    assert isinstance(c, np.ndarray), "The output of the function softsvm should be a numpy array"
    assert c.shape[0] == m and c.shape[1] == 1, f"The shape of the output should be ({m}, 1)"

if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()

    # here you may add any code that uses the above functions to solve question 2

    # Generate a random sample of size 1000
    data = np.load('mnist_all.npz')
    train0 = data['train0']
    train1 = data['train1']
    train2 = data['train2']
    train3 = data['train3']
    train4 = data['train4']
    train5 = data['train5']
    train6 = data['train6']
    train7 = data['train7']
    train8 = data['train8']
    train9 = data['train9']
    x_list = [train0, train1, train2, train3, train4, train5, train6, train7, train8, train9]
    y_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    x = np.vstack(x_list)
    y = np.concatenate([y_list[j] * np.ones(x_list[j].shape[0]) for j in range(len(y_list))])
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    rearranged_x = x[indices]
    rearranged_y = y[indices]
    X, Y = rearranged_x[:1000], rearranged_y[:1000].astype(int)

    k = 6
    # Run k-means on the sample
    clusters = kmeans(X, k, 10)
    # Initialize the table
    table = []
    # initialize the predictions
    preds = np.copy(Y)

    # Iterate over the clusters
    for c in range(k):
        # Get the indices of the examples in the cluster
        indices = np.where(clusters == c + 1)[0]
        # Get the true labels of the points in the cluster
        labels = Y[indices]
        # Count the occurrences of each label in the cluster
        counts = np.bincount(labels)
        # Find the most common label in the cluster
        most_common_label = np.argmax(counts)
        # assign most common label to predictions of this cluster examples
        preds[indices] = most_common_label
        # Calculate the percentage of examples with the most common label
        percentage = (counts[most_common_label] / len(labels)) * 100
        # Append the information to the table
        table.append([len(labels), most_common_label, percentage])

    # Convert the table to a numpy array
    table = np.array(table)

    # calculate classification error
    error = np.mean(Y != preds) * 100

    # Plot the table using matplotlib
    plt.bar(np.arange(1, k+1), table[:, 0], align='center', alpha=0.5)
    plt.xticks(np.arange(1, k+1))
    plt.xlabel('Cluster')
    plt.ylabel('Size')
    plt.title('Size of Clusters')
    for i in range(k):
        plt.text(i+1, table[i, 0], str(int(table[i, 1])), ha='center', color='red')
        plt.text(i+1, table[i, 0]+max(table[:, 0])*0.05, str(round(table[i, 2], 2))+"%", ha='center', color='blue')
    plt.text(0.5, max(table[:, 0]) + (max(table[:, 0]) * 0.34), "classification error - "+str(round(error, 2))+"%",
             color='black', fontsize=10)
    plt.text(0.5, max(table[:, 0]) + (max(table[:, 0]) * 0.28), "percentage", color='blue', fontsize=10)
    plt.text(0.5, max(table[:, 0]) + (max(table[:, 0]) * 0.22), "most common label", color='red', fontsize=10)
    plt.ylim(0, max(table[:, 0])*1.4)
    plt.subplots_adjust(left=0.09, right=0.97)
    plt.show()

