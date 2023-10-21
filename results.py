from matplotlib import pyplot as plt
from experiment import Experiment


def try_naive_bayes():
    e = Experiment()
    accuracies = []
    smoothing = [0.1, 0.5, 1.0, 2, 4, 8, 10, 100, 200, 400, 600, 800, 1000]
    for i in smoothing:
        accuracy: float = e.do_naive_bayes(i)
        accuracies.append((accuracy, i))

    sorted_accuracies = sorted(accuracies, key=lambda x: x[1])

    smoothing_values = [entry[1] for entry in sorted_accuracies]
    accuracy_values = [entry[0] for entry in sorted_accuracies]

    plt.plot(smoothing_values, accuracy_values, marker="o")
    plt.xlabel("Smoothing")
    plt.ylabel("Accuracy (%)")
    plt.title("Smoothing vs Accuracy in Naive Bayes for BBC")
    plt.grid(True)
    plt.show()


def try_logistic():
    e = Experiment()
    learning_rates = [0.001, 0.01, 0.1, 1.0, 1.5, 5, 10, 100, 1000]
    accuracies = []
    for i in learning_rates:
        accuracy: float = e.do_logistic_regression(learning_rate=i, num_iter=100)
        accuracies.append((accuracy, i))

    sorted_accuracies = sorted(accuracies, key=lambda x: x[1])

    learning_rates_values = [entry[1] for entry in sorted_accuracies]
    accuracy_values = [entry[0] for entry in sorted_accuracies]

    plt.plot(learning_rates_values, accuracy_values, marker="o")
    plt.xlabel("Learning Rates")
    plt.ylabel("Accuracy (%)")
    plt.title("Learning Rate vs Accuracy in Logistic Regression for BBC")
    plt.grid(True)
    plt.show()


try_logistic()
# try_naive_bayes()
