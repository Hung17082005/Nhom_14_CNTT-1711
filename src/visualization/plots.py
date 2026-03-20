import matplotlib.pyplot as plt

def plot_learning_curve(percentages, scores):
    plt.plot(percentages, scores)
    plt.xlabel("Labeled Data %")
    plt.ylabel("F1 Score")
    plt.title("Semi-supervised Learning Curve")
    plt.show()