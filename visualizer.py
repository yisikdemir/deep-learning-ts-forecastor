import matplotlib.pyplot as plt
import numpy as np


class ResultsPlotter:
    def __init__(self, figsize=(8, 4)):
        self.figsize = figsize

    def plot_results(
        self,
        y_test,
        models_results,
        model_names,
        y_label="Number of Passengers",
        x_label="Observation Index",
        legend_loc="upper left",
    ):

        plt.figure(figsize=self.figsize)

        plt.plot(y_test, label="actual")

        for i in range(len(models_results)):
            plt.plot(models_results[i], label=model_names[i], alpha=0.8, ls="--", lw=2)

        plt.ylabel(y_label, fontdict={"weight": "bold"})
        plt.xlabel(x_label, fontdict={"weight": "bold"})
        plt.legend(loc=legend_loc)
        plt.show()


if __name__ == "__main__":
    # Generate some sample data
    y_test = np.random.rand(10)
    vanilla_lstm_results = np.random.rand(10)
    stacked_lstm_results = np.random.rand(10)
    bidirectional_lstm_results = np.random.rand(10)

    # Plot the results
    plotter = ResultsPlotter()

    models_results = [
        vanilla_lstm_results,
        stacked_lstm_results,
        bidirectional_lstm_results,
    ]
    model_names = ["Vanilla LSTM", "Stacked LSTM", "Bidirectional LSTM"]

    plotter.plot_results(y_test, models_results, model_names)
