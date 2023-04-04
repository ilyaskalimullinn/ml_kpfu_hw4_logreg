import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Visualisation:
    def plot_accuracy(self, accuracy_valid: np.ndarray, accuracy_train: np.ndarray, number_of_epochs: int,
                      plot_title: str = "Accuracy of model", save_path: str = None):
        fig = go.Figure()

        epochs = np.arange(1, number_of_epochs + 1)

        fig.add_trace(go.Scatter(
            x=epochs,
            y=accuracy_valid,
            mode='lines',
            name='Accuracy on validation set',
        ))

        fig.add_trace(go.Scatter(
            x=epochs,
            y=accuracy_train,
            mode='lines',
            name='Accuracy on training set',
        ))

        fig.update_layout(
            title=plot_title,
            xaxis_title="Epochs",
            yaxis_title="Accuracy",
            legend_title="Legend",
        )

        fig.show()

        if save_path:
            fig.write_html(save_path)

    def plot_target_function(self, target_func_values: np.ndarray, number_of_epochs: int,
                             plot_title: str = "Target function values on train set, ln", save_path: str = None):
        fig = go.Figure()

        epochs = np.arange(1, number_of_epochs + 1)

        fig.add_trace(go.Scatter(
            x=epochs,
            y=np.log(target_func_values),
            mode='lines',
            name='Target function values on training set, ln',
        ))

        fig.update_layout(
            title=plot_title,
            xaxis_title="Epochs",
            yaxis_title="Target function value, ln",
            legend_title="Legend",
        )

        fig.show()

        if save_path:
            fig.write_html(save_path)

    def plot_matched_and_unmatched(self, predictions_matched_max, inputs_matched_max, targets_matched_max,
                                   predictions_not_matched_max, inputs_not_matched_max, targets_not_matched_max,
                                   save_path: str = None, title: str = "Best and worst predictions"):
        subplot_titles = []

        for pred, target in zip(predictions_matched_max, targets_matched_max):
            subplot_titles.append(f"Target: {target}, prediction: {pred}")
        for pred, target in zip(predictions_not_matched_max, targets_not_matched_max):
            subplot_titles.append(f"Target: {target}, prediction: {pred}")

        fig = make_subplots(cols=predictions_matched_max.shape[0], rows=2, subplot_titles=subplot_titles)

        for i in range(inputs_matched_max.shape[0]):
            fig.add_trace(go.Heatmap(z=inputs_matched_max[i].reshape(8, 8)[::-1], opacity=0.8, colorscale='Greens'), row=1, col=i+1)

        for i in range(inputs_not_matched_max.shape[0]):
            fig.add_trace(go.Heatmap(z=inputs_not_matched_max[i].reshape(8, 8)[::-1], opacity=0.8, colorscale='Reds'), row=2,
                          col=i + 1)

        fig.update_layout(
            title=title
        )

        fig.show()

        if save_path:
            fig.write_html(save_path)
