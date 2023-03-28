import numpy as np
import plotly.graph_objects as go


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
