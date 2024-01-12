# Logistic regression

1. Write logistic regression model for K classes and train it on digits dataset.
2. Divide dataset on training validation and test sets.
3. Before training use standardisation preprocessing. You need to write it by yourself (not library).
4. Write code for creating one-hot encoding vectors. You need to write it by yourself (not library).
5. Use total gradient descent for model training. Init weights using normal distribution. Stopping criteria number of training iterations
6. During training periodically print target function value on train set, Accuracy and confusion matrix on validation and train set. 
7. Make plot (plotly) where y axis is target function value on training set and x axis is number of training iteration
8. Make plot where y axis is accuracy value on training set and x axis is number of training iteration
9. Make plot where y axis is accuracy value on validation set and x axis is number of training iteration
10. Print test set accuracy and confusion matrix on validation set after training

## Bonus task

1. Use normalisation preprocessing instead of standardisation 
2. Add regularisation for gradient descent
3. Write code for batch gradient descent (gradient is calculated only on a part of data with fix size, gradient need to be calculate in whole dataset, use loop for this).
4. Write code for different initialisation (uniform distribution, Xavier, He)
5. Write code for different stopping criteria
6. Write code for saving and loading model (for example pickle)
7. Visualise digits images: 3 images for which classifier saved most confidence class prediction and was right.  3 images for which classifier saved most confidence class prediction and was wrong.

![image](https://github.com/ilyaskalimullinn/ml_kpfu_hw4_logreg/assets/90423658/6c96d4ed-02e1-4bf3-bc18-5fd5efc343c4)

![image](https://github.com/ilyaskalimullinn/ml_kpfu_hw4_logreg/assets/90423658/24c18849-baef-4b1b-9b4e-8ab6165c9455)

![image](https://github.com/ilyaskalimullinn/ml_kpfu_hw4_logreg/assets/90423658/ca2ea2d1-cf58-48f0-8ebc-c53eed57f9f4)

