"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
# (you will probably need to import more things here)
import numpy as np
from regression.utils import loadDataset
from regression.logreg import BaseRegressor, LogisticRegressor


#tests were written conferring with chatgpt

X, y = loadDataset()

n = X.shape[0]

indices = np.random.permutation(n)
split_data = int(0.8 * n)

X_train, y_train = X[indices[:split_data]], y[indices[:split_data]]
X_test, y_test = X[indices[split_data:]], y[indices[split_data:]]

X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))]) #we add in the bias term
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))]) #we add in the bias term

model = LogisticRegressor(num_feats=X.shape[1], learning_rate=0.003)



def test_prediction():

	y_pred = model.make_prediction(X_train)

	assert isinstance(y_pred, np.ndarray)
	assert y_pred.shape[0] == X_train.shape[0]
	assert np.all(y_pred <= 1)
	assert np.all(y_pred >= 0)


def test_loss_function():
	y = np.array([0, 1, 1, 0])
	y_pred = np.array([0.2, 0.6, 0.8, 0.1])

	loss = model.loss_function(y, y_pred)
	assert isinstance(loss, float)
	assert loss > 0
	assert np.isclose(loss, 0.266, rtol=1e-2) 



def test_gradient():

	y_pred = model.make_prediction(X_train)
	gradient = model.calculate_gradient(y_train, X_train)

	manual_gradient = (1 / X_train.shape[0]) * np.dot(X_train.T, (y_pred - y_train))

	assert isinstance(gradient, np.ndarray)
	assert gradient.shape == model.W.shape
	assert np.allclose(gradient, manual_gradient, atol=1e-6)

	


def test_training():

	initial_y_train = model.make_prediction(X_train)
	initial_loss = model.loss_function(y_train, initial_y_train)
	initial_weights = model.W.copy()

	assert X_train.shape[1] == X_test.shape[1]

	model.train_model(X_train, y_train, X_test, y_test)

	after_y_train = model.make_prediction(X_train)
	after_loss = model.loss_function(y_train, after_y_train)

	after_weights = model.W.copy()

	assert after_loss < initial_loss
	assert not np.allclose(initial_weights, after_weights)

