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

X, y = loadDataset()



def test_prediction():
	pass

def test_loss_function():
	pass

def test_gradient():
	pass

def test_training():
	pass