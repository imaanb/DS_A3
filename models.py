import numpy as np
import pandas as pd
import tensorflow as tf

import time
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import Lasso, Ridge
from sklearn.svm import SVR
from scipy.stats import randint, uniform
from scipy.sparse import csr_matrix
import warnings
import re
from keras import layers, models
from sklearn.model_selection import RandomizedSearchCV
from scikeras.wrappers import KerasRegressor
from sklearn.neighbors import KNeighborsRegressor


#   DL 
def create_DL_model_shallow(input_shape, units_1=128, units_2=64, dropout_rate=0.0):
    model = models.Sequential()
    model.add(layers.Dense(units_1, activation='relu', input_shape=(input_shape,)))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(units_2, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1))
    return model

def create_DL_model_deep(input_shape, units_1=128, units_2=64, units_3=64, units_4=32, dropout_rate=0.0, conv_filters=32, conv_kernel_size=3):
    model = models.Sequential()
    
    # Convolutional layer
    model.add(layers.Conv1D(filters=conv_filters, kernel_size=conv_kernel_size, activation='relu', input_shape=(input_shape, 1)))
    model.add(layers.Flatten())  # Flatten after Conv1D for Dense layers
    
    # First hidden layer
    model.add(layers.Dense(units_1, activation='relu'))
    if dropout_rate > 0.0:
        model.add(layers.Dropout(dropout_rate))
    
    # Second hidden layer
    model.add(layers.Dense(units_2, activation='relu'))
    if dropout_rate > 0.0:
        model.add(layers.Dropout(dropout_rate))
    
    # Third hidden layer
    model.add(layers.Dense(units_3, activation='relu'))
    if dropout_rate > 0.0:
        model.add(layers.Dropout(dropout_rate))
    
    # Fourth hidden layer
    model.add(layers.Dense(units_4, activation='relu'))
    if dropout_rate > 0.0:
        model.add(layers.Dropout(dropout_rate))
    
    # Output layer
    model.add(layers.Dense(1))
    
    return model

# ML

def create_knn_model(n): 
    return KNeighborsRegressor(n)


def create_tree_model():
    return  DecisionTreeRegressor(random_state = 42)
