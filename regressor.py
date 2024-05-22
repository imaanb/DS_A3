import numpy as np
import pandas as pd
import tensorflow as tf

import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Lasso, Ridge
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from scipy.stats import randint, uniform
from scipy.sparse import csr_matrix
import warnings
import re
from keras import layers, models, optimizers
from sklearn.model_selection import RandomizedSearchCV
from scikeras.wrappers import KerasRegressor
from models import create_DL_model_shallow, create_DL_model_deep, create_knn_model, create_tree_model
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore")

# Initialize Snowball stemmer
stemmer = SnowballStemmer('english')

# Read a sample of the training data
df_train = pd.read_csv("C:/Users/imaan/OneDrive/Bureaublad/DS_A3/data/train.csv", encoding="ISO-8859-1", nrows=10000)

# Split the training data
#ratio = 0.8
#df_train, df_test = train_test_split(df_train, test_size=(1-ratio), random_state=42)

# Read a sample of product descriptions data
df_pro_desc = pd.read_csv('C:/Users/imaan/OneDrive/Bureaublad/DS_A3/data/product_descriptions.csv', nrows=3000)

# Read a sample of attributes data
df_attributes = pd.read_csv('C:/Users/imaan/OneDrive/Bureaublad/DS_A3/data/attributes.csv', nrows=1010)

# Combine data for preprocessing
df_concatenated = df_attributes.fillna('').groupby('product_uid')['value'].apply(lambda x: ' '.join(x)).reset_index()

# Merge dataframes
df_all = pd.merge(df_train, df_pro_desc, how='left', on='product_uid')
df_all = pd.merge(df_all, df_concatenated, how='left', on='product_uid')

# Stemming function
def str_stemmer(s):
    return " ".join([stemmer.stem(word) for word in s.lower().split()])

# Preprocess text data
df_all['search_term'] = df_all['search_term'].astype(str)
df_all['search_term'] = df_all['search_term'].map(lambda x: str_stemmer(x))

df_all['product_title'] = df_all['product_title'].astype(str)
df_all['product_title'] = df_all['product_title'].map(lambda x: str_stemmer(x))

df_all['product_description'] = df_all['product_description'].astype(str)
df_all['product_description'] = df_all['product_description'].map(lambda x: str_stemmer(x))

df_all['value'] = df_all['value'].fillna('')


# Preprocess text data
df_all['search_term'] = df_all['search_term'].map(lambda x: str_stemmer(x))
df_all['product_title'] = df_all['product_title'].map(lambda x: str_stemmer(x))
df_all['product_description'] = df_all['product_description'].map(lambda x: str_stemmer(x))

df_all['value'] = df_all['value'].fillna('')

# TF-IDF Vectorization
tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
search_term_tfidf = tfidf.fit_transform(df_all['search_term'])
product_title_tfidf = tfidf.transform(df_all['product_title'])
product_description_tfidf = tfidf.transform(df_all['product_description'])
attribute_description_tfdif = tfidf.transform(df_all['value'])


search_term_tfidf_sparse = csr_matrix(search_term_tfidf)
product_description_tfidf_sparse = csr_matrix(product_description_tfidf)
product_title_tfidf_sparse = csr_matrix(product_description_tfidf)
attribute_description_tfdif_sparse = csr_matrix(attribute_description_tfdif)

# Calculate cosine similarity between search term and product description (sparse)
cosine_sim_description = cosine_similarity(search_term_tfidf_sparse, product_description_tfidf_sparse)

# Calculate cosine similarity between search term and product title/description
cosine_sim_title = cosine_similarity(search_term_tfidf, product_title_tfidf)
#cosine_sim_description = cosine_similarity(search_term_tfidf, product_description_tfidf)
cosine_sim_attributes = cosine_similarity(search_term_tfidf, attribute_description_tfdif)

# Add cosine similarity as features
df_all['cosine_sim_title'] = cosine_sim_title.diagonal()
df_all['cosine_sim_description'] = cosine_sim_description.diagonal()
df_all["cosine_sim_attributes"] = cosine_sim_attributes.diagonal()

# Other features
df_all['len_of_query'] = df_all['search_term'].map(lambda x: len(x.split())).astype(np.int64)
df_all['len_of_title'] = df_all['product_title'].map(lambda x: len(x.split())).astype(np.int64)
df_all['len_of_description'] = df_all['product_description'].map(lambda x: len(x.split())).astype(np.int64)


# ADDING FEATURES

# adding features containing numbers and units
df_all['numbers_product_title'] = df_all['product_title'].map(lambda x: ' '.join(re.findall(r'\d+\.\d+|\d+', x)))
df_all['numbers_search_term'] = df_all['search_term'].map(lambda x: ' '.join(re.findall(r'\d+\.\d+|\d+', x)))
df_all['units_product_title'] = df_all['product_title'].map(lambda x: ' '.join(re.findall(r' [a-zA-Z]{2,3}\.', x)))
df_all['units_search_term'] = df_all['search_term'].map(lambda x: ' '.join(re.findall(r' [a-zA-Z]{2,3}\.', x)))

numbers_prod_title_tfidf = tfidf.transform(df_all['numbers_product_title'])
numbers_search_term_tfidf = tfidf.transform(df_all['numbers_search_term'])
units_prod_title_tfidf = tfidf.transform(df_all['units_product_title'])
units_search_term_tfidf = tfidf.transform(df_all['units_search_term'])

cosine_sim_numbers = cosine_similarity(numbers_search_term_tfidf, numbers_prod_title_tfidf)
cosine_sim_units = cosine_similarity(units_search_term_tfidf, units_prod_title_tfidf)

df_all['cosine_sim_numbers'] = cosine_sim_numbers.diagonal()
df_all['cosine_sim_units'] = cosine_sim_units.diagonal()

# Concatenate features
X_other_features = df_all[['len_of_query', 'len_of_title', 'len_of_description', 'cosine_sim_title',
                           'cosine_sim_description', 'cosine_sim_attributes', 'cosine_sim_numbers', 'cosine_sim_units']].values
X = np.hstack((search_term_tfidf.toarray(), X_other_features))

# Split data into train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, df_train['relevance'].values, test_size=0.2, random_state=42)



#COMPILING MODELS 


# 1. shallow deep learning network: 2 hidden layers: 
import keras
dl_model = create_DL_model_shallow(X_train.shape[1])
dl_model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.001), loss='mse', metrics=['mae'])  # Mean Squared Error loss for regression

# Train the deeplearning model
dl_history = dl_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

#train the machine learning model

# Evaluating the models
dl_test_loss, dl_test_mae = dl_model.evaluate(X_test, y_test)
print("Test Loss deep learning model:", dl_test_loss)
print("Test MAE deep leanring model:", dl_test_mae)

y_pred_dl = dl_model.predict(X_test)

# mean squared error
mse_dl = mean_squared_error(y_test, y_pred_dl)

# RMSE
dl_rmse = np.sqrt(mse_dl)

print("RMSE deep learning model:", dl_rmse)


train_loss = dl_history.history['loss']
val_loss = dl_history.history['val_loss']
train_rmse = [np.sqrt(loss) for loss in train_loss]
val_rmse = [np.sqrt(loss) for loss in val_loss]
epochs = range(1, len(train_rmse) + 1)


plt.figure(figsize=(10, 6))
plt.plot(epochs, train_rmse, 'b', label='Training RMSE')
plt.plot(epochs, val_rmse, 'r', label='Validation RMSE')
plt.title('Training and Validation RMSE per Epoch for shallow neural net')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)
plt.show()

#2. Deeper neural network: four hidden layers, convolutional layer
"""
import keras 
dl_model = create_DL_model_deep(X_train.shape[1])
dl_model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.001), loss='mse', metrics=['mae'])  # Mean Squared Error loss for regression

# Train the deeplearning model
dl_history = dl_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
"""
"""
train_loss = dl_history.history['loss']
val_loss = dl_history.history['val_loss']
train_rmse = [np.sqrt(loss) for loss in train_loss]
val_rmse = [np.sqrt(loss) for loss in val_loss]
epochs = range(1, len(train_rmse) + 1)


plt.figure(figsize=(10, 6))
plt.plot(epochs, train_rmse, 'b', label='Training RMSE')
plt.plot(epochs, val_rmse, 'r', label='Validation RMSE')
plt.title('Training and Validation RMSE per Epoch for deeper neural net')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)
plt.show()
"""


#train the machine learning model

# Evaluating the models
dl_test_loss, dl_test_mae = dl_model.evaluate(X_test, y_test)
print("Test Loss deep learning model:", dl_test_loss)
print("Test MAE deep learning model:", dl_test_mae)

y_pred_dl = dl_model.predict(X_test)

# mean squared error
mse_dl = mean_squared_error(y_test, y_pred_dl)
rmse = np.sqrt(mse_dl)
print('rsme deeper convolutional neural net:', rmse)


#3: Decision Tree Regressor
min_samples_leaf = 10
max_dept = 10
min_val = []
rsme_vals = [] 
dept_val = []
time_vals = [] 
max_dept = []


#finding the optimal value for the minimum amount of samples per leaf 

"""

for i in range(1, 10, 5):
    min_val.append(i)
    rf = create_tree_model(min_samples_leaf = i)
    begin_time = time.time()
    rf.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time-begin_time
    time_vals.append(training_time)
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    rsme_vals.append(rmse)
    print(f"RMSE for minimum leaf = {i}", rmse,"   traning time: ", training_time)

    """
# plots for finding the opt val for  min_samples per leave: 
"""
plt.figure(figsize=(10, 6))
plt.plot(min_val, rsme_vals, marker='o', linestyle='-', color='b')
plt.title('RMSE for random forrest Regression with Different values for minimum instances per leaf')
plt.xlabel('k')
plt.ylabel('RMSE')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(min_val, time_vals, marker='o', linestyle='-', color='b')
plt.title('Training time for random forrest Regression with Different values for minimum instances per leaf')
plt.xlabel('k')
plt.ylabel('training time')
plt.grid(True)
plt.show()
"""

#finding the optimal value for the maximum dept of the tree, after finding 31 = optimal min_sample_leaf
"""
for i in range(1, 50, 5):
    max_dept.append(i)
    rf = create_tree_model(max_dept = i, min_samples_leaf= 31)
    begin_time = time.time()
    rf.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time-begin_time
    time_vals.append(training_time)
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    rsme_vals.append(rmse)
    time_vals.append(training_time)
    print(f"RMSE for max dept = {i}", rmse,"   traning time: ", training_time)
"""
# plots for finding the opt val for  max_dept of the tree per leave: 
"""
plt.figure(figsize=(10, 6))
plt.plot(max_dept, rsme_vals, marker='o', linestyle='-', color='b')
plt.title('RMSE for random forrest Regression with Different values for the maximum dept of the tree')
plt.xlabel('max dept')
plt.ylabel('RMSE')
plt.grid(True)
plt.show()
"""

"""
plt.figure(figsize=(10, 6))
plt.plot(dept_val, time_vals, marker='o', linestyle='-', color='b')
plt.title('Training time for random forrest Regression with Different values for the maximum dept of the tree')
plt.xlabel('max dept')
plt.ylabel('training time')
plt.grid(True)
plt.show()
"""


#4. KNN Regressor 

n = 20
k_vals = []
rsme_vals = []

"""
knn = create_knn_model(n)
start_time = time.time()
knn.fit(X_train, y_train)
end_time = time.time()
training_time = end_time-start_time
print("training time knn", training_time)
y_pred = knn.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("RSME knn regressor:", rmse )

"""

"""
for i in range(n): 
    k = 1+i
    k_vals.append(k)
    knn = create_knn_model(k)
    start_time = time.time()
    knn.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time-start_time
    print(f"training time knn, k = {k}", training_time)
    y_pred = knn.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rsme = np.sqrt(mse)
    rsme_vals.append(rsme)
    print(f"RSME knn regressor, k = {k}:", rsme )

plt.figure(figsize=(10, 6))
plt.plot(k_vals, rsme_vals, marker='o', linestyle='-', color='b')
plt.title('RMSE for k-NN Regression with Different k Values')
plt.xlabel('k')
plt.ylabel('RMSE')
plt.grid(True)
plt.show()

"""