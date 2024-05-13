import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import Lasso, Ridge
from sklearn.svm import SVR
from scipy.stats import randint, uniform
import warnings
import re
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore")

# Initialize Snowball stemmer
stemmer = SnowballStemmer('english')


# Read the data
df_train = pd.read_csv("C:/Users/imaan/OneDrive/Bureaublad/DS_A3/data/train.csv", encoding="ISO-8859-1")
ratio = 0.8
df_train, df_test = train_test_split(df_train, test_size=(1-ratio), random_state=42)

df_pro_desc = pd.read_csv('C:/Users/imaan/OneDrive/Bureaublad/DS_A3/data/product_descriptions.csv')
df_pro_desc = df_pro_desc.iloc[:3000]

df_attributes = pd.read_csv('C:/Users/imaan/OneDrive/Bureaublad/DS_A3/data/attributes.csv')
df_attributes = df_attributes.iloc[:1010]
df_concatenated = df_attributes.fillna('').groupby('product_uid')['value'].apply(lambda x: ' '.join(x)).reset_index()


# Combine train and test data for preprocessing
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
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


print(df_all[:100])