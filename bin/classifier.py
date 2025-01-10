# import libraries
from utils.helpers import get_element, flatten_array, convert_arrays_to_columns

import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, FunctionTransformer
from sklearn.pipeline import make_pipeline

# instatiate models and transformers
le = LabelEncoder()
ft1 = FunctionTransformer(flatten_array)
ft2 = FunctionTransformer(convert_arrays_to_columns)
clf = RandomForestClassifier()

# import data
data = pd.read_pickle("image_pr.pkl")


# set feature matrix X and response vector y
X = data.image_array
y = le.fit_transform(
        data.category.str.split("/").apply(get_element, position=0))

# glance of the data before split
print("Feature Matrix\n", X.head())
print("Response Vector\n", y[0:5])


# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

print("Training input shape", X_train.shape)
print("Training target shape", y_train.shape)
print("Testing input shape", X_test.shape)
print("Testing target shape", y_test.shape)


# make pipeline of transformers and classifier
pipe = make_pipeline(ft1, ft2, clf)
# train the classifier
pipe.fit(X_train, y_train)
# make predictions on out of sample data
print(metrics.accuracy_score(y_test, pipe.predict(X_test)))