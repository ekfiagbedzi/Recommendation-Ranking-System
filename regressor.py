# import libraries
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn import metrics


# read cleaned data
products = pd.read_pickle("image_product.pkl")


# define feature matrix X and response vector y
X = products.loc[:, ["product_name", "product_description", "location"]]
y = products.price


# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# make column transform to transform text columns of feature matrix
column_trans = make_column_transformer(
    (TfidfVectorizer(), "product_name"),
    (TfidfVectorizer(), "product_description"),
    (TfidfVectorizer(), "location")
    )


# define a pipeline
pipe = make_pipeline(column_trans, LinearRegression())

# transform columns and train the model
pipe.fit(X_train, y_train)


# make predictions on the test set
y_pred = pipe.predict(X_test)


# calculate rmse of predictions
print("RMSE of model = {}".format(
    metrics.mean_squared_error(y_test, y_pred, squared=False)))