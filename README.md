# Recommendation Ranking System
A model that finds, recommends and ranks products, based on a user's search and previous interactions

### Description of Datasets
Three datasets are available for this project.
1. `images.csv` This file contains all the image filenames and their associated labels
2. `Products.csv` This is the main dataset and it contains all the features needed to train the model includin `product_name`, `price` and `location`.
3. `.jpg` These are the corresponding to the products

### Cleaning of datasets
First I checked if there are any missing data in both `images.csv` and `products.csv`. There was none.
Next I reshaped all `.jpg` images to a consistent size of (3, 56, 56) using the python library `Pillow`
I then filtered the Images.csv dataset by selecting those that had a corresponiding product ID.
I then merged the columns of the `Products.csv` to `Images.csv` to get the full complement of data
Next, I attached the numpy array associated with each image as an array to a column named `image_array`


### Creation of baseline regression model
I created a regression model to predict the price of an item using `product_name`, `product_description`, and `location` as features.
I converted each feature into count words, using `TFIDF-Vectorizer`, which i fed into the model
I used a `LinearRegression` model from `sklearn.linear_model` module and obtained a Root Mean Squared Error (RMSE) of 101695.2935
which is very high and can be imporved by feature engineering such as removing stop words from text before vectorization


### Creation of baseline classification model
I created an image classification model to predict the class of a product based on its associated image
I first converted each image into numpy arrays and stored them in a pandas DataFrame
Then, I flattened each array and into 1D arrays and converted these arrays into pandas DataFrame columns using `FunctionTransforemer`
I then combined these transformer steps with a `RandomForest` classifier into a pipeline and trained this pipe using my training data
I then predicted on the out of sample test data and obtained an accuracy of 20% which is very low.
However, a better model can be made using deep neural networks


