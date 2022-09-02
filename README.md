# Recommendation Ranking System
A model that finds, recommends and ranks products, based on a user's search and previous interactions

### Description of Datasets
Three datasets are available for this project.
1. `images.csv` This file contains all the image filenames and their associated labels
2. `Products.csv` This is the main dataset and it contains all the features needed to train the mmodel includin `product_name`, `price` and location.
3. `.jpg` These are the corresponding to the products

### Cleaning of datasets
First I checked if there are any missing data in both `images.csv` and `products.csv`. There was none.
Next I reshaped all `.jpg` images to a consistent size of (3, 56, 56) using the python library `Pillow`
I then filtered the Images.csv dataset by selecting those that had a corresponiding product ID.
I then merged the columns of the `Products.csv` to `Images.csv` to get the full complement of data
Next, I attached the numpy array associated with each image as an array to a column named `image_array`


