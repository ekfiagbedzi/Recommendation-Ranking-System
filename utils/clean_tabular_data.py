# %%
# import libraries
import os

from helpers import image_to_array
import pandas as pd


# load both image and product dataframes
images = pd.read_csv("../Images.csv")
products = pd.read_csv("../Products.csv", lineterminator="\n")


# make corresponding ids the indices of both dataframes
images.set_index("product_id", inplace=True)
products.set_index("id", inplace=True)


# drop first occurences of duplicates in both dataframes
images_df1 = images[images.index.duplicated(keep="first")]
products = products[~products.index.duplicated(keep="first")]


# extract common ids between the two dataframes
common_ids= list(set(images_df1.index).intersection(set(products.index)))


# combine dataframes with first occurences kept
combined_df1 = pd.concat(
    [images_df1.loc[common_ids], products.loc[common_ids]], axis=1)


# drop last occurences of duplicates in both dataframes
images_df2 = images[images.index.duplicated(keep="last")]
products = products[~products.index.duplicated(keep="last")]


# combine dataframes with last occurences kept
combined_df2 = pd.concat(
    [images_df2.loc[common_ids], products.loc[common_ids]], axis=1)


# join the two dataframes
combined = pd.concat([combined_df1, combined_df2])


# keep only first of duplicates
combined.drop_duplicates(subset=["bucket_link"], keep="first", inplace=True)


# clear cache
combined_df1=None
combined_df2=None
images=None
images_df1=None
images_df2=None
products=None
products_unique=None
common_ids=None


# convert images to numpy arrays and store in dataframe
combined["image_array"] = combined.loc[:, "id"].apply(image_to_array)


# select relevant columns
combined.reset_index(inplace=True)
combined = combined.iloc[:, [2, 7, 8, 9, 10, 11, 14, 15]]


# convert price dtype to float
combined.price = combined.loc[
    :, "price"].str.replace("£", "").str.replace(",", "").astype(float)


# convert create_time dtype to float
combined.create_time = pd.to_datetime(combined.create_time)


# save data as a pickle file
pd.to_pickle(combined, "../image_product.pkl")

# %%
