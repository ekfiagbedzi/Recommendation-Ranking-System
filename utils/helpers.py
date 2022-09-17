import os

import pandas as pd
import cv2
from PIL import Image
import fnmatch
import torchvision
from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import Dataset


def resize_image(final_size, im):
    """Resize all images to same sizes
       Args:
            final_size: (int) Size of image in pixels needed
            im: (Image object) Image to be processed
            
        Returns:
            Resized image
    """
    
    size = im.size
    ratio = float(final_size) / max(size) 
    new_image_size = tuple([int(x*ratio) for x in size])
    im = im.resize(new_image_size)
    new_im = Image.new("RGB", (final_size, final_size))
    new_im.paste(
        im,
        (0, 0)) # define coordinate as upper left corner
    return new_im

def clean_image_data(path, final_size, save_path):
    """Reshape an image to a specified size
       Args:
            path: (str) - Location of image
            final_size: (int) - Size in pixels of image
            save_path: (str) - Location to store images
       Returns:
            New Image with specified size
    """
    
    dirs = fnmatch.filter(os.listdir(path), "*.jpg")
    for item in dirs:
        im = Image.open(path + item)
        new_im = resize_image(final_size, im)
        item = item.rstrip(".jpg")
        new_im.save(f"{save_path}/{item}_resized.jpg")
    print("Images Cleaned Succesfully. Have a Nice Day!!!")


def get_element(list, position):
    """Get a member from a pandas series of lists by position
       Args:
            list: (python list) A python list
            position: Index to select member
       Return:
             (obj) A member of list from position
    """
    return list[position]


def image_to_array(img_id):
    """Convert Image to a numpy array
       Args:
            img_id (str): ID of Image
       Return:
            Array of pixel values
    """
    img = cv2.imread(
        "cleaned_images/{}_resized.jpg".format(img_id)
)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def flatten_array(ser):
    """Converts array of any shape in a Series to 1D array
       Args:
            ser (pandas.Series): Series object containing arrays
       Return:
            pandas.Series object containing 1D arrays
    """
    ser.apply(lambda x: x.resize(2352))
    return ser


def convert_arrays_to_columns(ser):
    """Convert a pandas Series of 1D arrays of shape n into n DataFrame columns
       Args: (ser)
            pandas.Series object containing 1D arrays of shape (n,)
       Return: (pandas.DataFrame)
            pandas DataFrame with n columns
    """
    return pd.DataFrame(ser.values.tolist())


class ImageDataset(Dataset):
    """Create a PyTorch Dataset
        Args:
              features (*array) - Any array of values
              labels (*array) - Any array of values
        
        Return:
              (obj) torch.utils.data.Dataset
    """
    le = LabelEncoder() # encoder/decoder attribute
    
    def __init__(self, features=None, labels=None):
        super().__init__()
        self.features = features
        self.labels = labels

    def __getitem__(self, index):
        """Get example at specified index"""
        return self.features[index], self.labels[index]

    def __len__(self):
        """Get number of examples in dataset"""
        return len(self.features)

    
    @classmethod
    def load_data(cls, data):
        """Alternative PyTorch Dataset constructor
           Args:
                data (pandas.DataFrame) - A pandas DataFrame object

           Return:
              (obj) torch.utils.data.Dataset
        """
        features = []
        labels = []
        ind = 0
        IDs = data.id.tolist()
        cats = ImageDataset.le.fit_transform(
        data.category.str.split("/").apply(get_element, position=0))
        for ID in IDs:
            img_path = "cleaned_images/{}_resized.jpg".format(ID)
            with Image.open(img_path) as im:
                features.append(torchvision.transforms.functional.to_tensor(im))
            labels.append(torch.tensor(cats[ind]))
            ind += 1
        return cls(features, labels)