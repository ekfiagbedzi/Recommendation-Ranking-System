import os

import pandas as pd
import cv2
from PIL import Image
import fnmatch
from sklearn.preprocessing import LabelEncoder

import torch
import torchvision
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel


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


def get_element(list: list=None, position: int=0):
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
        "data/cleaned_images/{}_resized.jpg".format(img_id)
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


def image_processor(img_path: str=None, transformers: object=None):
    """Process images to feed into Pytorch model
       Args:
            img_path: (str) - Path/Buffer to image
            transformers: (torchvision.transforms.Compose Object) - List of
            transformers compiled into a torchvision.transforms.Compose object
            
       Returns:
            torch.tensor with shape (1, n_channels, width, height)
    """
    
    with Image.open(img_path) as im:
        if transformers:
            im = transformers(im)
        else:
            im = torchvision.transforms.functional.to_tensor(im)
    return im.unsqueeze(dim=0)


def text_processor(sentence: str=None, model=None, tokenizer=None, max_length: int=None, truncation: bool=True):
    """Process text to feed Pytorch model
       Args:
            sentence: (str) - Text to process
            model: (BertModel) - Model to apply processing
            tokenizer: (BertTokenizer) - Tokenizer object to convert text into embeddings
            max_length: (int) - Maximum lenght of each sentence based on padding and truncation
            padding: (int) - Maximum level to pad to
            truncation: (bool) - If True, Shorten embedding to max_length
            
       Return:
            Embeddings as 3D torch Tensor of batch_size=1
    
    """
    encoded = tokenizer.batch_encode_plus([sentence], max_length=max_length, padding="max_length", truncation=truncation)
    encoded = {key:torch.LongTensor(value) for key, value in encoded.items()}
    with torch.no_grad():
        description = model(**encoded).last_hidden_state.swapaxes(1, 2)

    return description
    


class ImageDataset(Dataset):
    """Create a PyTorch Dataset
        Args:
              data: (pandas.DataFrame) - A pandas DataFrame
              transformers: (torchvision.transforms.Compose Object) - List of
            transformers compiled into a torchvision.transforms.Compose object
        
        Return:
              (obj) torch.utils.data.Dataset
    """
    
    le = LabelEncoder() # encoder/decoder attribute
    
    def __init__(self, data, transformers: object=None):
        super().__init__()
        features = []
        labels = []
        ind = 0
        IDs = data.id.tolist()
        cats = ImageDataset.le.fit_transform(
        data.category.str.split("/").apply(get_element, position=0))
        for ID in IDs:
            img_path = "data/cleaned_images/{}_resized.jpg".format(ID)
            with Image.open(img_path) as im:
                if transformers:
                    features.append(transformers(im))
                else:
                    features.append(
                        torchvision.transforms.functional.to_tensor(im))
            labels.append(torch.tensor(cats[ind]))
            ind += 1
        self.features = features
        self.labels = labels
        self.encoder = dict(
            zip(self.le.inverse_transform(self.le.transform(self.le.classes_)), self.le.classes_))
        self.decoder = dict(enumerate(self.le.classes_))


    def __getitem__(self, index):
        """Get example at specified index"""
        return self.features[index], self.labels[index]

    def __len__(self):
        """Get number of examples in dataset"""
        return len(self.features)


class TextDataSet(Dataset):
    le = LabelEncoder()
    def __init__(self, position: int=0, data: str="/home/biopythoncodepc/Documents/git_repositories/Recommendation-Ranking-System/data/tables/image_product.pkl", max_length: int=50):
        self.data = data
        if not os.path.exists(self.data):
            raise FileNotFoundError(f"The file {self.data} does not exist")
        data = pd.read_pickle(self.data)
        cats = self.le.fit_transform(
        data.category.str.split("/").apply(get_element, position))
        self.labels = torch.tensor(cats)
        self.descriptions = data.product_description.to_list()
        self.num_classes = len(set(self.labels))
        self.encoder = dict(
            zip(self.le.classes_, self.le.transform(self.le.classes_)))
        self.decoder = dict(enumerate(self.le.classes_))

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
        self.model.eval()
        self.max_length = max_length

    def __getitem__(self, index):
        label = self.labels[index]
        sentence = self.descriptions[index]
        encoded = self.tokenizer.batch_encode_plus([sentence], max_length=self.max_length, padding="max_length", truncation=True)
        encoded = {key:torch.LongTensor(value) for key, value in encoded.items()}
        with torch.no_grad():
            description = self.model(**encoded).last_hidden_state.swapaxes(1, 2)

        description = description.squeeze(0)

        return description, label


    def __len__(self):
        return len(self.labels)


class CombinedDataset(Dataset):
    le = LabelEncoder()
    def __init__(self, transformers: object=None, position: int=0, data: str="/home/biopythoncodepc/Documents/git_repositories/Recommendation-Ranking-System/data/tables/image_product.pkl", max_length: int=50):
        super().__init__()
        self.transformers = transformers
        self.position = position
        self.max_length = max_length
        self.data = data
        if not os.path.exists(self.data):
            raise FileNotFoundError(f"The file {self.data} does not exist")
        data = pd.read_pickle(self.data)
        features = []
        labels = []
        ind = 0
        IDs = data.id.tolist()
        cats = ImageDataset.le.fit_transform(
        data.category.str.split("/").apply(get_element, position=0))
        for ID in IDs:
            img_path = "data/cleaned_images/{}_resized.jpg".format(ID)
            with Image.open(img_path) as im:
                if transformers:
                    features.append(transformers(im))
                else:
                    features.append(
                        torchvision.transforms.functional.to_tensor(im))
            labels.append(torch.tensor(cats[ind]))
            ind += 1
        self.features = features
        self.labels = labels
        self.descriptions = data.product_description.to_list()
        self.num_classes = len(set(self.labels))
        self.encoder = dict(
            zip(self.le.inverse_transform(self.le.transform(self.le.classes_)), self.le.classes_))
        self.decoder = dict(enumerate(self.le.classes_))
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
        self.model.eval()
        self.max_length = max_length
    
if __name__ == "__main__":
    CombinedDataset()