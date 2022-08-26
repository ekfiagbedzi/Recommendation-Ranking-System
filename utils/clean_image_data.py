from logging import raiseExceptions
from PIL import Image
from utils import resize_image
import os
import fnmatch


def clean_image_data(path, final_size, save_path):
    dirs = fnmatch.filter(os.listdir(path), "*.jpg")
    for item in dirs:
        im = Image.open(path + item)
        new_im = resize_image(final_size, im)
        item = item.rstrip(".jpg")
        new_im.save(f"{save_path}/{item}_resized.jpg")
    print("Images Cleaned Succesfully. Have a Nice Day!!!")
    

path = "/home/ubuntu/images/"
save_path = "/home/ubuntu/Recommendation-Ranking-System/cleaned_images/"
final_size = 28
clean_image_data(path, final_size, save_path)