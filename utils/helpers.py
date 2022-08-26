import cv2
from numpy import array
from PIL import Image

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
    img = cv2.imread("/home/ubuntu/Recommendation-Ranking-System/cleaned_images/{}_resized.jpg".format(img_id)
)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img