from PIL import Image
from numpy import array

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
    img = Image.open(
        "/home/ubuntu/Recommendation-Ranking-System/cleaned_images/{}_resized.jpg".format(img_id)
        )
    img_arr = asarray(img)
    return img_arr

# %%
from PIL import Image
from numpy import asarray

#img = Image.open("/home/ubuntu/Recommendation-Ranking-System/cleaned_images/7cdb36be-888e-4dc8-81ed-0b98e9e6a1c9_resized.jpg")
#asarray(img)

def image_to_array(img_id):
    img = Image.open(
        "/home/ubuntu/Recommendation-Ranking-System/cleaned_images/{}_resized.jpg".format(img_id)
        )
    return asarray(img)

image_to_array("7cdb36be-888e-4dc8-81ed-0b98e9e6a1c9")


# %%
