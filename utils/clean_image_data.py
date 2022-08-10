from PIL import Image
import os


def resize_image(final_size, im):
    size = im.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
    im = im.resize(new_image_size)
    new_im = Image.new("RGB", (final_size, final_size))
    new_im.paste(
        im,
        (0, 0))
    return new_im

if __name__ == "__main__":
    path = "/home/ubuntu/images/"
    dirs = os.listdir(path)
    final_size = 512
    for n, item in enumerate(dirs[:5], 1):
        im = Image.open(path + item)
        new_im = resize_image(final_size, im)
        new_im.save(f"{n}_resized.jpg")


