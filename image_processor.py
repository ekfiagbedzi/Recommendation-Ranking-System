import sys

from PIL import Image

from torchvision import transforms


transformers = transforms.Compose(
    [transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomResizedCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])]
)


def process_image(image_id, transformers: object=None):
    with Image.open("data/cleaned_images/{}_resized.jpg".format(image_id)) as im:
        if transformers:
            im = (transformers(im))
        else:
            im = transforms.functional.to_tensor(im)
    return im.unsqueeze(dim=0)


if __name__ == "__main__":
    (sys.argv[1], sys.argv[2])
