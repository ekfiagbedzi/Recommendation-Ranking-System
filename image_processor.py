from utils.helpers import image_processor

from PIL import Image

from torchvision import transforms


transformers=transforms.Compose([transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomResizedCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

img = image_processor("/home/biopythoncodepc/Documents/git_repositories/Recommendation-Ranking-System/data/cleaned_images/0a1baaa8-4556-4e07-a486-599c05cce76c_resized.jpg",
    transformers=transformers)

print(img.shape)

