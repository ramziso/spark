# original code is from https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
import torch
from torchvision import datasets, transforms

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    Warning! torch.utils.data.DataLoader try to convert any of the ImageFolder result to
    tensor, when it used with default_collate argument. This class must be used with
    dataloader batch size = 1. otherwise it will collapse.
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


"""
# instantiate the dataset and dataloader
data_dir = "/media/kimjonghyuk/ROBOKEN-KIM-HDD/DATASET/etc/dogs_and_cats_Redux_small/train"
transform = transforms.ToTensor()
dataset = ImageFolderWithPaths(data_dir, transform=transform)  # our custom dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size = 1) # 

# iterate over data
for inputs, labels, paths in dataloader:
    # use the above variables freely
    print(inputs, labels, paths)
    print(inputs.shape, labels.shape, type(paths))
    break
"""