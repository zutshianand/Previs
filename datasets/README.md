# Data sets and loaders

## Datasets for Data in csv files 
In order to use the tabular data set, we have to decide upon the following:
* The csv file which we need to use
* The pre-processing we have to do on each row of the csv file
* For traversing the tabular data for every item:
```python
from datasets.TabularDataset import TabularDataset

dataset = TabularDataset(first_file_path="file_path")
for i in range(len(dataset)):
    print(dataset[i])
```

* In case we want to transform the row before printing it or consuming it in our model. 
Or it can be the case that we want to pre-process the data set in some manner, we do the following:
```python
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.TabularDataset import TabularDataset
from processors.TabularProcessor import TabularProcessor

dataset = TabularDataset(first_file_path="/Users/anandzutshi/Desktop/RelationshipDataset.csv",
                            preprocess=transforms.Compose([
                                TabularProcessor(columns_to_encode=['col1','col2'], max_encoded_values=[2,1]),
                                transforms.ToTensor()
                            ]))
dataloader = DataLoader(dataset, 
                        batch_size=64,
                        shuffle=True, 
                        num_workers=4)

for i_batch, sample_batched in enumerate(dataloader):
    print(sample_batched)
```

## Datasets for Visual data in image file paths (.png)
In order for this transformation and data loading, we need to have the data in the form of images (.png) 
files in different folders. For the sake of simplicity, we would assume that the training images are in a different 
folder and the testing images are in a different folder.

We will now look at the different transformations which are available for both the images and 
ndarrays in pytorch comprehensively.

* **Compose** : Composes the different transformations together
* **ToTensor** : Converts a PIL image or ndarray into a tensor
* **ToPILImage** : Converts tensor or ndarray to PIL image
* **Normalize** : Normalises a tensor image with mean and standard deviation
* **Resize** : Resizes input PIL image to the given shape
* **CenterCrop** : Crops the given PIL image at the center
* **Pad** : Pad the given PIL Image on all sides with the given "pad" value
* **RandomOrder** : Apply a list of transformations in a random order
* **RandomCrop** : Crop the given PIL Image at a random location
* **RandomHorizontalFlip** : Horizontally flip the given PIL Image randomly with a given probability
* **RandomVerticalFlip** : Vertically flip the given PIL Image randomly with a given probability
* **RandomResizedCrop** : Crop the given PIL Image to random size and aspect ratio 
* **FiveCrop** : Crop the given PIL Image into four corners and the central crop
* **TenCrop** : Crop the given PIL Image into four corners and the central crop plus the flipped version of
    these (horizontal flipping is used by default)
* **ColorJitter** : Randomly change the brightness, contrast and saturation of an image
* **RandomRotation** : Rotate the image by angle
* **RandomAffine** : Random affine transformation of the image keeping center invariant
* **Grayscale** : Convert image to grayscale
* **RandomGrayscale** : Randomly convert image to grayscale with a probability of p (default 0.1)
* **RandomErasing** : Randomly selects a rectangle region in an image and erases its pixels

In case of loading an image dataset where the images are present in a directory, we use the ImageDataset defined 
below and then use the inbuilt DataLoader for the same

```python
from torchvision import transforms, models
from datasets.ImageDataset import ImageDataset
from torch.utils.data import DataLoader
from processors.ImageFeatureProcessor import ImageFeatureProcessor

image_transforms = transforms.Compose([
                                transforms.RandomRotation(30),
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                ImageFeatureProcessor(models.resnet18(pretrained=True), 224)])

transformed_image_data_set = ImageDataset("image_dir_path", classification_format=False, transform=image_transforms)
image_data_loader = DataLoader(transformed_image_data_set, batch_size=64, shuffle=True)
image_iter = iter(image_data_loader)

image = next(image_iter)
print(image)
```

In case the image dataset is present in a classification format where the images are present as follows:
        
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        .
        .
        .
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

In that case, we use the inbuilt ImageLoader along with the inbuilt transformations to load the dataset.
```python
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

data_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
hymenoptera_dataset = ImageFolder(root='dir_path', transform=data_transform)
dataset_loader = DataLoader(hymenoptera_dataset, batch_size=64, shuffle=True, num_workers=4)

image_iter = iter(dataset_loader)
image = next(image_iter)
print(image)
```

There can be a case when the dataset has both the csv and images together. Now we 
would look at how to handle that. There can be a case when we have to combine both the data in thw
csv together with the embeddings or features obtained from the images after some transformations.
In this case, the csv path is provided and for each csv data row, we have one image present in another
data directory.

```python
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.ImageTabularDataset import ImageTabularDataset
from processors.TabularProcessor import TabularProcessor

dataset = ImageTabularDataset(image_dir_path="dir_images",
                              tabular_dir_path="tabular_path",
                              classification_format=False,
                              image_transforms = transforms.Compose([
                                                    transforms.RandomSizedCrop(224),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                         std=[0.229, 0.224, 0.225])]),
                              tabular_transforms = transforms.Compose([
                                TabularProcessor(columns_to_encode=['col1','col2'], max_encoded_values=[2,1]),
                                transforms.ToTensor()]),
                              separator=',')
dataloader = DataLoader(dataset, 
                        batch_size=64,
                        shuffle=True, 
                        num_workers=4)

image_iter = iter(dataloader)
image, sample = next(image_iter)
print(image)
```

## Datasets for Textual data in textual file paths (.txt)

Since text is a sequential form of data, there can be two forms of problems associated to textual data.
We will be looking at both of them and the way of preprocessing them in this library. 

### Many to many modelling
These include problem statements where we are faced with modelling **many** (the textual data) to **many**
output data points. These many output data points are usually sentence tagging. We will be looking at how to
process data in sequence tagging problems.

```python
from dataloaders.MultiStreamDataLoader import MultiStreamDataLoader
from datasets.TextDataset import TextDataset

predict_list = [
    [0, 0, 1]
]
data_list = [
    ["I play ball"]
 ]
datasets = TextDataset.split_datasets(data_list, 3, 1, maintain_order=True, predict_list=predict_list)
loader = MultiStreamDataLoader(datasets)
for batch in loader:
    print(batch)
```

Please note that the *predict_list* and *data_list* needs to be of the above format for this to work.
In case the textual data is present in a dataframe, we need to read it from the dataframe and then
invoke the above methods. 

In case the textual data is present in separate text files, 
we need to incorporate the ```build_text_file_from_cls``` method to build a list of lists of the text files and
the corresponding tags. 

### Many to one modelling
These include problem statements where we are faced with modelling **many** (the textual data) to **one**
output data points. These one output data points are usually classification values (or classes). We will be looking at how to
process data in classification problems as well.

For the sake of simplification, we can consider the distribution of the textual data into different
classes as follows:
        
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        .
        .
        .
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
        
```python
from torchvision import transforms

from dataloaders.MultiStreamDataLoader import MultiStreamDataLoader
from datasets.TextDataset import TextDataset
from processors.TextProcessor import TextProcessor

from util.PreprocessingUtils import build_text_file_from_cls

datasets_for_classes = []
loaders_for_classes = []
classes = [1]
num_epochs = 10

for cls in classes:
    data_list_for_class = build_text_file_from_cls("<base_dir_path>", cls)
    datasets_for_class = TextDataset.split_datasets(data_list_for_class, 3, 1, transform=transforms.Compose([
             TextProcessor()
         ]))
    loaders_for_classes.append(MultiStreamDataLoader(datasets_for_class))


for epoch in range(num_epochs):
    for i in range(len(classes)):
        loader = loaders_for_classes[i]
        for batch in loader:
            print(batch)
            # train on model using this data bach
```

Once you do this, you will have the preprocessed data set. We stop here!
Post this, you can modify the output of the preprocessing and do your classification. 
We encourage to use [this](https://pytorch.org/hub/huggingface_pytorch-transformers/) for
help. You can also take a look at [this](https://medium.com/@aniruddha.choudhury94/part-2-bert-fine-tuning-tutorial-with-pytorch-for-text-classification-on-the-corpus-of-linguistic-18057ce330e1) Medium blog if you want to understand how to implement
BERT.
