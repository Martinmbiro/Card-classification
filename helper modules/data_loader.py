import kagglehub as kh, zipfile, shutil, os, pathlib, torch, numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms.v2 as T

# Dataset Wrapper class
class CardsWrapper(Dataset):
  def __init__(self, path:pathlib.PosixPath, transform:T.Compose):
    self.d_set = ImageFolder(root=path, transform=transform)

  def __len__(self):
    return len(self.d_set)

  def __getitem__(self, index):
    img, lb = self.d_set[index]
    return img, lb

  @property
  def idx_to_class(self):
    mapper = {idx : cls for cls, idx in self.d_set.class_to_idx.items()}
    return mapper

  @property
  def classes(self):
    return self.d_set.classes

# define image transforms
_card_transforms = T.Compose([
    T.PILToTensor(),
    T.ToDtype(torch.float32, scale=True),
    T.Resize((128,128)),
    T.CenterCrop((128, 128))
])

# download folder
_DOWNLOAD_DIR = pathlib.Path('cards')
_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# function to return dataloaders for train, validation, test
def get_dataloaders() -> tuple[DataLoader, DataLoader, DataLoader, dict[int, str]]:
  """
    Downloads, extracts, and creates dataloaders for the cards image dataset.

    This function downloads the "cards-image-datasetclassification" dataset from Kaggle,
    archives it, extracts it to a local directory, and then creates PyTorch DataLoader
    objects for the training, testing, and validation sets.

    Returns
    -------
    tuple
        A tuple containing the following elements:
            - train_dl (DataLoader): DataLoader for the training set.
            - test_dl (DataLoader): DataLoader for the testing set.
            - val_dl (DataLoader): DataLoader for the validation set.
            - idx_to_class (dict[int, str]): A dictionary mapping class indices to class labels.

    Raises
    ------
    FileNotFoundError
        If the Kaggle dataset download fails or the zip file cannot be found.
    Exception
        If any other errors occur during dataset processing.

    Example
    -------
    >>> train_loader, test_loader, val_loader, class_mapping = get_dataloaders()
    >>> print(type(train_loader))
    <class 'torch.utils.data.dataloader.DataLoader'>
    >>> print(type(class_mapping))
    <class 'dict'>
  """
  # Download latest version of dataset from kaggle
  _cache = kh.dataset_download("gpiosenka/cards-image-datasetclassification")

  # archive the files
  shutil.make_archive(base_name=_DOWNLOAD_DIR/'data', format='zip', root_dir=_cache)

  # extract the files
  with zipfile.ZipFile(_DOWNLOAD_DIR/'data.zip', mode='r') as zipf:
    zipf.extractall(path=_DOWNLOAD_DIR)

  # create datasets first
  train_data = CardsWrapper(path='/content/cards/train', transform=_card_transforms)
  test_data = CardsWrapper(path='/content/cards/test', transform=_card_transforms)
  val_data = CardsWrapper(path='/content/cards/valid', transform=_card_transforms)

  # create dataloaders
  train_dl = DataLoader(
      dataset=train_data, batch_size=32, shuffle=True, pin_memory=True, num_workers=os.cpu_count())
  test_dl = DataLoader(
      dataset=test_data, batch_size=32, shuffle=True, pin_memory=True, num_workers=os.cpu_count())
  val_dl = DataLoader(
      dataset=val_data, batch_size=32, shuffle=True, pin_memory=True, num_workers=os.cpu_count())

  # return
  return train_dl, test_dl, val_dl, train_data.idx_to_class
