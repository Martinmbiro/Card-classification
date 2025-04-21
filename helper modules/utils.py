import torch, pathlib, numpy as np, matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import Dataset, Subset
import torch, seaborn as sns, pandas as pd
from copy import deepcopy
from google.colab import files
from PIL import Image
import torchvision.transforms.v2 as T
from pathlib import Path
from itertools import chain
from matplotlib.gridspec import GridSpec

class EarlyStopping:
  """
  Early stopping to prevent overfitting.

  Attributes
  ----------
  counter : int
      Counter to track the number of epochs without improvement.
  patience : int
      Number of epochs to wait after the last best score.
  min_delta : float
      Minimum change in the monitored quantity to qualify as an improvement.
  score_type : str
      'loss' or 'metric', determines the direction of improvement.
  best_epoch : int
      Epoch with the best score.
  best_score : float
      Best score achieved so far.
  best_state_dict : dict
      State dictionary of the model at the best score.
  stop_early : bool
      Flag to indicate if early stopping should be triggered.
  """

  def __init__(self, score_type: str, min_delta: float = 0.0, patience: int = 5):
    """
    Initializes the EarlyStopping object.

    Parameters
    ----------
    score_type : str
        'loss' or 'metric', determines the direction of improvement.
    min_delta : float, optional
        Minimum change in the monitored quantity to qualify as an improvement. Defaults to 0.0.
    patience : int, optional
        Number of epochs to wait after the last best score. Defaults to 5.

    Raises
    ------
    Exception
        If score_type is not 'metric' or 'loss'.
    """
    self.counter = 0
    self.patience = patience
    self.min_delta = min_delta
    self.score_type = score_type
    self.best_epoch = None
    self.best_score = None
    self.best_state_dict = None
    self.stop_early = False

    if (self.score_type != 'metric') and (self.score_type != 'loss'):
        err_msg = 'score_type can only be "metric" or "loss"'
        raise Exception(err_msg)

  def __call__(self, model: torch.nn.Module, ep: int, ts_score: float):
    """
    Checks if early stopping should be triggered based on the current score.

    Parameters
    ----------
    model : torch.nn.Module
        The model being trained.
    ep : int
        The current epoch number.
    ts_score : float
        The current score (loss or metric).
    """
    if self.best_epoch is None:
        self.best_epoch = ep
        self.best_score = ts_score
        self.best_state_dict = deepcopy(model.state_dict())

    elif (self.best_score - ts_score >= self.min_delta) and (self.score_type == 'loss'):
        self.best_epoch = ep
        self.best_score = ts_score
        self.best_state_dict = deepcopy(model.state_dict())
        self.counter = 0

    elif (ts_score - self.best_score >= self.min_delta) and (self.score_type == 'metric'):
        self.best_epoch = ep
        self.best_score = ts_score
        self.best_state_dict = deepcopy(model.state_dict())
        self.counter = 0

    else:
        self.counter += 1
        if self.counter >= self.patience:
            self.stop_early = True

# function to plot train and test results
def plot_train_results(ep_list: list, train_score: list, test_score: list,
                       ylabel: str, title: str, best_epoch: int):
  """
  Plots training and test results against each other.

  Parameters
  ----------
  ep_list : list
      A list containing all epochs used in the optimization loop.
  train_score : list
      A list containing the training scores from the optimization loop.
  test_score : list
      A list containing the test scores from the optimization loop.
  ylabel : str
      Label for the y-axis of the plot.
  title : str
      Title for the plot.
  best_epoch : int
      Best epoch for which early stopping occurred.

  Returns
  -------
  None
  """
  f, ax = plt.subplots(figsize=(5, 3), layout='constrained')

  # train loss
  ax.plot(ep_list, train_score, label='Training',
          linewidth=1.7, color='#0047ab')

  # test loss
  ax.plot(ep_list, test_score, label='Validation',
          linewidth=1.7, color='#990000')
  # vertical line (for early stopping)
  if best_epoch is not None:
      ax.axvline(best_epoch, linestyle='--', color='#000000', linewidth=1.0,
                  label=f'Best ep ({best_epoch})')

  # axis, title
  ax.set_title(title, weight='black')
  ax.set_ylabel(ylabel)
  ax.set_xlabel('Epoch')
  ax.tick_params(axis='both', labelsize=9)
  plt.grid(color='#e5e4e2')

  # legend
  f.legend(fontsize=9, loc='upper right',
            bbox_to_anchor=(1.28, 0.93),
            fancybox=False)

  plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
  """
  Plots a confusion matrix for all classes.

  Parameters
  ----------
  y_true : np.ndarray
      An ndarray containing the true label values.
  y_pred : np.ndarray
      An ndarray containing the predicted label values.

  Returns
  -------
  None
  """
  # define figure and plot
  _, ax = plt.subplots(figsize=(8.0,8.0), layout='compressed')
  # plot
  ConfusionMatrixDisplay.from_predictions(
      y_true=y_true,
      y_pred=y_pred, cmap='Blues', colorbar=False, ax=ax)

  # set x and y labels
  ax.set_ylabel('True Labels', weight='black')
  ax.set_xlabel('Predicted Labels', weight='black',
                color='#dc143c')
  # set tick size and position
  ax.xaxis.tick_top()
  ax.xaxis.set_label_position('top')
  ax.tick_params(axis='both', labelsize=7)

  # change annotation font
  for txt in ax.texts:
      txt.set_fontsize(7.5)

  plt.show()

# function to save model to specified directory
def save_model(model: torch.nn.Module, path: pathlib.PosixPath):
    """
    Saves the model's state_dict to a specified path.

    Parameters
    ----------
    model : torch.nn.Module
        The model to save.
    path : pathlib.PosixPath
        The path where the model's state_dict will be saved.

    Returns
    -------
    None
    """
    torch.save(obj=model.cpu().state_dict(), f=path)
    print(f"MODEL'S state_dict SAVED TO: {path}")

# function to load model from a specified path
def load_model(model: torch.nn.Module, path: pathlib.PosixPath):
  """
  Loads the model's state_dict from a specified path.

  Parameters
  ----------
  model : torch.nn.Module
      A new object of the model class.
  path : pathlib.PosixPath
      Path pointing to a previously saved model's state_dict.

  Returns
  -------
  model : torch.nn.Module
      The model returned after loading the state_dict.
  """
  # overwrite state_dict
  model.load_state_dict(torch.load(f=path, weights_only=True))
# function to make inference on a single random image from testset
def make_single_inference(model: torch.nn.Module, dataset: torch.utils.data.Dataset,
                          label_map: dict, device: str):
  """
  Makes inference using a random data point from the test dataset.

  Parameters
  ----------
  model : torch.nn.Module
      A model (subclassing torch.nn.Module) to make inference.
  dataset : torch.utils.data.Dataset
      The Dataset to use for testing purposes.
  label_map : dict
      A dictionary mapping indices to labels (e.g., {0: 'O', 1: 'X'}).
  device : str
      Device on which to perform computation.

  Returns
  -------
  None
  """
  # get random image from test_set
  idx = np.random.choice(len(dataset))
  img, lb = dataset[idx]

  # make prediction
  with torch.inference_mode():
    model.to(device)  # move model to device
    model.eval()  # set eval mode
    lgts = model.to(device)(img.unsqueeze(0).to(device))
    pred = F.softmax(lgts, dim=1).argmax(dim=1)

  # print actual retrieved image
  plt.figure(figsize=(3.0, 3.0))
  # title with label
  if pred == lb:
    plt.title(
        f'Actual: {label_map[lb].title()}\nPred: {label_map[pred.item()].title()}',
        fontsize=8)
  else:  # if labels do not match, title = with red color
    plt.title(
        f'Actual: {label_map[lb].title()}\nPred: {label_map[pred.item()].title()}',
        fontsize=8, color='#de3163', weight='black')
  plt.axis(False)
  plt.imshow(img.permute(1,2,0))
  plt.show()

# function to make inference on 12 random images from testset
def make_multiple_inference(model: torch.nn.Module, dataset: torch.utils.data.Dataset,
                          label_map: dict, device: str):
  """
  Makes inference on multiple random images from the test dataset.

  Parameters
  ----------
  model : torch.nn.Module
      A model (subclassing torch.nn.Module) to make inference.
  dataset : torch.utils.data.Dataset
      The Dataset used for evaluation purposes.
  label_map : dict
      A dictionary mapping indices to labels (e.g., {0: 'O', 1: 'X'}).
  device : str
      Device on which to perform computation.

  Returns
  -------
  None
  """
  # get array of 12 random indices of images in test_dataset
  indices = np.random.choice(len(dataset), size=12, replace=False)
  # create subset from the 12 indices
  sub_set = Subset(dataset=dataset, indices=indices)

  # define a figure and subplots
  f, axs = plt.subplots(2, 6, figsize=(10, 8.5), layout='compressed')

  # move model to device & set eval mode
  model.to(device)
  model.eval()

  # loop through each subplot
  for i, ax in enumerate(axs.flat):
    img, lb = sub_set[i]  # return image and label

    # make inference on image returned
    with torch.inference_mode():
        lg = model(img.unsqueeze(0).to(device))
        pred = F.softmax(lg, dim=1).argmax(dim=1)

    ax.imshow(img.permute(1,2,0))
    ax.axis(False)
    if pred == lb:
        ax.set_title(
            f'Actual: {label_map[lb].title()}\nPred: {label_map[pred.item()].title()}',
            fontsize=7.5)
    else:  # if labels do not match, title = with red color
        ax.set_title(
            f'Actual: {label_map[lb].title()}\nPred: {label_map[pred.item()].title()}',
            fontsize=7.5, color='#de3163', weight='black')

  f.suptitle('Inference Made on 12 Random Test Images',
            # y=0.9,
            weight='black')
  plt.show()

# image transforms for inference
_img_transform = T.Compose(transforms=[
    T.PILToTensor(),
    T.ToDtype(dtype=torch.float32, scale=True),
    T.Resize((128,128)),
    T.CenterCrop((128, 128))
])

# image transform for plotting
_plot_transform = T.Compose([
    T.PILToTensor(),
    T.Resize((256, 256)),
    T.ToDtype(dtype=torch.float32, scale=True)
])

# function to upload an image for custom inference
def _upload_image() -> tuple[torch.Tensor, torch.Tensor]:
  """
    Handles image upload, validation, transformation, and cleanup.

    This function performs the following steps:
        - Ensures an upload directory exists.
        - Prompts the user to upload a single image file.
        - Validates that only one file is uploaded.
        - Applies image transformations for inference and plotting.
        - Deletes all uploaded images after processing.

    Returns
    -------
    tuple of torch.Tensor
        A tuple containing:
        - `inf_img`: The transformed image tensor used for inference.
        - `plot_img`: The transformed image tensor used for plotting or visualization.

    Raises
    ------
    Exception
        If more than one file is uploaded

    Notes
    -----
    Supported image formats include: .png, .jpg, .jpeg, and .gif.
    This function is intended to be used in a Colab notebook environment using `google.colab.files.upload`.
  """
  # create folder to upload, if not already there
  UPLOAD_DIR = Path('uploads')
  UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

  # upload the file to the directory above
  upd = files.upload(target_dir=UPLOAD_DIR)

  # get all uploaded images
  iter_gen = chain(UPLOAD_DIR.glob('*.png'),
                    UPLOAD_DIR.glob('*.jpg'),
                    UPLOAD_DIR.glob('*.jpeg'),
                    UPLOAD_DIR.glob('*.gif'))

  if len(upd) != 1:
    # delete all uploaded images
    for img in iter_gen:
      img.unlink(missing_ok=True)
    # raise Exception
    raise Exception(f'Expected ONE image, but got {len(upd)}.\nRE-RUN the cell and upload a single image.')

  else:
    # get image path
    img_path = Path(next(iter(upd)))

    # turn image into tensor for inference
    inf_img = _img_transform(Image.open(img_path))
    # for plotting
    plot_img = _plot_transform(Image.open(img_path))

    # delete all uploads
    for img in iter_gen:
      img.unlink(missing_ok=True)

    return inf_img, plot_img

# function to perform infernce on a single image and display results
def custom_inference(model:torch.nn.Module, device:str, label_mapper:dict):
  """
    Performs inference on a single uploaded image, displays the prediction, and visualizes the top-5 class probabilities.

    This function:
        * Uploads a single image and applies necessary transformations.
        * Runs inference using the provided model and device.
        * Computes class probabilities using softmax.
        * Displays the top predicted label and its probability.
        * Creates and displays a bar plot of the top-5 predicted classes with their probabilities.

    Parameters
    ----------
    model : torch.nn.Module
        A trained PyTorch model for image classification.

    device : str
        The device to perform inference on (e.g., 'cpu' or 'cuda').

    label_mapper : dict
        A dictionary mapping class indices (int) to human-readable class labels (str).

    Raises
    ------
    Exception
        If more than one image is uploaded

    Notes
    -----
    - The image must be uploaded using an interactive Colab file upload widget.
    - The `_upload_image()` helper is responsible for transformation and validation.
    - Torch is used in inference mode to avoid tracking gradients.
    - This function assumes the model outputs raw logits for classification.

    Examples
    --------
    >>> label_map = {0: 'cat', 1: 'dog', 2: 'car'}
    >>> custom_inference(trained_model, device='cuda', label_mapper=label_map)
  """
  # get tensors from uploaded image
  inf_img, plot_img = _upload_image()

  #_____MAKE INFERENCE_____
  with torch.inference_mode():
    model.to(device)
    # make prediction
    logits = model(inf_img.unsqueeze(0).to(device))
    # scale logits on 0 -> scale
    logits = F.softmax(logits, dim=1)

    # label & class probability
    y_pred = logits.argmax(dim=1).item()
    class_prob = logits.max().item()

    #_____PROBABILITY THRESHOLD (Out Of Distribution Detection)_____
    if class_prob < 0.85:
      err_msg = 'The image uploaded is most likely NOT of a valid PLAYING CARD\n\nRE-RUN THE CODE CELL AND UPLOAD A VALID PLAYING CARD IMAGE'
      raise Exception(err_msg)

    # get top5 indices & values from logits
    top5 = logits.topk(5)

  #_____MAKE DATAFRAME_____
  # make a dataframe out of the top5 predictions & probabilities
  df = pd.DataFrame({
      'Classes': [label_mapper[c.item()].title() for c in top5.indices.squeeze()],
      'Probabilities': top5.values.squeeze().cpu().numpy()
  })
  # sort dataframe in decsending order
  df.sort_values(by='Probabilities', ascending=False, inplace=True)

  #_____PLOT_____
  # set up figure, gridspec, axes
  f = plt.figure(figsize=(9, 3), layout='compressed')
  gs = GridSpec(figure=f, nrows=1, ncols=2, width_ratios=[1,3], wspace=0.05)
  ax1 = f.add_subplot(gs[0])
  ax2 = f.add_subplot(gs[1])

  # plot the card
  ax1.set_title(f'Predicted: {label_mapper[y_pred].title()}\nProbability: {class_prob:.2f}',
            fontsize=8.5, weight='black', color='#de3163')
  ax1.axis(False)
  ax1.imshow(plot_img.permute(1,2,0).clamp(min=0, max=1.0))

  # bar plot
  ax2.set_title('Top 5 Prediction Probabilities', weight='black', fontsize=10,
                color='#1f305e')
  ax2.set_ylabel('Classes', weight='black', fontsize=8.5)
  ax2.set_xlabel('Probabilities', weight='black', fontsize=8.5)
  ax2.tick_params(axis='both', labelsize=8)
  ax2.grid(axis='x', color='#dbd7d2')
  sns.barplot(
      data=df,
      x='Probabilities', y='Classes', hue='Classes', orient='h', palette='crest',
      legend=False, width=0.8, ax=ax2)
  # display
  plt.show()
