import timm, torch

def get_model(device:str) -> tuple[torch.nn.Module, torch.optim.Optimizer, torch.nn.Module]:
  """
    Creates and initializes a model, optimizer, and loss function for training.

    Parameters
    ----------
    device : str
        The device to which the model should be moved. Common values include 'cuda' for GPU or 'cpu' for CPU.

    Returns
    -------
    tuple
        A tuple containing three elements:
        - model (torch.nn.Module): The model initialized with the EfficientNet-B0 architecture for 53 classes.
        - opt (torch.optim.Optimizer): The AdamW optimizer initialized with the model's parameters.
        - loss_fn (torch.nn.Module): The CrossEntropyLoss function used for multi-class classification.
    """
  # pretrained model, with 53 classes for output layer
  model = timm.create_model(
      model_name='resnet14t',
      num_classes=53,
      pretrained=True).to(device)

  # optimizer
  opt = torch.optim.Adam(params=model.parameters(), lr=0.001)

  # loss function
  loss_fn = torch.nn.CrossEntropyLoss()

  return model, opt, loss_fn
