# Card Classification

<p align="center">
  <img src='pics/cards.png'  width='500'/>
</p>

Hello again 👋
+ [Transfer learning](https://www.datacamp.com/blog/what-is-transfer-learning-in-ai-an-introductory-guide) is a machine learning technique where a model trained on one task is repurposed as the starting point for a model on a new task. This approach significantly reduces the time & computational resources needed to train a model from scratch, while also improving a model's generalization abilities.
+ In this repository, I leverage the power of transfer learning in training a computer vision model to classify images of playing cards, using the _card classification dataset_ linked [`here`](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification). The training set consists of `7,624` images, while the test and validation sets consist of `265` images each.
+ The model architecture used for this exercise is [resnet14t](https://huggingface.co/timm/resnet14t.c3_in1k), from the [`timm`](https://huggingface.co/docs/timm/en/index) library. The pre-trained model is fine-tuned for our task-specific problem by altering the `classifier` layer and subsequent retraining.
+ The first notebook, `01 Cards modular.ipynb` entails turning reusable code into modules, while the second notebook, `02 Cards end to end.ipynb` is where I put everything together.
+ Comments, working code, and links to the latest official documentation are included every step of the way. There's links to open each notebook (_labeled 01...02_) in Google Colab - feel free to play around with the code.

## Milestones 🏁
**Concepts covered in this exercise include:**  
1. [x] Transfer learning
2. [x] Training and evaluating a multi-class image classification model - built using [`PyTorch`](https://pytorch.org/)
3. [x] Regularization using [Early stopping](https://www.linkedin.com/advice/1/what-benefits-drawbacks-early-stopping#:~:text=Early%20stopping%20is%20a%20form,to%20increase%20or%20stops%20improving.)
4. [x] [Modular programming](https://en.wikipedia.org/wiki/Modular_programming)
5. [x] Data visualization

## Tools ⚒️
1. [`Google Colab`](https://colab.google/) - A hosted _Jupyter Notebook_ service by Google.
2. [`PyTorch`](https://pytorch.org/) -  An open-source machine learning (ML) framework based on the Python programming language that is used for building **Deep Learning models**
3. [`scikit-learn`](https://scikit-learn.org/stable/#) - A free, open-source library that offers Machine Learning tools for the Python programming language
4. [`numpy`](https://numpy.org/) - The fundamental package for scientific computing with Python
5. [`matplotlib`](https://matplotlib.org/) - A comprehensive library for making static, animated, and interactive visualizations in Python
6. [`seaborn`](https://seaborn.pydata.org/index.html) - A python data visualization library based on `matplotlib` that provides a high-level interface for drawing attractive and informative statistical graphics
7. [`requests`](https://requests.readthedocs.io/en/latest/) - An elegant and simple HTTP library for Python
8. [`torchinfo`](https://github.com/TylerYep/torchinfo) - A library for viewing model summaries in PyTorch
9. [`PIL`](https://pillow.readthedocs.io/) - A Python library for image manipulation and basic image processing tasks
10. [`kagglehub`](https://github.com/Kaggle/kagglehub) - A library that provides a simple way to interact with [Kaggle](https://www.kaggle.com/) resources such as datasets, models, notebook outputs in Python
11. [`timm`](https://huggingface.co/docs/timm/en/index) - An open-source Python library that provides SOTA deep learning models for computer vision tasks

## Results 📈
> On a scale of `0` -> `1`, the final best-performing model achieved:
+ A weighted `precision`, `recall`, and `f1_score` of `0.97`
+ An overall model accuracy of `0.9698`
+ An overall `roc_auc_score` of `1.000`

> The saved model's `state_dict` can be found in the drive folder linked [here](https://drive.google.com/file/d/1zjzTKhPjbjAHmy2myER4zcGnSWQUILZa/view?usp=drive_link)


## Reference 📚
+ Thanks to the insight gained from [`Daniel Bourke`](https://x.com/mrdbourke?s=21&t=1Fg4dWHIo5p7EaMHhv2rng), [`Modern Computer Vision with Pytorch, 2nd Edition`](https://www.packtpub.com/en-us/product/modern-computer-vision-with-pytorch-9781803240930) and [`Rob Mulla`](https://www.youtube.com/watch?v=tHL5STNJKag)
+ Not forgetting these gorgeous gorgeous [`emojis`](https://gist.github.com/FlyteWizard/468c0a0a6c854ed5780a32deb73d457f) 😻

> _Dataset by [kaggle](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification)_ ♥  
> _Illustration by [`Storyset`](https://storyset.com)_ ♥

