# Beyond Color: Analyzing the Robustness of Contrastive Learning under Structural Augmentations

**Course:** ECE 570 - Course Project  
**Track:** Track 3 (Research)  
**Author:** Henry Izere

## 1. Project Overview
This project reproduces the SimCLR Contrastive Learning framework to evaluate the role of color in self-supervised representation learning. It introduces a novel structural augmentation pipeline using Sobel Edge Filtering to test if a ResNet-18 model can learn robust geometric features when all color and texture data are mathematically stripped from the training set.

## 2. Dependencies
This project was built and tested in Google Colab. It requires a GPU runtime (e.g., T4 GPU) to train in a reasonable timeframe. The following Python packages are required:
* `torch`
* `torchvision`
* `matplotlib`
* `Pillow` (PIL)

*(Note: If running in Google Colab, these dependencies are pre-installed).*

## 3. Dataset
The project uses the **CIFAR-10** dataset. 
* **Automatic Download:** You do not need to download the dataset manually. The code utilizes `torchvision.datasets.CIFAR10(..., download=True)`, which will automatically fetch and extract the dataset into a local `./data` folder upon execution.

## 4. Instructions to Run the Code
The entire project is contained within a single Jupyter Notebook: `BeyondColor.ipynb`.

1. Upload `BeyondColor.ipynb` to Google Colab (or open it in a local Jupyter environment with GPU support).
2. Go to **Runtime > Change runtime type** and select **T4 GPU** (or equivalent hardware accelerator).
3. Select **Runtime > Run all**.
4. The notebook will sequentially:
   - Download the CIFAR-10 dataset.
   - Define the custom transformations and architectures.
   - Run **Experiment 1** (Color Baseline): Trains for 50 epochs and evaluates.
   - Run **Experiment 2** (Initial Edge Twist): Trains for 50 epochs and evaluates.
   - Run **Experiment 3** (Optimized Edge Twist): Trains for 150 epochs with $\tau=0.1$ and evaluates.
5. Total execution time on a single T4 GPU is approximately 2.5 to 3 hours.

## 5. Code Structure
The notebook is structured into logical blocks:
* **Block 1-3:** Data loading and custom augmentation pipelines (`SimCLR_Twin_Transform`, `SobelEdgeTransform`).
* **Block 4:** Base architecture (`SimCLR_ResNet`) and the "CIFAR Hack" to preserve spatial dimensions.
* **Block 5:** The Master Training Loop (`train_simclr`) and the `NTXentLoss` function.
* **Block 6:** The Linear Evaluation protocol (`LinearEvaluation`, `train_and_test_linear`).
* **Block 7-9:** The execution blocks for Experiments 1, 2, and 3.

## 6. Code Attribution and Editing
In accordance with the academic integrity and project rubric guidelines, the code is attributed as follows:

* **Written Entirely by Me (with conceptual logic refinement via AI assistant):**
  * The custom `SobelEdgeTransform` class utilizing PIL filters.
  * The custom data augmentation pipelines (`Color_Twin_Transform`, `Edge_Twin_Transform`, and `Optimized_Edge_Twin_Transform`).
  * The logic for the linear evaluation training loop (`train_and_test_linear`).
  * The experimental execution blocks combining the data loaders, model initialization, and training calls.

* **Adapted from Prior Code / External Knowledge:**
  * **ResNet-18 Architecture:** Imported from standard `torchvision.models.resnet18`. 
    * *Edits:* I modified the initialization block of this model (specifically the `__init__` function of `SimCLR_ResNet`) to replace the first `7x7` convolution with a `3x3` convolution and removed the `MaxPool` layer to accommodate 32x32 CIFAR images.
  * **NT-Xent Loss Function:** The mathematical implementation of the contrastive loss function (`class NTXentLoss`) was adapted from standard open-source SimCLR PyTorch tutorials to fit this specific batch size and temperature scaling.
  * **Cosine Annealing Scheduler:** Utilized standard `torch.optim.lr_scheduler`.

* **Copied from External Repositories:**
  * No entire files or repositories were cloned. All code was written or adapted directly into the notebook cells.
