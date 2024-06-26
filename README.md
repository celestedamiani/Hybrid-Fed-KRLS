# Hybrid Federated Kernel Regularized Least Squares


## Overview
This repository contains the code used in: [[1]](#1). The project provides a Jupyter notebook that demonstrates how to use **FedCG** in a simulated federated implementation, along with the necessary dependencies listed in the `requirements.txt` file. We also provide the code to carry out the performance tests. Moreover, `EDM_tests/` we report the Matlab code used for the experiments of Euclidean Distance Matrice (EDM) reconstruction that we carried out to test the efficiency of FedCG in protecting the EDM of the dataset points for different cardinalities of Nystrom landmarks sets. 

## Files in the Repository
- **FedCG_usage.ipynb**: A Jupyter notebook demonstrating how to use FedCG and containing launching codes for the tests in the paper.
- **requirements.txt**: A list of Python dependencies required to run FedCG.
- **.gitignore**: specifies files and directories to ignore in the Git repository.
- **EMD_tests/**: Directory containing scripts and tests results for EDM reconstruction experiments.
- **py_scripts/**: Directory containing the main Python scripts for the FedCG class.
  - **\_\_init\_\_.py**: Initialization file for the `py_scripts` module.
  - **dataset_handler.py**: Script for loading and preprocessing the datasets used in the paper.
   - **federated_functions.py**: Contains the main FedCG class with methods for training, prediction, and federated cycles. In particular: 
      - **FedCG Class**: Manages the federated learning process, including initialization, training cycles, and prediction. 
      - **simulated_federated_cycles**: Simulates federated learning cycles across multiple batches.
      - **initialize_alpha**: Initializes the alpha vector.
      - **initialize_fed_kernel**: Initializes the kernel matrix for a batch of data.
      - **build_Y_matrix**: Constructs a matrix with labels along the diagonal, repeated in blocks.
      - **feature_federated_gradient**: Computes the gradient for a batch of data.
      - **fit**: Trains the model using the federated learning process.
      - **predict**: Makes predictions using the trained model.
      - **solve_linear**: Solves a linear system for kernel regression.
  - **kernel.py**: Defines the Kernel class, which is used to construct various types of kernel matrices.
  - **plots_lambda_adapt_and_cen.py**: Script for plotting performance results.
  - **plots_parallel.py**: Script for plotting performance results, parallelised. 
  - **simulation.py**: Script for running Monte Carlo simulations for FedCG.


## Getting Started

### Prerequisites
Ensure you have the following installed:
- Python 3.8 or higher
- Jupyter Notebook

### Installation

1. **Clone the Repository**:
   ```sh
   git clone https://gitlab.iit.it/hpc/hybrid-fed-krls.git
   cd hybrid-fed-krls

2. **Create a virtual environment (optional)**:
    ```sh
   python -m venv venv
   source venv/bin/activate

3. **Install the Required Packages**:

    ```sh
   pip install -r requirements.txt

### Running the Notebook

1. Start Jupyter Notebook
    ```sh
    jupyter notebook

2. Open the Notebook:
Navigate to FedCG_usage.ipynb and run the cells. The notebook is composed of three sections:
   1. Explore datasets
   2. Using the class 
   3. Simulations




## References
<a id="1">[1]</a> 
Damiani, C., Rodina, Y., Decherchi, S. (2024). 
A Hybrid Federated Kernel Regularized Least Squares Algorithm
In:
[![DOI:](https://)](https://doi.org/)