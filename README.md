# Hybrid Federated Kernel Regularized Least Squares


## Overview
This repository contains the code used in: [[1]](#1). The project provides a Jupyter notebook that demonstrates how to use the algorithm **FedCG** in a simulated federated implementation, along with the necessary dependencies listed in the `requirements.txt` file. We also provide the code to carry out the performance tests. Moreover, in `EDM_tests/` we report the Matlab code used for the experiments of Euclidean Distance Matrice (EDM) reconstruction that we carried out to test the efficiency of FedCG in protecting the EDM of the dataset points for different cardinalities of Nystrom landmarks sets. 

## Files in the Repository
- **FedCG_usage.ipynb**: A Jupyter notebook demonstrating how to use the FedCG class and containing launching codes for the performance tests in the paper.
- **py_scripts/**: Directory containing the main Python scripts for the FedCG class.
  - **\_\_init\_\_.py**: Initialization file for the `py_scripts` module.
  - **dataset_handler.py**: Scripts for loading and preprocessing the datasets used in the paper.
  - **federated_functions.py**: Contains the main FedCG class with methods for training, prediction, and federated cycles. Defined methods are: 
      - **FedCG Class**: Manages the federated learning process, including initialization, training cycles, and prediction. 
      - **simulated_federated_cycles**: Simulates federated learning cycles across multiple hospitals/clients.
      - **initialize_alpha**: Initializes the alpha vector.
      - **initialize_fed_kernel**: Initializes the kernel matrix for a block (hospital/features) of data.
      - **build_Y_matrix**: Constructs a matrix with labels along the diagonal, repeated in blocks.
      - **feature_federated_gradient**: Cycles on omics centres to compute partial gradients.
      - **fit**: Trains the model.
      - **predict**: Makes predictions using the trained model.
      - **solve_linear**: Solves a linear system for kernel regression - use for testing.
  - **kernel.py**: Defines the Kernel class, which is used to construct various types of kernel matrices.
  - **plots_lambda_adapt_and_cen.py**: Script for plotting performance results.
  - **plots_parallel.py**: Script for plotting performance results, parallelised. 
  - **simulation.py**: Script for running Monte Carlo simulations for FedCG.
- **EMD_tests/**: Directory containing Matlab scripts and tests results for EDM completion experiments.
- **requirements.txt**: A list of Python dependencies required to run FedCG.
- **.gitignore**: Specifies files and directories to ignore in the Git repository.

## Getting Started with the FedCG class

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
   python3 -m venv venv
   source venv/bin/activate   # On Windows, use `venv\Scripts\activate`

3. **Install the Required Packages**:

    ```sh
   pip install -r requirements.txt

### Running the Notebook

1. Start Jupyter Notebook
    ```sh
    jupyter notebook

2. Open the Notebook:
Navigate to FedCG_usage.ipynb and run the cells. The notebook is composed of three sections:
   1. Explore datasets.
   2. Using the class.
   3. Simulations.


## EDM completion experiments
In `main.m` one can find the script used to test several EDM completion methods on different datasets, for different numbers of Nystrom landmarks, generated with different methods. Results obtained running this script on IIT HPC Franklin machine.
Results are saved in the three datasets directories (`Ionosphere/`, `Iris/`, `Sonar/`). The summary plots presented in the paper have been derived form these results with the script `plotMultiErrors.m`.
EDM completion methods `alternating_descent.m` and `rank_complete_edm.m` used in `EDM_tests/` are taken from [[2]](#2). Method `spectralReconstruction.m` is an implementation of Algorithm 1 from [[3]](#3).


## References
<a id="1">[1]</a> 
Damiani, C., Rodina, Y., Decherchi, S. (2024). 
A Hybrid Federated Kernel Regularized Least Squares Algorithm

<a id="2">[2]</a> 
Dokmanic, I., Parhizkar, R., Ranieri, J., Vetterli, M. (2015)
EDMbox. https://github.com/LCAV/edmbox.

<a id="3">[3]</a> 
Mazumder, R., Hastie, T., Tibshirani., R. (2010).
Spectral regularization algorithms for learning large incomplete matrices. 
Journal of Machine Learning Research, 11(80):2287–2322.