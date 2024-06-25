import numpy as np
import matplotlib.pyplot as plt
from federated_functions import FedCG
from dataset_handler import load_ionosphere_dataset_validation
from kernel import Kernel
from tqdm import tqdm  # Import tqdm
from sklearn.metrics import accuracy_score
import multiprocessing

def run_simulation(params):
    lambda_val, nystrom_method, nystrom_points, X_train, y_train, X_test, y_test, num_runs = params
    accuracies = []

    for runseed in range(num_runs):
        W_train = load_ionosphere_dataset_validation(seed=runseed, nystr_pt=nystrom_points, nystr_method=nystrom_method)[-1]
        fed_CG_model = FedCG(kernel_params=kernel_params, nyst_points=nystrom_points, lam=lambda_val, Nb=10, toll=1e-3)
        fed_CG_model.fit(X_train, y_train, X_test, y_test, W=W_train, silent=True)
        y_pred_test_fedCG = fed_CG_model.predict(X_test)
        y_pred_test_fedCG_sign = np.sign(y_pred_test_fedCG)
        accuracies.append(accuracy_score(y_test, y_pred_test_fedCG_sign))

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    return mean_accuracy, std_accuracy


kernel_params = Kernel(kernel_type=1, kernel_sigma=1, poly_degree=None, binary=None)

# Load the dataset once
X_train, X_val, X_test, y_train, y_val, y_test, _ = load_ionosphere_dataset_validation(seed=0, nystr_method='uniform')

# Define the range of Nystrom point values
nystrom_points_range = range(10, 220, 10)  # Adjust the range as needed - 250

# Define the list of Nystrom methods
nystrom_methods = ['uniform', 'normal', 'subsampling']  

# Initialize a dictionary to store the results
results = {}

# Define the number of runs
num_runs = 30

# Lambda selection for RLS - CG model
min_lambda, max_lambda = -9, -1
lambda_values = np.logspace(min_lambda, max_lambda, num=10)
#lambda_values = [10**i for i in range(-9, 0)]

lambda_optimal_RLS_CG = None
max_accuracy_RLS_CG = -1

for lambda_val in lambda_values:
    RLS_CG_model = FedCG(kernel_params=kernel_params, nyst_points=X_train.shape[0], lam=lambda_val, Nb=1, toll=1e-3)
    RLS_CG_model.fit(X_train, y_train, X_val, y_val, W=X_train, silent=True)
    
    # Calculate predictions on the validation set
    y_pred_val_RLS_CG = RLS_CG_model.predict(X_val)
    
    # Ensure predictions are in -1 and 1 format
    y_pred_val_RLS_CG_sign = np.sign(y_pred_val_RLS_CG)
    
    # Calculate validation accuracy for RLS - CG model
    validation_accuracy_RLS_CG = accuracy_score(y_val, y_pred_val_RLS_CG_sign)

    if validation_accuracy_RLS_CG > max_accuracy_RLS_CG:
        max_accuracy_RLS_CG = validation_accuracy_RLS_CG
        lambda_optimal_RLS_CG = lambda_val

print(f"Selected Lambda for RLS_CG: {lambda_optimal_RLS_CG}")

# Loop over each Nystrom method
for nystrom_method in nystrom_methods:
    # Initialize lists to store mean accuracies and standard deviations for this method
    method_mean_accuracies = []
    method_std_accuracies = []

    pbar = tqdm(total=len(nystrom_points_range), desc='Nystrom Method: {}'.format(nystrom_method))

    for nystrom_points in nystrom_points_range:
        W_train = load_ionosphere_dataset_validation(seed=0, nystr_pt=nystrom_points, nystr_method=nystrom_method)[-1]

        # Perform lambda search for the current method and number of Nystrom points
        lambda_optimal_fedCG = None
        max_accuracy_fedCG = -1

        # Lambda selection for FedCG model
        for lambda_val in lambda_values:
            fed_CG_model = FedCG(kernel_params=kernel_params, nyst_points=nystrom_points, lam=lambda_val, Nb=10, toll=1e-3)
            fed_CG_model.fit(X_train, y_train, X_val, y_val, W=W_train, silent=True)

            # Calculate predictions on the validation set
            y_pred_val_fedCG = fed_CG_model.predict(X_val)
            
            # Ensure predictions are in -1 and 1 format
            y_pred_val_fedCG_sign = np.sign(y_pred_val_fedCG)

            # Calculate validation accuracy for FedCG model
            validation_accuracy_fedCG = accuracy_score(y_val, y_pred_val_fedCG_sign)

            if validation_accuracy_fedCG > max_accuracy_fedCG:
                max_accuracy_fedCG = validation_accuracy_fedCG
                lambda_optimal_fedCG = lambda_val

        # Perform Monte Carlo simulations for the optimal lambda value of FedCG model in parallel
        pool = multiprocessing.Pool(processes=num_runs)
        params = [(lambda_optimal_fedCG, nystrom_method, nystrom_points, X_train, y_train, X_test, y_test, num_runs)]
        results_list = pool.map(run_simulation, params)
        pool.close()
        pool.join()

        # Calculate mean and standard deviation of accuracies
        method_mean_accuracies.append(np.mean([res[0] for res in results_list]))
        method_std_accuracies.append(np.mean([res[1] for res in results_list]))

        # Update the progress bar
        pbar.update(1)

    # Close the progress bar
    pbar.close()

    # Store mean and standard deviation accuracies for this method
    results[nystrom_method] = {
        'mean_accuracies': method_mean_accuracies,
        'std_accuracies': method_std_accuracies
    }

# Perform simulations for RLS - CG model
RLS_CG_accuracies = []

# Perform multiple runs for RLS - CG model
for _ in range(num_runs):
    RLS_CG_model = FedCG(kernel_params=kernel_params, nyst_points=X_train.shape[0], lam=lambda_optimal_RLS_CG, Nb=1, toll=1e-3)
    RLS_CG_model.fit(X_train, y_train, X_test, y_test, W=X_train, silent=True)
    
    # Ensure predictions are in -1 and 1 format
    y_pred_test_RLS_CG = RLS_CG_model.predict(X_test)
    y_pred_test_RLS_CG_sign = np.sign(y_pred_test_RLS_CG)
    
    RLS_CG_accuracies.append(accuracy_score(y_test, y_pred_test_RLS_CG_sign))

# Calculate mean accuracy and standard deviation for RLS - CG model
RLS_CG_mean_accuracy = np.mean(RLS_CG_accuracies)
RLS_CG_std_accuracy = np.std(RLS_CG_accuracies)

# Plotting after the loop completes
for nystrom_method in nystrom_methods:
    plt.errorbar(nystrom_points_range, results[nystrom_method]['mean_accuracies'], yerr=results[nystrom_method]['std_accuracies'], marker='o', label= 'FedCG - {}'.format(nystrom_method))

# Plot RLS - CG model as a constant function
plt.errorbar(nystrom_points_range, [RLS_CG_mean_accuracy] * len(nystrom_points_range), yerr=[RLS_CG_std_accuracy] * len(nystrom_points_range), label='RLS - CG', linestyle='--')

plt.title('Mean Accuracy vs Nystrom Points\nDataset: Ionosphere, Method: {} (optimal lambda)'.format(nystrom_method))
plt.xlabel('Nystrom Points')
plt.ylabel('Mean Accuracy')
plt.grid(True)
plt.legend()
plt.savefig('mean_accuracy_vs_nystrom_points_par.png')  # Save the plot
plt.show()  # Display the plot
