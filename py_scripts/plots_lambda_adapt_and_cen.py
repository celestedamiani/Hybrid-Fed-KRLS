import numpy as np
import matplotlib.pyplot as plt
from federated_functions import FedCG
from dataset_handler import load_ionosphere_dataset_validation
from kernel import Kernel
from tqdm import tqdm  
from sklearn.metrics import accuracy_score

tol=1e-3

kernel_params = Kernel(kernel_type=1, kernel_sigma=1, poly_degree=None, binary=None)

# Load the dataset 
X_train, X_val, X_test, y_train, y_val, y_test, _ = load_ionosphere_dataset_validation(seed=0, nystr_method='uniform')

# Range Nystrom point values
nystrom_points_range = range(10, 220, 10)  # Adjust the range as needed - con val set fino a 220

# Nystrom methods
nystrom_methods = ['uniform', 'normal', 'subsampling']  

# Dictionary to save results
results = {}

num_runs = 30

# Lambda selection for RLS - CG model: due modi, scegli uno
min_lambda, max_lambda = -9, 9
lambda_values = np.logspace(min_lambda, max_lambda, num=20)
#lambda_values = [10**i for i in range(-9, 0)]
#lambda_values = [1]

lambda_optimal_RLS_CG = None
max_accuracy_RLS_CG = -1

for lambda_val in lambda_values:
    RLS_CG_model = FedCG(kernel_params=kernel_params, nyst_points=X_train.shape[0], lam=lambda_val, Nb=1, toll=tol)
    RLS_CG_model.fit(X_train, y_train, X_val, y_val, W=X_train, silent=True)
    
    # Calculate predictions on the validation set
    y_pred_val_RLS_CG = RLS_CG_model.predict(X_val)
    y_pred_val_RLS_CG_sign = np.sign(y_pred_val_RLS_CG)
    
    # Calculate validation accuracy for RLS - CG model
    validation_accuracy_RLS_CG = accuracy_score(y_val, y_pred_val_RLS_CG_sign)

    if validation_accuracy_RLS_CG > max_accuracy_RLS_CG:
        max_accuracy_RLS_CG = validation_accuracy_RLS_CG
        lambda_optimal_RLS_CG = lambda_val

print(f"Selected Lambda for RLS_CG: {lambda_optimal_RLS_CG}")

# Loop over Nystrom methods
for nystrom_method in nystrom_methods:

    method_mean_accuracies = []
    method_std_accuracies = []

    pbar = tqdm(total=len(nystrom_points_range), desc='Nystrom Method: {}'.format(nystrom_method))

    for nystrom_points in nystrom_points_range:
        #print(f"Method: {nystrom_method}, Nystrom Points: {nystrom_points}")
        
        W_train = load_ionosphere_dataset_validation(seed=0, nystr_pt=nystrom_points, nystr_method=nystrom_method)[-1]

        lambda_optimal_fedCG = None
        max_accuracy_fedCG = -1

        # Lambda selection for FedCG model
        for lambda_val in lambda_values:
            fed_CG_model = FedCG(kernel_params=kernel_params, nyst_points=nystrom_points, lam=lambda_val, Nb=10, toll=tol)
            fed_CG_model.fit(X_train, y_train, X_val, y_val, W=W_train, silent=True)

            # Calculate predictions on the validation set
            y_pred_val_fedCG = fed_CG_model.predict(X_val)
            y_pred_val_fedCG_sign = np.sign(y_pred_val_fedCG)

            # Calculate validation accuracy for FedCG model
            validation_accuracy_fedCG = accuracy_score(y_val, y_pred_val_fedCG_sign)

            if validation_accuracy_fedCG > max_accuracy_fedCG:
                max_accuracy_fedCG = validation_accuracy_fedCG
                lambda_optimal_fedCG = lambda_val

        print(f"Selected Lambda for FedCG: {lambda_optimal_fedCG}")

        accuracies = []
        # Perform runs for each Nystrom points value
        for run_seed in range(num_runs):
            # Generate W matrix with runseed
            if nystrom_method == 'uniform':
                W = np.random.rand(nystrom_points, X_train.shape[1])
            elif nystrom_method == 'normal':
                X_train_std = np.std(X_train, axis=0)
                X_train_mean = np.mean(X_train, axis=0)
                W = np.random.randn(nystrom_points, X_train.shape[1]) * X_train_std + X_train_mean
            elif nystrom_method == 'subsampling':
                idx = np.random.choice(X_train.shape[0], nystrom_points, replace=False)
                W = X_train[idx, :]
            else:
                raise ValueError("nystr_method should be 'uniform', 'normal', or 'subsampling'")
            
            # W_train = load_ionosphere_dataset_validation(seed=run_seed, nystr_pt=nystrom_points, nystr_method=nystrom_method)[-1]
            fed_CG_model = FedCG(kernel_params=kernel_params, nyst_points=nystrom_points, lam=lambda_optimal_fedCG, Nb=10, toll=tol)
            fed_CG_model.fit(X_train, y_train, X_test, y_test, W=W, silent=True)
            
            y_pred_test_fedCG = fed_CG_model.predict(X_test)
            y_pred_test_fedCG_sign = np.sign(y_pred_test_fedCG)
            
            
            y_pred_train_fedCG = fed_CG_model.predict(X_train)
            y_pred_train_fedCG_sign = np.sign(y_pred_train_fedCG)
            
            # print(f"Test Accuracy: {accuracy_score(y_test, y_pred_test_fedCG_sign)}")
            # print("y_test", y_test)
            # print("y_pred_test_fedCG_sign: ", y_pred_test_fedCG_sign)
            # print("alpha", fed_CG_model.alpha)
            
            # print(f"Train Accuracy: {accuracy_score(y_train, y_pred_train_fedCG_sign)}")
            # print("y_train", y_train)
            # print("y_pred_train_fedCG_sign: ", y_pred_train_fedCG_sign)
            

            accuracies.append(accuracy_score(y_test, y_pred_test_fedCG_sign))

        # Calculate mean accuracy and standard deviation for the current combi Nystrom methods - number points
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        method_mean_accuracies.append(mean_accuracy)
        method_std_accuracies.append(std_accuracy)
        
        pbar.update(1)

    pbar.close()

    # Store mean and standard deviation accuracies
    results[nystrom_method] = {
        'mean_accuracies': method_mean_accuracies,
        'std_accuracies': method_std_accuracies
    }

# Perform simulations for RLS-CG model
RLS_CG_accuracies = []

# Perform  runs for RLS-CG model
for _ in range(num_runs):
    RLS_CG_model = FedCG(kernel_params=kernel_params, nyst_points=X_train.shape[0], lam=lambda_optimal_RLS_CG, Nb=1, toll=tol)
    RLS_CG_model.fit(X_train, y_train, X_test, y_test, W=X_train, silent=True)
    
    y_pred_test_RLS_CG = RLS_CG_model.predict(X_test)
    y_pred_test_RLS_CG_sign = np.sign(y_pred_test_RLS_CG)
    
    RLS_CG_accuracies.append(accuracy_score(y_test, y_pred_test_RLS_CG_sign))

RLS_CG_mean_accuracy = np.mean(RLS_CG_accuracies)
RLS_CG_std_accuracy = np.std(RLS_CG_accuracies)

# Plot
for nystrom_method in nystrom_methods:
    plt.errorbar(nystrom_points_range, results[nystrom_method]['mean_accuracies'], yerr=results[nystrom_method]['std_accuracies'], marker='o', label= 'FedCG - {}'.format(nystrom_method))

# RLS-CG model as a constant function
plt.errorbar(nystrom_points_range, [RLS_CG_mean_accuracy] * len(nystrom_points_range), yerr=[RLS_CG_std_accuracy] * len(nystrom_points_range), label='RLS - CG', linestyle='--')

plt.title('Mean Accuracy vs Nystrom Points\nDataset: Ionosphere, Method: {} (optimal lambda)'.format(nystrom_method))
plt.xlabel('Nystrom Points')
plt.ylabel('Mean Accuracy')
plt.grid(True)
plt.legend()
plt.savefig('mean_accuracy_vs_nystrom_points_seq.png')  
plt.show()  
