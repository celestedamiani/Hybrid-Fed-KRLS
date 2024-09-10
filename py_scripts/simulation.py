import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
from kernel import Kernel  # Make sure to import your Kernel class
from federated_functions import FedCG  # Make sure to import your FedCG class
from dataset_handler import * 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from scipy.interpolate import griddata



def model_simulation(dataset_name, num_runs, X_train, y_train, X_test, y_test, W, kernel_params, nyst_points, lam, Nb, toll):
    '''
    Args:
        dataset_name: Name of the dataset, can be 'Iris', 'Sonar', 'Ionosphere', 'Generated', 'BreastCancer', 
                    'Wine' followed by '_' and the nystrom method. Only for file naming purposes
        num_runs: Number of Monte Carlo simulations
        X_train: Training data
        y_train: Training labels
        X_test: Test data
        y_test: Test labels
        W: Nystrom points matrix
        kernel_params: instantiation of Kernel class
        nyst_points: Number of Nyström points
        lam: Regularization parameter
        Nb: Number of hospitals/clients
        toll: Tolerance for the stopping criterion
    Output:
        Saves the results of the Monte Carlo simulations to CSV files
    '''

    # Initialize dataframes to store results
    fedcg_results_df = pd.DataFrame(columns=['Run', 'Seed', 'Central test accuracy', 'Train time', 'Train StoppingEpoch',
                                            'Test accuracy', 'Test Precision', 'Test Recall', 'Test F1 Score',
                                            'Test MSE', 'Test MAE'])
    cencg_results_df = pd.DataFrame(columns=['Run', 'Seed', 'Central test accuracy', 'Train time', 'Train StoppingEpoch',
                                            'Test accuracy', 'Test Precision', 'Test Recall', 'Test F1 Score',
                                            'Test MSE', 'Test MAE'])

    # Monte Carlo simulation loop
    for run in range(num_runs):
        
        # print(f'\n Run {run + 1} of {num_runs} \n')
        # Set a unique seed for each run
        
        np.random.seed(run)

        # Initialize FedCG model
        fedcg_model = FedCG(kernel_params, nyst_points, lam, Nb, toll)

        # Initialize Centralized CG model
        cencg_model = FedCG(kernel_params, nyst_points, lam, 1, toll)

        # FedCG run
        localResidue, cent_test_accuracy, test_accuracy, elapsed, alpha = fedcg_model.fit(X_train, y_train, X_test, y_test, W=W, alpha_init=None, silent=True)

        # Predictions on the test set
        predictions = fedcg_model.predict(X_test)
        #print('predictions: ', predictions)

        # Calculate various metrics
        #accuracy = accuracy_score(y_test, np.sign(predictions), normalize=True)
        test_precision = precision_score(y_test, np.sign(predictions), zero_division=0)
        test_recall = recall_score(y_test, np.sign(predictions), zero_division=0)
        test_f1 = f1_score(y_test, np.sign(predictions), zero_division=0)
        test_mse = mean_squared_error(y_test, predictions)
        test_mae = mean_absolute_error(y_test, predictions)
        train_stopping_epoch = len(localResidue) - 1 

        # Append individual metrics to fedcg_results_df
        fedcg_results_df = pd.concat([fedcg_results_df, pd.DataFrame({'Run': run + 1, 'Seed': run, 'Central test accuracy': cent_test_accuracy, 
                                                                   'Train time': elapsed, 'Train StoppingEpoch': train_stopping_epoch,
                                                                   'Test accuracy': test_accuracy,
                                                                   'Test Precision': test_precision, 'Test Recall': test_recall, 'Test F1 Score': test_f1,
                                                                   'Test MSE': test_mse, 'Test MAE': test_mae}, index=[0])], ignore_index=True)


        # Centralized CG run
        central_localResidue,  cent_test_accuracy, central_test_accuracy, central_elapsed, central_alpha = cencg_model.fit(X_train, y_train, X_test, y_test, W=W, alpha_init=None, silent=True)

        # Predictions on the test set
        central_predictions = cencg_model.predict(X_test)

        # Calculate various metrics
        #central_accuracy = accuracy_score(y_test, np.sign(central_predictions), normalize=True)
        central_precision = precision_score(y_test, np.sign(central_predictions))
        central_recall = recall_score(y_test, np.sign(central_predictions))
        central_f1 = f1_score(y_test, np.sign(central_predictions))
        central_mse = mean_squared_error(y_test, central_predictions)
        central_mae = mean_absolute_error(y_test, central_predictions)
        central_stopping_epoch = len(central_localResidue) - 1  # Assuming len(central_localResidue) represents the number of epochs

        # Append individual metrics to cencg_results_df
        cencg_results_df = pd.concat([cencg_results_df, pd.DataFrame({'Run': run + 1, 'Seed': run, 'Cantral test accuracy': cent_test_accuracy,
                                                                   'Train time': central_elapsed, 'Train StoppingEpoch': central_stopping_epoch,
                                                                   'Test accuracy': central_test_accuracy, 'Test Precision': central_precision,
                                                                   'Test Recall': central_recall, 'Test F1 Score': central_f1,
                                                                   'Test MSE': central_mse, 'Test MAE': central_mae}, index=[0])], ignore_index=True)

    # Save dataframes to CSV with dataset_name in the file name
    fedcg_results_df.to_csv(f'fedcg_results_{dataset_name}.csv', index=False)
    cencg_results_df.to_csv(f'cencg_results_{dataset_name}.csv', index=False)




def nys_simulation(dataset_name, num_runs, X_train, y_train, X_test, y_test, nyst_method, kernel_params, nyst_points, lam, Nb, toll):
    '''
    Args:
        dataset_name: Name of the dataset, can be 'Iris', 'Sonar', 'Ionosphere', 'Generated', 'BreastCancer', 'Wine'
        num_runs: Number of Monte Carlo simulations
        X_train: Training data
        y_train: Training labels
        X_test: Test data
        y_test: Test labels
        nyst_method: Method for Nyström approximation, can be 'uniform' or 'normal' or 'subsampling'
        kernel_params: instantiation of Kernel class
        nyst_points: Number of Nyström points
        lam: Regularization parameter
        Nb: Number of hospitals/clients
        toll: Tolerance for the stopping criterion
    Output: 
        Saves the results of the Monte Carlo simulations to CSV files
    '''

    # Initialize FedCG model
    fedcg_model = FedCG(kernel_params, nyst_points, lam, Nb, toll)

    # Initialize Centralized CG model
    cencg_model = FedCG(kernel_params, nyst_points, lam, 1, toll)
    
    alpha_init = fedcg_model.initialize_alpha()

    
    # Initialize dataframes to store results
    fedcg_results_df = pd.DataFrame(columns=['Run', 'Seed', 'Train accuracy', 'Train time', 'Train StoppingEpoch',
                                            'Test accuracy', 'Test Precision', 'Test Recall', 'Test F1 Score',
                                            'Test MSE', 'Test MAE'])
    cencg_results_df = pd.DataFrame(columns=['Run', 'Seed', 'Train accuracy', 'Train time', 'Train StoppingEpoch',
                                            'Test accuracy', 'Test Precision', 'Test Recall', 'Test F1 Score',
                                            'Test MSE', 'Test MAE'])

    # Monte Carlo simulation loop
    for run in range(num_runs):
        # Set a unique seed for each run
        seedrun = np.random.seed(run)
        
        if dataset_name == 'Iris':            
            W = load_iris_dataset(seed=seedrun, nystr_pt=nyst_points, nystr_method=nyst_method)[-1]  # Get the W matrix from the last element of the tuple
        elif dataset_name == 'Sonar':
            W = load_sonar_dataset(seed=seedrun, nystr_pt=nyst_points, nystr_method=nyst_method)[-1]
        elif dataset_name == 'Ionosphere':
            W = load_ionosphere_dataset(seed=seedrun, nystr_pt=nyst_points, nystr_method=nyst_method)[-1]
        elif dataset_name == 'Generated':            
            W = load_random_gen_dataset(N=2000, feat=3, seed = seedrun, nystr_pt=nyst_points, nystr_method=nyst_method)[-1]
        elif dataset_name == 'BreastCancer':
            W = load_bc_dataset(seed=seedrun, nystr_pt=nyst_points, nystr_method=nyst_method)[-1]
        elif dataset_name == 'Wine':
            W = load_wine_dataset(seed=seedrun, nystr_pt=nyst_points, nystr_method=nyst_method)[-1]

        # FedCG run
        localResidue, cent_test_accuracy, test_accuracy, elapsed, alpha = fedcg_model.fit(X_train, y_train, X_test, y_test, W=W, alpha_init=alpha_init, silent=True)

        # Predictions on the test set
        predictions = fedcg_model.predict(X_test)

        # Calculate various metrics
        #accuracy = accuracy_score(y_test, np.sign(predictions), normalize=True)
        test_precision = precision_score(y_test, np.sign(predictions))
        test_recall = recall_score(y_test, np.sign(predictions))
        test_f1 = f1_score(y_test, np.sign(predictions))
        test_mse = mean_squared_error(y_test, predictions)
        test_mae = mean_absolute_error(y_test, predictions)
        train_stopping_epoch = len(localResidue) - 1 

        # Append individual metrics to fedcg_results_df
        fedcg_results_df = pd.concat([fedcg_results_df, pd.DataFrame({'Run': run + 1, 'Seed': run, 'Central test accuracy': cent_test_accuracy, 
                                                                   'Train time': elapsed, 'Train StoppingEpoch': train_stopping_epoch,
                                                                   'Test accuracy': test_accuracy,
                                                                   'Test Precision': test_precision, 'Test Recall': test_recall, 'Test F1 Score': test_f1,
                                                                   'Test MSE': test_mse, 'Test MAE': test_mae}, index=[0])], ignore_index=True)


        # Centralized CG run
        central_localResidue,  central_test_accuracy, central_test_accuracy, central_elapsed, central_alpha = cencg_model.fit(X_train, y_train, X_test, y_test, W=W, alpha_init=alpha_init, silent=True)

        # Predictions on the test set
        central_predictions = cencg_model.predict(X_test)

        # Calculate various metrics
        #central_accuracy = accuracy_score(y_test, np.sign(central_predictions), normalize=True)
        central_precision = precision_score(y_test, np.sign(central_predictions))
        central_recall = recall_score(y_test, np.sign(central_predictions))
        central_f1 = f1_score(y_test, np.sign(central_predictions))
        central_mse = mean_squared_error(y_test, central_predictions)
        central_mae = mean_absolute_error(y_test, central_predictions)
        central_stopping_epoch = len(central_localResidue) - 1  # Assuming len(central_localResidue) represents the number of epochs

        # Append individual metrics to cencg_results_df
        cencg_results_df = pd.concat([cencg_results_df, pd.DataFrame({'Run': run + 1, 'Seed': run, 'Central test accuracy': central_test_accuracy,
                                                                   'Train time': central_elapsed, 'Train StoppingEpoch': central_stopping_epoch,
                                                                   'Test accuracy': central_test_accuracy, 'Test Precision': central_precision,
                                                                   'Test Recall': central_recall, 'Test F1 Score': central_f1,
                                                                   'Test MSE': central_mse, 'Test MAE': central_mae}, index=[0])], ignore_index=True)

    # Save dataframes to CSV with dataset_name in the file name
    fedcg_results_df.to_csv(f'nystr_fedcg_results_{dataset_name}_{nyst_method}.csv', index=False)
    cencg_results_df.to_csv(f'nystr_cencg_results_{dataset_name}_{nyst_method}.csv', index=False)


# kernel_params = Kernel(kernel_type=1, kernel_sigma = 1, poly_degree = None, binary = None)  
# [X_train, y_train, X_test, y_test, W ]=load_random_gen_dataset( N=2000, feat=3, seed=0, nystr_pt=50, nystr_method='subsampling')
# K = kernel_params.build_kernel(X_train, W)
# print('shape of K is:', K.shape)


def nys_simulation_surface(dataset_name, num_runs, X_train, y_train, X_test, y_test, nyst_method, kernel_params, nystrom_landmarks_range, lambda_range, Nb, toll):
    '''
    Args:
        dataset_name: can be 'Iris', 'Sonar', 'Ionosphere', 'Generated', 'BreastCancer', 'Wine'
        num_runs: Number of Monte Carlo simulations
        X_train: Training data
        y_train: Training labels
        X_test: Test data
        y_test: Test labels
        nyst_method: Method for Nyström landmarks generation, can be 'uniform' or 'normal' or 'subsampling'
        kernel_params: instantiation of Kernel class
        nystrom_landmarks_range: Range of Nyström points to evaluate
        lambda_range: Range of regularization parameters to evaluate
        Nb: Number of hospitals/clients
        toll: Tolerance for the stopping criterion
    Output: 
        Saves the results of the simulations to CSV files.
        Generates and saves surface and contour plots of the accuracy results.
    '''

    # Initialize arrays to store accuracy results for the surface plot
    fedcg_accuracy_surface = np.zeros((len(nystrom_landmarks_range), len(lambda_range)))
    cencg_accuracy_surface = np.zeros((len(nystrom_landmarks_range), len(lambda_range)))
    
    # Total iterations (for progress bar)
    total_iterations = len(nystrom_landmarks_range) * len(lambda_range) * num_runs
    
    with tqdm(total=total_iterations, desc="Running Simulation", ncols=100) as pbar:
        for i, nyst_points in enumerate(nystrom_landmarks_range):
            for j, lam in enumerate(lambda_range):
                fedcg_accuracies = []
                cencg_accuracies = []

                # Initialize models
                fedcg_model = FedCG(kernel_params, nyst_points, lam, Nb, toll)
                cencg_model = FedCG(kernel_params, nyst_points, lam, 1, toll)
                
                # Initialize alpha once and use it for all runs
                alpha_init = fedcg_model.initialize_alpha()

                for run in range(num_runs):
                    seedrun = np.random.seed(run)

                    # Load W matrix based on the dataset
                    if dataset_name == 'Iris':
                        W = load_iris_dataset(seed=seedrun, nystr_pt=nyst_points, nystr_method=nyst_method)[-1]
                    elif dataset_name == 'Sonar':
                        W = load_sonar_dataset(seed=seedrun, nystr_pt=nyst_points, nystr_method=nyst_method)[-1]
                    elif dataset_name == 'Ionosphere':
                        W = load_ionosphere_dataset(seed=seedrun, nystr_pt=nyst_points, nystr_method=nyst_method)[-1]
                    elif dataset_name == 'Generated':
                        W = load_random_gen_dataset(N=2000, feat=3, seed=seedrun, nystr_pt=nyst_points, nystr_method=nyst_method)[-1]
                    elif dataset_name == 'BreastCancer':
                        W = load_bc_dataset(seed=seedrun, nystr_pt=nyst_points, nystr_method=nyst_method)[-1]
                    elif dataset_name == 'Wine':
                        W = load_wine_dataset(seed=seedrun, nystr_pt=nyst_points, nystr_method=nyst_method)[-1]

                    # Federated run
                    _, _, fedcg_test_accuracy, _, _ = fedcg_model.fit(X_train, y_train, X_test, y_test, W=W, alpha_init=alpha_init, silent=True)
                    fedcg_accuracies.append(fedcg_test_accuracy)

                    # Centralized run
                    _, _, cencg_test_accuracy, _, _ = cencg_model.fit(X_train, y_train, X_test, y_test, W=W, alpha_init=alpha_init, silent=True)
                    cencg_accuracies.append(cencg_test_accuracy)
                    
                    # Update the progress bar
                    pbar.update(1)
                
                # Average accuracies
                fedcg_accuracy_surface[i, j] = np.mean(fedcg_accuracies)
                cencg_accuracy_surface[i, j] = np.mean(cencg_accuracies)

    # Save surfaces results as CSV 
    pd.DataFrame(fedcg_accuracy_surface, index=nystrom_landmarks_range, columns=lambda_range).to_csv(f'fedcg_accuracy_surface_{dataset_name}.csv')
    pd.DataFrame(cencg_accuracy_surface, index=nystrom_landmarks_range, columns=lambda_range).to_csv(f'cencg_accuracy_surface_{dataset_name}.csv')

    # Create a meshgrid for plotting 
    X, Y = np.meshgrid(nystrom_landmarks_range, lambda_range)

    # Plot and Save FedCG surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, fedcg_accuracy_surface.T, cmap='viridis')
    ax.set_xlabel('Number of Nystrom Landmarks', fontsize=10)
    ax.set_ylabel('Regularization Parameter λ', fontsize=10)
    ax.set_zlabel('Accuracy', fontsize=10)
    ax.set_title(f'FedCG Accuracy Surface - {dataset_name}', fontsize=12)
    plt.savefig(f'{dataset_name}_fedcg_surface.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Plot and Save CenCG surface
    fig = plt.figure()
    ax2 = fig.add_subplot(111, projection='3d')
    ax2.plot_surface(X, Y, cencg_accuracy_surface.T, cmap='plasma')
    ax2.set_xlabel('Number of Nystrom Landmarks', fontsize=10)
    ax2.set_ylabel('Regularization Parameter λ', fontsize=10)
    ax2.set_zlabel('Accuracy', fontsize=10)
    ax2.set_title(f'CenCG Accuracy Surface - {dataset_name}', fontsize=12)
    plt.savefig(f'{dataset_name}_cencg_surface.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Optionally, add contour plots for a 2D view
    fig, ax = plt.subplots(figsize=(8, 6))
    CS = ax.contourf(X, Y, fedcg_accuracy_surface.T, cmap='viridis')
    fig.colorbar(CS)
    ax.set_xlabel('Number of Nystrom Landmarks', fontsize=10)
    ax.set_ylabel('Regularization Parameter λ', fontsize=10)
    ax.set_title(f'FedCG Accuracy Contour - {dataset_name}', fontsize=12)
    
    # Mark the peak accuracy point
    fedcg_peak_idx = np.unravel_index(np.argmax(fedcg_accuracy_surface, axis=None), fedcg_accuracy_surface.shape)
    ax.plot(nystrom_landmarks_range[fedcg_peak_idx[0]], lambda_range[fedcg_peak_idx[1]], 'ro')  # Red dot at the peak
    ax.text(nystrom_landmarks_range[fedcg_peak_idx[0]], lambda_range[fedcg_peak_idx[1]], 
            f'Peak: {fedcg_accuracy_surface[fedcg_peak_idx]:.2f}', color='red', fontsize=10)

    plt.savefig(f'{dataset_name}_fedcg_contour.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig, ax2 = plt.subplots(figsize=(8, 6))
    CS2 = ax2.contourf(X, Y, cencg_accuracy_surface.T, cmap='plasma')
    fig.colorbar(CS2)
    ax2.set_xlabel('Number of Nystrom Landmarks', fontsize=10)
    ax2.set_ylabel('Regularization Parameter λ', fontsize=10)
    ax2.set_title(f'CenCG Accuracy Contour - {dataset_name}', fontsize=12)
    
    # Mark the peak accuracy point
    cencg_peak_idx = np.unravel_index(np.argmax(cencg_accuracy_surface, axis=None), cencg_accuracy_surface.shape)
    ax2.plot(nystrom_landmarks_range[cencg_peak_idx[0]], lambda_range[cencg_peak_idx[1]], 'ro')  # Red dot at the peak
    ax2.text(nystrom_landmarks_range[cencg_peak_idx[0]], lambda_range[cencg_peak_idx[1]], 
             f'Peak: {cencg_accuracy_surface[cencg_peak_idx]:.2f}', color='red', fontsize=10)

    plt.savefig(f'{dataset_name}_cencg_contour.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
