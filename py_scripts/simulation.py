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



def find_peak_accuracy(accuracy_surface):
    peak_idx = np.unravel_index(np.argmax(accuracy_surface), accuracy_surface.shape)
    peak_value = accuracy_surface[peak_idx]
    return peak_idx, peak_value


def plot_surface(X, Y, Z, model_name, dataset_name, peak_nystrom, peak_lambda, peak_accuracy):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('Number of Nystrom Landmarks', fontsize=10)
    ax.set_ylabel('Regularization Parameter λ', fontsize=10)
    ax.set_zlabel('Accuracy', fontsize=10)
    ax.set_title(f'{model_name} Accuracy Surface - {dataset_name}', fontsize=12)
    
    # Mark the peak accuracy point
    ax.scatter([peak_nystrom], [peak_lambda], [peak_accuracy], color='red', s=50, marker='*')
    
    plt.savefig(f'{dataset_name}_{model_name.lower()}_surface.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_contour(X, Y, Z, model_name, dataset_name, peak_nystrom, peak_lambda, peak_accuracy):
    fig, ax = plt.subplots(figsize=(10, 8))
    CS = ax.contourf(X, Y, Z, cmap='viridis')
    fig.colorbar(CS)
    ax.set_xlabel('Number of Nystrom Landmarks', fontsize=12)
    ax.set_ylabel('Regularization Parameter λ', fontsize=12)
    ax.set_title(f'{model_name} Accuracy Contour - {dataset_name}', fontsize=14)
    
    # Mark the peak accuracy point
    ax.plot(peak_nystrom, peak_lambda, 'ro')
    ax.annotate(f'Peak: {peak_accuracy:.2f}', (peak_nystrom, peak_lambda), xytext=(5, 5), 
                textcoords='offset points', color='red', fontweight='bold')
    
    plt.savefig(f'{dataset_name}_{model_name.lower()}_contour.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    
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
    
    # Find peak accuracies
    fedcg_peak_idx, fedcg_peak_accuracy = find_peak_accuracy(fedcg_accuracy_surface)
    cencg_peak_idx, cencg_peak_accuracy = find_peak_accuracy(cencg_accuracy_surface)
    
    # Get corresponding Nyström landmarks and lambda values
    fedcg_peak_nystrom = nystrom_landmarks_range[fedcg_peak_idx[0]]
    fedcg_peak_lambda = lambda_range[fedcg_peak_idx[1]]
    cencg_peak_nystrom = nystrom_landmarks_range[cencg_peak_idx[0]]
    cencg_peak_lambda = lambda_range[cencg_peak_idx[1]]

    # Plot and Save FedCG surface
    plot_surface(X, Y, fedcg_accuracy_surface.T, "FedCG", dataset_name, fedcg_peak_nystrom, fedcg_peak_lambda, fedcg_peak_accuracy)

    # Plot and Save CenCG surface
    plot_surface(X, Y, cencg_accuracy_surface.T, "CenCG", dataset_name, cencg_peak_nystrom, cencg_peak_lambda, cencg_peak_accuracy)

    # Plot and Save FedCG contour
    plot_contour(X, Y, fedcg_accuracy_surface.T, "FedCG", dataset_name, fedcg_peak_nystrom, fedcg_peak_lambda, fedcg_peak_accuracy)

    # Plot and Save CenCG contour
    plot_contour(X, Y, cencg_accuracy_surface.T, "CenCG", dataset_name, cencg_peak_nystrom, cencg_peak_lambda, cencg_peak_accuracy)




### NEW APPROACH FOR LESS COMPUTATION TIME

def nys_simulation_surface_chunk(dataset_name, num_runs, X_train, y_train, X_test, y_test, nyst_method, kernel_params, nystrom_landmarks_range, lambda_range, Nb, toll, chunk_id):
    # Initialize arrays to store accuracy results for this chunk
    fedcg_accuracy_surface = np.zeros((len(nystrom_landmarks_range), len(lambda_range)))
    cencg_accuracy_surface = np.zeros((len(nystrom_landmarks_range), len(lambda_range)))
    
    total_iterations = len(nystrom_landmarks_range) * len(lambda_range) * num_runs
    
    with tqdm(total=total_iterations, desc=f"Running Chunk {chunk_id}", ncols=100) as pbar:
        for i, nyst_points in enumerate(nystrom_landmarks_range):
            for j, lam in enumerate(lambda_range):
                fedcg_accuracies = []
                cencg_accuracies = []

                fedcg_model = FedCG(kernel_params, nyst_points, lam, Nb, toll)
                cencg_model = FedCG(kernel_params, nyst_points, lam, 1, toll)
                
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

                    _, _, fedcg_test_accuracy, _, _ = fedcg_model.fit(X_train, y_train, X_test, y_test, W=W, alpha_init=alpha_init, silent=True)
                    fedcg_accuracies.append(fedcg_test_accuracy)

                    _, _, cencg_test_accuracy, _, _ = cencg_model.fit(X_train, y_train, X_test, y_test, W=W, alpha_init=alpha_init, silent=True)
                    cencg_accuracies.append(cencg_test_accuracy)
                    
                    pbar.update(1)
                
                fedcg_accuracy_surface[i, j] = np.mean(fedcg_accuracies)
                cencg_accuracy_surface[i, j] = np.mean(cencg_accuracies)

    # Save chunk results
    pd.DataFrame(fedcg_accuracy_surface, index=nystrom_landmarks_range, columns=lambda_range).to_csv(f'fedcg_accuracy_surface_{dataset_name}_chunk_{chunk_id}.csv')
    pd.DataFrame(cencg_accuracy_surface, index=nystrom_landmarks_range, columns=lambda_range).to_csv(f'cencg_accuracy_surface_{dataset_name}_chunk_{chunk_id}.csv')



def combine_and_plot_results(dataset_name, chunks):
    fedcg_combined = pd.DataFrame()
    cencg_combined = pd.DataFrame()

    for chunk_id in range(1, len(chunks) + 1):
        fedcg_chunk = pd.read_csv(f'fedcg_accuracy_surface_{dataset_name}_chunk_{chunk_id}.csv', index_col=0)
        cencg_chunk = pd.read_csv(f'cencg_accuracy_surface_{dataset_name}_chunk_{chunk_id}.csv', index_col=0)
        
        fedcg_combined = pd.concat([fedcg_combined, fedcg_chunk])
        cencg_combined = pd.concat([cencg_combined, cencg_chunk])

    # Sort and save combined results
    fedcg_combined.sort_index().to_csv(f'fedcg_accuracy_surface_{dataset_name}_combined.csv')
    cencg_combined.sort_index().to_csv(f'cencg_accuracy_surface_{dataset_name}_combined.csv')

    # Create meshgrid for plotting
    nystrom_landmarks_range = fedcg_combined.index.astype(float)
    lambda_range = fedcg_combined.columns.astype(float)
    X, Y = np.meshgrid(nystrom_landmarks_range, lambda_range)

    # Find peaks and plot for FedCG
    fedcg_peak_idx, fedcg_peak_accuracy = find_peak_accuracy(fedcg_combined.values)
    fedcg_peak_nystrom = nystrom_landmarks_range[fedcg_peak_idx[0]]
    fedcg_peak_lambda = lambda_range[fedcg_peak_idx[1]]
    plot_surface(X, Y, fedcg_combined.values.T, "FedCG", dataset_name, fedcg_peak_nystrom, fedcg_peak_lambda, fedcg_peak_accuracy)
    plot_contour(X, Y, fedcg_combined.values.T, "FedCG", dataset_name, fedcg_peak_nystrom, fedcg_peak_lambda, fedcg_peak_accuracy)

    # Find peaks and plot for CenCG
    cencg_peak_idx, cencg_peak_accuracy = find_peak_accuracy(cencg_combined.values)
    cencg_peak_nystrom = nystrom_landmarks_range[cencg_peak_idx[0]]
    cencg_peak_lambda = lambda_range[cencg_peak_idx[1]]
    plot_surface(X, Y, cencg_combined.values.T, "CenCG", dataset_name, cencg_peak_nystrom, cencg_peak_lambda, cencg_peak_accuracy)
    plot_contour(X, Y, cencg_combined.values.T, "CenCG", dataset_name, cencg_peak_nystrom, cencg_peak_lambda, cencg_peak_accuracy)

