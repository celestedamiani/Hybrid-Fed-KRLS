import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
from kernel import Kernel  # Make sure to import your Kernel class
from federated_functions import FedCG  # Make sure to import your FedCG class
from dataset_handler import * 



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