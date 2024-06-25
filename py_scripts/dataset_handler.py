import numpy as np
import pandas as pd
from numpy.random import seed
from numpy.random import randn
from numpy.random import rand
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import datasets
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, fetch_openml
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from mpl_toolkits.mplot3d import Axes3D




    
def load_random_gen_dataset( N=2000, feat=3, seed=0,nystr_pt=50, nystr_method='uniform'):
    '''Generate a synthetic dataset with 3 hospitals.
    Args:   N: number of samples per hospital
            feat: number of features
            seed: random seed
            nystr_pt: number of Nystrom points
            nystr_method: method to generate Nystrom points, can be 'uniform', 'normal', or 'subsampling'
    Returns: X_train, y_train, X_test, y_test, W (Nystrom matrix)'''
    print("Loading random generated dataset...")
    # Constants definition
    HOSPITAL_COORDS = np.array([[2, 5], [0, 2], [2, -1]])
    NUM_HOSPITALS = len(HOSPITAL_COORDS)
    ni = feat - 1
    
    # Setting random seed for reproducibility
    np.random.seed(seed)
    
    # Generate  data
    X = np.concatenate([np.random.randn(N, ni) + coord for coord in HOSPITAL_COORDS], axis=0)
    X_train = np.concatenate([X, np.ones((NUM_HOSPITALS * N, 1))], axis=1)
    X_test = np.concatenate([np.random.randn(N, ni) + coord for coord in HOSPITAL_COORDS], axis=0)
    X_test = np.concatenate([X_test, np.ones((NUM_HOSPITALS * N, 1))], axis=1)

    # normalise columns of X and Xt
    for i in range(ni):
        mi = min(X_train[:, i])         # min in the i-th column
        ma = max(X_train[:, i])         # max in the i-th column
        X_train[:, i] = (X_train[:, i] - mi) / (ma-mi)         # normalise X_train by column
        X_test[:, i] = (X_test[:, i] - mi) / (ma-mi)         # normalise X_test by column with X_train's min and max

    # Creating labels
    y_train = np.concatenate([np.ones(N), -np.ones(N), np.ones(N)], axis=0)
    y_test= np.concatenate([np.ones(N), -np.ones(N), np.ones(N)], axis=0)

    # sense check
    #print("y_train and y_test shape: ", y_train.shape, y_test.shape)
    #print("X_train and X_test shape: ", X_train.shape, X_test.shape)
    
    if nystr_method == 'uniform':
        W = rand(nystr_pt, feat)
    elif nystr_method == 'normal':
        # Calculate the standard deviation of each feature in X_train
        X_train_std = np.std(X_train, axis=0)
        X_train_mean = np.mean(X_train, axis=0)
        # Generate matrix with normal distribution using adjusted sigma
        W = np.random.randn(nystr_pt, X_train.shape[1]) * X_train_std + X_train_mean
    elif nystr_method == 'subsampling':
        idx = np.random.choice(X_train.shape[0], nystr_pt, replace=False)
        W = X_train[idx, :]
    else:
        raise ValueError("nystr_method should be 'uniform', 'normal', or 'subsampling'")
    
    print("W matrix shape: ", W.shape)

    return X_train, y_train, X_test, y_test, W
    

def load_iris_dataset(seed=0, nystr_pt=50, nystr_method='uniform'):
    '''Load the iris dataset from sklearn.datasets'''
    print("Loading Iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target
    
    print("Features are:", ', '.join(iris.feature_names))
    print("Classes are:", ', '.join(iris.target_names))
    print("but we'll consider 'setosa' as one class and 'versicolor'+'virginica' as another class")
    
    # reducing to two classes
    y = np.where(y == 0, 1, -1)  # 'setosa' as class 1 and others as class -1

    X_train_temp, X_test_temp, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)    
    
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train_temp) 
    X_test = min_max_scaler.transform(X_test_temp)

    print("y_train and y_test shape:", y_train.shape, y_test.shape)
    print("X_train and X_test shape:", X_train.shape, X_test.shape)
    
    if nystr_method == 'uniform':
        W = rand(nystr_pt, X_train.shape[1] )
    elif nystr_method == 'normal':
        # Calculate the standard deviation of each feature in X_train
        X_train_std = np.std(X_train, axis=0)
        X_train_mean = np.mean(X_train, axis=0)
        # Generate matrix with normal distribution using adjusted sigma
        W = np.random.randn(nystr_pt, X_train.shape[1]) * X_train_std + X_train_mean
    elif nystr_method == 'subsampling':
        idx = np.random.choice(X_train.shape[0], nystr_pt, replace=False)
        W = X_train[idx, :]
    else:
        raise ValueError("nystr_method should be 'uniform', 'normal', or 'subsampling'")
    
    print("W matrix shape: ", W.shape)

    
    return X_train, y_train, X_test, y_test, W


def load_ionosphere_dataset(seed=0, nystr_pt=50, nystr_method='uniform'):
    '''Load the Ionosphere dataset from OpenML'''
    #print("Loading Ionosphere dataset...")
    ionosphere = fetch_openml(name='ionosphere', version=1, parser='auto')  # Explicitly setting parser='auto'
    X, y = ionosphere.data, ionosphere.target
    
    # Convert labels to numeric values (1 for 'g' and -1 for 'b')
    y = np.where(y == 'g', 1, -1)

    X_train_temp, X_test_temp, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train_temp)
    X_test = min_max_scaler.transform(X_test_temp)

    #print("y_train and y_test shape:", y_train.shape, y_test.shape)
    #print("X_train and X_test shape:", X_train.shape, X_test.shape)
    
    if nystr_method == 'uniform':
        W = np.random.rand(nystr_pt, X_train.shape[1])
    elif nystr_method == 'normal':
        # Calculate the standard deviation of each feature in X_train
        X_train_std = np.std(X_train, axis=0)
        X_train_mean = np.mean(X_train, axis=0)
        # Generate matrix with normal distribution using adjusted sigma
        W = np.random.randn(nystr_pt, X_train.shape[1]) * X_train_std + X_train_mean
    elif nystr_method == 'subsampling':
        idx = np.random.choice(X_train.shape[0], nystr_pt, replace=False)
        W = X_train[idx, :]
    else:
        raise ValueError("nystr_method should be 'uniform', 'normal', or 'subsampling'")
    
    return X_train, y_train, X_test, y_test, W

# Modify load_ionosphere_dataset function to return a validation set
def load_ionosphere_dataset_validation(seed=0, nystr_pt=50, nystr_method='uniform', split_size=(0.6, 0.2, 0.2)):
    '''Load the Ionosphere dataset from OpenML'''
    ionosphere = fetch_openml(name='ionosphere', version=1, parser='auto')
    X, y = ionosphere.data, ionosphere.target
    y = np.where(y == 'g', 1, -1)

    # Split dataset into training, validation, and test sets
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=split_size[2], random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=split_size[1] / (split_size[0] + split_size[1]), random_state=seed)

    # Normalize features
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_val = min_max_scaler.transform(X_val)
    X_test = min_max_scaler.transform(X_test)
    #print("X_train, X_val and X_test shape:", X_train.shape, X_val.shape, X_test.shape)


    # Generate W matrix based on nystr_method
    if nystr_method == 'uniform':
        W = np.random.rand(nystr_pt, X_train.shape[1])
    elif nystr_method == 'normal':
        X_train_std = np.std(X_train, axis=0)
        X_train_mean = np.mean(X_train, axis=0)
        W = np.random.randn(nystr_pt, X_train.shape[1]) * X_train_std + X_train_mean
    elif nystr_method == 'subsampling':
        idx = np.random.choice(X_train.shape[0], nystr_pt, replace=False)
        W = X_train[idx, :]
    else:
        raise ValueError("nystr_method should be 'uniform', 'normal', or 'subsampling'")

    return X_train, X_val, X_test, y_train, y_val, y_test, W
    
def load_sonar_dataset(seed=0, nystr_pt=50, nystr_method='uniform'):
    '''Load the Sonar dataset from OpenML'''
    print("Loading Sonar dataset...")
    sonar = fetch_openml(name='sonar', version=1, parser='auto')  # Explicitly setting parser='auto'
    X, y = sonar.data, sonar.target
    
    # Print unique labels to verify class assignment
    print("Unique labels in the dataset:", np.unique(y))
    
    # Convert labels to numeric values (1 for 'R' and -1 for 'M')
    y = np.where(y == 'Rock', 1, -1)

    X_train_temp, X_test_temp, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train_temp)
    X_test = min_max_scaler.transform(X_test_temp)

    #print("y_train and y_test shape:", y_train.shape, y_test.shape)
    #print("X_train and X_test shape:", X_train.shape, X_test.shape)
    
    if nystr_method == 'uniform':
        W = np.random.rand(nystr_pt, X_train.shape[1])
    elif nystr_method == 'normal':
        # Calculate the standard deviation of each feature in X_train
        X_train_std = np.std(X_train, axis=0)
        X_train_mean = np.mean(X_train, axis=0)
        # Generate matrix with normal distribution using adjusted sigma
        W = np.random.randn(nystr_pt, X_train.shape[1]) * X_train_std + X_train_mean
    elif nystr_method == 'subsampling':
        idx = np.random.choice(X_train.shape[0], nystr_pt, replace=False)
        W = X_train[idx, :]
    else:
        raise ValueError("nystr_method should be 'uniform', 'normal', or 'subsampling'")
    
    return X_train, y_train, X_test, y_test, W


def load_bc_dataset(seed=0, nystr_pt=50, nystr_method='uniform'):
    '''Load the Breast Cancer Wisconsin dataset'''
    print("Loading the Breast Cancer Wisconsin dataset...")
    breast_ds= load_breast_cancer()
    
    X, y = breast_ds.data, breast_ds.target
    
    for i in range(len(y)):
        if y[i] == 0:
            y[i] = 1
        else:
            y[i] = -1 
    
    # Splitting the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    
    # Scaling features using MinMaxScaler
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.transform(X_test)


    print("X_train and X_test shape:", X_train.shape, X_test.shape)
    print("y_train and y_test shape:", y_train.shape, y_test.shape)
    
    # Nystrom matrix generation based on nystr_method
    if nystr_method == 'uniform':
        W = np.random.rand(nystr_pt, X_train.shape[1])
    elif nystr_method == 'normal':
        # Calculate the standard deviation of each feature in X_train
        X_train_std = np.std(X_train, axis=0)
        X_train_mean = np.mean(X_train, axis=0)
        # Generate matrix with normal distribution using adjusted sigma
        W = np.random.randn(nystr_pt, X_train.shape[1]) * X_train_std + X_train_mean
    elif nystr_method == 'subsampling':
        idx = np.random.choice(X_train.shape[0], nystr_pt, replace=False)
        W = X_train[idx, :]
    else:
        raise ValueError("nystr_method should be 'uniform', 'normal', or 'subsampling'")
    
    return X_train, y_train, X_test, y_test, W

def load_wine_dataset(seed=0, nystr_pt=50, nystr_method='uniform'):
    '''Load the Wine Recognition dataset from sklearn.datasets'''
    print("Loading Wine Recognition dataset...")
    wine = load_wine()
    X, y = wine.data, wine.target
    
    print("Features are:", ', '.join(wine.feature_names))
    print("Classes are:", ', '.join(wine.target_names))
    print("We will consider 'class_1' as one class and 'class_0'+'class_2' as another")
    
    # reducing to two classes
    y = np.where(y == 1, 1, -1)  # 'class_1' as class 1 and others as class -1

    X_train_temp, X_test_temp, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)    
    
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train_temp) 
    X_test = min_max_scaler.transform(X_test_temp)

    print("y_train and y_test shape:", y_train.shape, y_test.shape)
    print("X_train and X_test shape:", X_train.shape, X_test.shape)
    
    if nystr_method == 'uniform':
        W = np.random.rand(nystr_pt, X_train.shape[1])
    elif nystr_method == 'normal':
        # Calculate the standard deviation of each feature in X_train
        X_train_std = np.std(X_train, axis=0)
        X_train_mean = np.mean(X_train, axis=0)
        # Generate matrix with normal distribution using adjusted sigma
        W = np.random.randn(nystr_pt, X_train.shape[1]) * X_train_std + X_train_mean
    elif nystr_method == 'subsampling':
        idx = np.random.choice(X_train.shape[0], nystr_pt, replace=False)
        W = X_train[idx, :]
    else:
        raise ValueError("nystr_method should be 'uniform', 'normal', or 'subsampling'")
    
    print("W matrix shape: ", W.shape)

    return X_train, y_train, X_test, y_test, W








    
def visualize_dataset_tsne(X, y,  dataset_name):
    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(X)
    
    plt.figure(figsize=(8, 6))
    
    # Mapping labels to unique integers for color differentiation
    unique_labels = np.unique(y)
    label_colors = plt.cm.get_cmap('viridis', len(unique_labels))
    

    for i, label in enumerate(unique_labels):
        label_X = X_embedded[y == label]
        plt.scatter(label_X[:, 0], label_X[:, 1], c=[label_colors(i)], label=f'Class {label}', alpha=0.7, edgecolors='k')

    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis')
    plt.title(f"Visualization of {dataset_name} dataset (t-SNE)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.colorbar(label='Class')
    plt.legend()
    plt.grid(True)
    plt.colorbar(label='Class')    
    plt.show()
    plt.savefig(f"{dataset_name}_viz_tsne.png")  # Save as an image file


def visualize_dataset_pca(X, y, dataset_name):
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(8, 6))

    unique_labels = np.unique(y)
    label_colors = plt.cm.get_cmap('viridis', len(unique_labels))

    for i, label in enumerate(unique_labels):
        label_X = X_pca[y == label]
        plt.scatter(label_X[:, 0], label_X[:, 1], c=[label_colors(i)], label=f'Class {label}', alpha=0.7, edgecolors='k')

    plt.title(f"Visualization of {dataset_name} dataset (PCA)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True)
    plt.colorbar(label='Class')
    plt.show()
    plt.savefig(f"{dataset_name}_viz_pca.png")  # Save as an image file

def visualize_dataset_tsne_with_W(X, y, W, dataset_name):
    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(X)
    W_embedded = tsne.fit_transform(W)  # Embedding W using t-SNE
    
    plt.figure(figsize=(8, 6))
    
    # Mapping labels to unique integers for color differentiation
    unique_labels = np.unique(y)
    label_colors = plt.cm.get_cmap('viridis', len(unique_labels))
    
    # Plotting original dataset
    for i, label in enumerate(unique_labels):
        label_X = X_embedded[y == label]
        plt.scatter(label_X[:, 0], label_X[:, 1], c=[label_colors(i)], label=f'Class {label}', alpha=0.7, edgecolors='k')

    # Plotting points from matrix W in a different color
    plt.scatter(W_embedded[:, 0], W_embedded[:, 1], c='red', label='Matrix W', alpha=0.7, edgecolors='k')

    plt.title(f"Visualization of {dataset_name} dataset with Matrix W (t-SNE)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{dataset_name}_viz_tsne.png")  # Save as an image file
    plt.show()


def visualize_dataset_pca_with_W(X, y, W, dataset_name):
    pca = PCA(n_components=2, random_state=42)
    X_embedded = pca.fit_transform(X)
    W_embedded = pca.fit_transform(W)  # Embedding W using t-SNE
    
    plt.figure(figsize=(8, 6))
    
    # Mapping labels to unique integers for color differentiation
    unique_labels = np.unique(y)
    label_colors = plt.cm.get_cmap('viridis', len(unique_labels))
    
    # Plotting original dataset
    for i, label in enumerate(unique_labels):
        label_X = X_embedded[y == label]
        plt.scatter(label_X[:, 0], label_X[:, 1], c=[label_colors(i)], label=f'Class {label}', alpha=0.7, edgecolors='k')

    # Plotting points from matrix W in a different color
    plt.scatter(W_embedded[:, 0], W_embedded[:, 1], c='red', label='Matrix W', alpha=0.7, edgecolors='k')

    plt.title(f"Visualization of {dataset_name} dataset with matrix W (PCA)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{dataset_name}_viz_pca.png")  # Save as an image file
    plt.show()


def visualize_dataset_tsne_with_W_3d(X, y, W, dataset_name):
    tsne = TSNE(n_components=3, random_state=42)
    X_embedded = tsne.fit_transform(X)
    W_embedded = tsne.fit_transform(W)  # Embedding W using t-SNE
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Mapping labels to unique integers for color differentiation
    unique_labels = np.unique(y)
    label_colors = plt.cm.get_cmap('viridis', len(unique_labels))
    
    # Plotting original dataset
    for i, label in enumerate(unique_labels):
        label_X = X_embedded[y == label]
        ax.scatter(label_X[:, 0], label_X[:, 1], label_X[:, 2], c=[label_colors(i)], label=f'Class {label}', alpha=0.7, edgecolors='k')

    # Plotting points from matrix W in a different color
    ax.scatter(W_embedded[:, 0], W_embedded[:, 1], W_embedded[:, 2], c='red', label='Matrix W', alpha=0.7, edgecolors='k')

    ax.set_title(f"Visualization of {dataset_name} dataset with Matrix W (t-SNE 3D)")
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    ax.set_zlabel("t-SNE Component 3")
    ax.legend()
    plt.savefig(f"{dataset_name}_viz_tsne_3d.png")  # Save as an image file
    plt.show()


def visualize_dataset_pca_with_W_3d(X, y, W, dataset_name):
    pca = PCA(n_components=3, random_state=42)
    X_embedded = pca.fit_transform(X)
    W_embedded = pca.fit_transform(W)  # Embedding W using PCA
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Mapping labels to unique integers for color differentiation
    unique_labels = np.unique(y)
    label_colors = plt.cm.get_cmap('viridis', len(unique_labels))
    
    # Plotting original dataset
    for i, label in enumerate(unique_labels):
        label_X = X_embedded[y == label]
        ax.scatter(label_X[:, 0], label_X[:, 1], label_X[:, 2], c=[label_colors(i)], label=f'Class {label}', alpha=0.7, edgecolors='k')

    # Plotting points from matrix W in a different color
    ax.scatter(W_embedded[:, 0], W_embedded[:, 1], W_embedded[:, 2], c='red', label='Matrix W', alpha=0.7, edgecolors='k')

    ax.set_title(f"Visualization of {dataset_name} dataset with Matrix W (PCA 3D)")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_zlabel("PCA Component 3")
    ax.legend()
    plt.savefig(f"{dataset_name}_viz_pca_3d.png")  # Save as an image file
    plt.show()



#[X_train, X_val, X_test, y_train, y_val, y_test, W] = load_ionosphere_dataset_validation(seed=0, nystr_pt=106, nystr_method='uniform', split_size=(0.6, 0.2, 0.2))
# print("Unique labels in y_train:", np.unique(y_train))
# print("Unique labels in y_val:", np.unique(y_val))
# # print("Unique labels in y_test:", np.unique(y_test))
# print("y_test", y_test)
# print("y_val", y_val)
# #visualize_dataset_tsne_with_W(X_train, y_train, W, "Ionosphere")
# visualize_dataset_pca_with_W(X_train, y_train, W, "Ionosphere")

# [X_train, y_train, X_test, y_test, W ]=load_random_gen_dataset( N=2000, feat=3, seed=0, nystr_pt=50, nystr_method='subsampling')
# print("Unique labels in y_train:", np.unique(y_train))
# visualize_dataset_tsne(X_train, y_train, "Generated")
# visualize_dataset_pca(X_train, y_train, "Generated")

# [X_train, y_train, X_test, y_test, W ]=load_iris_dataset(seed=0, nystr_pt=50, nystr_method='uniform')
# print("Unique labels in y_train:", np.unique(y_train))
# visualize_dataset_tsne_with_W(X_train, y_train, W, "Iris_uniform")
# visualize_dataset_pca(X_train, y_train, "Iris")

# [X_train, y_train, X_test, y_test, W ]=load_ionosphere_dataset(seed=0, nystr_pt=50, nystr_method='uniform')
# print("Unique labels in y_train:", np.unique(y_train))
# visualize_dataset_tsne(X_train, y_train, "Ionosphere")
# visualize_dataset_pca(X_train, y_train, "Ionosphere")


#[X_train, y_train, X_test, y_test, W ]=load_sonar_dataset(seed=0, nystr_pt=50, nystr_method='subsampling')
# print("Unique labels in y_train:", np.unique(y_train))
# visualize_dataset_pca_with_W_3d(X_train, y_train, W, "Sonar_subsampling")
# # visualize_dataset_pca(X_train, y_train, "Sonar")



# print(X_train)

