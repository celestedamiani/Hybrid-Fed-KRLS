import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel

        
        
class Kernel:
    '''This is a class storing the parameters to build a kernel
        type can be: 0 for linear, 1 for RBF, 2 for polynomial, 3 for random kernel, 4 wu kernel
        sigma: width/variance of the rbf kernel, default is None
        polydegree: degree of the polynomial kernel, default is None
        binary: 1 for yes, default is None
    '''
    
    def __init__(self, kernel_type, kernel_sigma = None, poly_degree = None, binary = None, nh=10, k=0.1, seed = None):
        
        self.type = kernel_type
        self.sigma = kernel_sigma
        self.degree = poly_degree
        self.binary = binary 
        self.nh = nh
        self.k = k
        self.seed = seed



    def build_kernel(self, X1, X2=None):
        """
        Computes the kernel matrix based on input data and kernel parameters.

        Args:
        - X1 (numpy.ndarray): Input data matrix 1 of shape (n_samples1, n_features).
        - X2 (numpy.ndarray, optional): Input data matrix 2 of shape (n_samples2, n_features).

        Returns:
        - K (numpy.ndarray): Computed kernel matrix of shape (n_samples1, n_samples2)
        if X2 is provided, otherwise of shape (n_samples1, n_samples1).
        """

        n1 = X1.shape[0]

        if X2 is not None:
            n2 = X2.shape[0]
            
        if self.type == 0:  # Linear kernel
            if X2 is not None:
                # For the feature-wise kernel computation
                if X1.shape[1] == 1 and X2.shape[1] == 1:
                    # Handle single feature case
                    K = np.matmul(X1, X2.T)
                else:
                    # Handle multiple features case
                    K = np.matmul(X1, X2.T)
            else:
                K = np.matmul(X1, X1.T)

        elif self.type == 1:  # RBF kernel
            if X2 is not None:
                K = rbf_kernel(X1, X2, gamma=1.0 / (2 * self.sigma**2))
            else:
                K = rbf_kernel(X1, X1, gamma=1.0 / (2 * self.sigma**2))

        elif self.type == 2:  # Polynomial kernel
            if X2 is not None:
                K = polynomial_kernel(X1, X2, gamma=self.degree)
            else:
                K = polynomial_kernel(X1, X1, gamma=self.degree)
                
        elif self.type == 3:
            if self.binary == 1:
                num = 4
            else:
                num = 3
            
            # Implement logic for type == 3 (random kernel)
            np.random.seed(self.seed)
            R = np.random.randn(X1.shape[1], self.nh, num)
            
            if X2 is not None:
                H1 = np.tanh(np.dot(X1, R[:, :, -1]))
                H2 = np.tanh(np.dot(X2, R[:, :, -1]))
                K = np.dot(H2, H1.T)
            else:
                H = np.tanh(np.dot(X1, R[:, :, -1]))
                K = np.dot(H, H.T)
        elif self.type == 4: 
            print('Wu kernel not defined')
            
        else:        
            raise ValueError("Kernel type not valid: " + str(self.type))
        
        return K