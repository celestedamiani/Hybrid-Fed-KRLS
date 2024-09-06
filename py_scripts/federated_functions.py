from kernel import Kernel

import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt  #for basic plots
from sklearn.metrics import accuracy_score
import time




class FedCG:
    
    def __init__(self, kernel_params, nyst_points, lam, Nb, toll):
        self.kernel_params = kernel_params
        self.nyst_points = nyst_points
        self.lam = lam
        self.Nb = Nb
        self.toll = toll
        self.alpha_init = None  
        self.localResidue = None
        self.central_test_accuracy= None
        self.test_accuracy = None
        self.elapsed = None
        self.alpha = None
        self.W = None
        
    def simulated_federated_cycles(self, X, y, Xt, yt, alpha_init, silent=False):
        '''
        Parameters
        -----------
        X: Samples array
        y: labels
        Xt: test samples
        yt: test labels
        alpha_init: initialisation of the alpha vector
   
        Returns
        -----------
        localResidue: list of local residues
        central_test_accuracy: accuracy of the central model on the test set
        test_accuracy: accuracy on the federated model on the test set
        elapsed: time elapsed
        alpha: final alpha vector

        '''
        # Accessing the parameters from the class attributes
        kernel_params = self.kernel_params
        lam = self.lam
        Nb = self.Nb
        toll = self.toll
        Mnys = self.nyst_points
        W = self.W

        if W is None or alpha_init is None:
            raise ValueError("W and alpha_init must be provided for this method.")
    
    
        ##### Global parameters: #####
        M = X.shape[0]
        Ne =  M     # at worst it converges in this number of steps 
        ni= X.shape[1]        # number of features/omic centres
        block = Ne/Nb
        
    
        ###########################
        ##### Initialisation: #####
        ##### first time estimate the full gradient
        ##### with cycles on the clients for the same block
        ###########################
        alpha = alpha_init.copy()
        Y = self.build_Y_matrix(M, Mnys, y) # write labels in a 'wrapping' diagonal matrix
        start = 0  
        final = int(start + block)  
        #print('start, final: ', start, final)
      
        G =  np.zeros(Mnys)
        p = np.zeros(Mnys)
        v = []    # v will be (Kalpha-y+eta)
        eta = 100*randn(M) # noise
        ETA_ETA = self.build_Y_matrix(M, Mnys, eta) # write noise in a 'wrapping' diagonal matrix
        
        for ib in range(Nb): # cycle through batches - hospitals
            Xb = X[start:final, :]            
            [KtKp_b, v_b] = self.feature_federated_gradient(Xb, start, final, Y, M,  ETA_ETA, alpha, p)

            if ib == 0:
                v =v_b.copy()
            else:
                v = v + v_b         # sever removes the noise and sum 
            start = int(start + block)
            final = int(final + block)
        
        # resetting to start again at first batch     
        start = 0
        final =  int(start + block)

        # compute KtKalpha-Kty_noisy
        for ib in range(Nb):
            Xb = X[start:final, :]
            KtKalpha_Kty_noisy = []

            # cycle on the features
            for iff in range(ni,0,- 1):
                Ki = kernel_params.build_kernel(Xb[:, iff - 1:iff], W[:, iff - 1:iff])

                
                # the server delivers a noisy version of v
                # the client removes the noise before processing
                if (iff == ni):
                    KtKalpha_Kty_noisy =np.matmul(Ki.T,np.diag(v[start:final]))
                else:
                    KtKalpha_Kty_noisy =np.multiply(Ki.T, KtKalpha_Kty_noisy)

            G = G + np.sum(KtKalpha_Kty_noisy,axis=1)
            start = int(start + block)
            final = int(final + block)
            
        # Remove noise from the gradient 
        noise_removal=0
        start = 0
        final =  int(start + block)
        
        for ib in range(Nb):
            Xb = X[start:final, :]
                        
            #cycle on the features
            for iff in range(ni,0,- 1):
                Ki = kernel_params.build_kernel(Xb[:, iff - 1:iff], W[:, iff - 1:iff])

                
                if (iff == ni):
                    noise_removal = np.matmul(Ki.T,np.diag(eta[start:final]))
                else:
                    noise_removal = np.multiply(Ki.T, noise_removal)
            G =  G - np.sum(noise_removal, axis=1)
            start = int(start + block)
            final = int(final + block)

        G = G + lam * np.matmul(np.eye(Mnys), alpha) 
        p = -G.copy()
        

        r = p.copy()
            
        ###########################
        ##### Iterations: #####
        ###########################
        localResidue = []
        err = np.sum((r)**2)
        localResidue.append(err)
        t = time.time()

        # global epochs - external cycle of the coniugated gradient
        for epoch in range(Ne):
            if not silent:
                print("\n \n Starting epoch: ", epoch)

            start = 0  
            final = int(start + block)

            KtKp =  np.zeros(Mnys)
            pKtKp = 0
            # Cycle throguh batches - hospitals
            for h in range(Nb):
                Xb = X[start:final, :]

                [KtKp_b, not_used] = self.feature_federated_gradient(Xb, start, final, Y, M, ETA_ETA, alpha, p)        
                
                KtKp = KtKp + KtKp_b
                pKtKp = pKtKp + np.matmul(p.T, KtKp_b)
                start = int(start + block)
                final = int(final + block)
            a = (np.dot(r.transpose(), r))/pKtKp  # take conj grad step
            alpha = alpha+a*p            
            rold = r.copy()       
            r = r-a*KtKp               
            err = np.sum((r)**2)
            
            # test
            Kt = kernel_params.build_kernel(Xt, W)
            # Print out the entire kernel matrix Kt
            # print("Kernel matrix Kt:", Kt)

            # # Check the shape of the kernel matrix
            # print("Shape of the kernel matrix Kt:", Kt.shape)

            # # Check for NaN or infinity values in the kernel matrix
            # print("NaN values in Kt:", np.isnan(Kt).any())
            # print("Infinity values in Kt:", np.isinf(Kt).any())

            fed_f =np.matmul(Kt, alpha)
            #print(f"fed_f contains NaN: {np.isnan(fed_f).any()}")  # Check for NaN in fed_f
            #print("Values of fed_f:", fed_f)

            if np.isnan(fed_f).any():
                # Handle NaN values or investigate the cause further
                print("NaN values found in fed_f. Investigate further.")

            elif np.count_nonzero(yt) == 0:
                print("All labels in the test set are zero. Investigate further.")
                
            # Calculate accuracy score only if fed_f does not contain NaN values
            else:
                acc_fe = accuracy_score(yt, np.sign(fed_f), normalize=True)
                if not silent:
                    print('Epoch {}, local residue {}, test acc {}'.format(epoch, err, acc_fe))
                localResidue.append(err)            
                
            if (err<toll):
                    break  
            beta = np.dot(r.transpose(), r)/(np.dot(rold.transpose(), rold))
            p = r+beta*p
         
        elapsed = time.time() - t
        if not silent:
            print('Time elapsed in global epochs: ', elapsed)

        # the global kernel will be used for testing performace
        K = kernel_params.build_kernel(X, W)
       
        fed_cost = np.sum((np.matmul(K, alpha)-y)**2) + lam * np.sum(alpha**2) 
        Kt = kernel_params.build_kernel(Xt, W)

        fed_f = np.matmul(Kt,alpha)
        acc_fe = accuracy_score(yt, np.sign(fed_f), normalize = True)
        if not silent:
            print('Federated cost: {}, federated test accuracy: {}'.format(fed_cost,acc_fe))

        #alpha_true = np.linalg.lstsq((np.matmul(K.transpose(), K)+ lam * np.eye(Mnys)), (np.matmul(K.transpose(), y)), rcond=None)[0]
        alpha_true = np.linalg.solve((np.matmul(K.transpose(), K)+ lam * np.eye(Mnys)), (np.matmul(K.transpose(), y)))
        f = np.matmul(Kt, alpha_true)
        cost = np.sum((np.matmul(K, alpha_true)-y)**2)+ lam * np.sum(alpha_true**2)
        acc = accuracy_score(yt, np.sign(f), normalize = True)
        if not silent:
            print('Cost: {}, central test accuracy {}'.format(cost,acc))
            print('\n')
                
            plt.plot(localResidue)
            plt.title('Local Residue')
            plt.ylabel('error')
            plt.xlabel('epoch')
            plt.show()
                    
        return localResidue, acc_fe, acc, elapsed, alpha
    
    def initialize_alpha(self):
        if self.alpha_init is None:
            self.alpha_init = np.array([each * np.random.choice([-1, 1]) for each in np.random.randn(self.nyst_points)])
            return self.alpha_init  # Add this line to return the initialized alpha_init
        else:
            raise ValueError("nyst_points is not defined or alpha_init is already provided.")


    
    def initialize_fed_kernel(self, Xb, yb):
        ''' 
        Parameters
        -----------
        Xb: Data matrix for one batch   
        yb: Labels 
                
        Returns
        -----------
        Kty: Kernel matrix multiplied by labels array         
        '''
        W = self.W
        kernel_params = self.kernel_params
        ni = Xb.shape[1]         # number of features - one per omic centre
        for iff in range(ni,0,- 1):
                    Ki=kernel_params.build_kernel(Xb[:,iff-1:iff],W[:,iff-1:iff])
                
                    #update the gradient
                    if iff==ni:
                        Kty = np.matmul(Ki.T,np.diag(yb))
                    else:
                        Kty=np.multiply(Ki.T, Kty)
        return Kty
        
        
        
        
    # def build_Y_matrix(self, n, m, y):
    #     ''' 
    #     Parameters
    #     -----------
    #     n: integer, number of samples (and labels)
    #     m: integer, usually <= n, number of nystrom points
    #     y: an n x 1 array (usually labels)
        
    #     Returns
    #     -----------
    #     Y: n x m array, with y wrapping along the diagonal
    #     '''
    #     if (n<m):
    #         raise ValueError("n must be greater than or equal to m")
        
    #     stopBlock = int(-1)
    #     blocks = int(np.floor(n/m))
    #     for i in range(blocks):
    #         if (i==0):
    #             Y = np.diag(y[:m])
    #         else:
    #             startBlock = int((i)*m)
    #             stopBlock = int((i)*m+m -1)
    #             #temp = np.diag(y[startBlock:stopBlock+1])
    #             #print('temp shape: ', temp.shape)
    #             Y = np.concatenate([Y, np.diag(y[startBlock:stopBlock+1])], axis =0)            
    #     remainder = np.mod(n,m)
    #     if (remainder>0):
    #         Y = np.concatenate([Y, np.concatenate([np.diag(y[stopBlock+1:stopBlock+remainder+1]), np.zeros(( remainder,  m-remainder), dtype=int)], axis =1)], axis =0)
    #     #np.savetxt('matriceY.txt', Y, fmt='%1.0f')
    #     # np.set_printoptions(threshold=np.inf)
    #     # print('Y shape: ', Y.shape)        
    #     # print('Y: ', Y)
    #     return Y
    def build_Y_matrix(self, n, m, y):
        ''' 
        Parameters
        -----------
        n: integer, number of samples (and labels)
        m: integer, usually <= n, number of nystrom points
        y: an n x 1 array (usually labels)
        
        Returns
        -----------
        Y: n x m array, with y wrapping along the diagonal
        '''
        if (n<m):
            raise ValueError("n must be greater than or equal to m")

        num_blocks = n // m
        remainder = n % m

        Y = np.zeros((n, m))

        # Fill the diagonal blocks
        for i in range(num_blocks):
            start_idx = i * m
            end_idx = start_idx + m
            diag_block = np.diag(y[start_idx:end_idx])
            block_size = min(m, len(diag_block))
            Y[start_idx:end_idx, :block_size] = diag_block[:, :block_size]

        # Fill the last block with the remainder
        if remainder > 0:
            start_idx = num_blocks * m
            end_idx = start_idx + remainder
            diag_block = np.diag(y[start_idx:end_idx])
            block_size = min(remainder, len(diag_block))
            Y[start_idx:end_idx, :block_size] = diag_block[:, :block_size]

        return Y

    
    def feature_federated_gradient(self, Xb, sB, eB, YY, M, ETA_ETA, alpha, p):
        ''' 
        Parameters
        -----------
        Xb: Data matrix   
        sB, eB start and end of the batch with respect to total samples. Needed to mask y and get only the right labels.
        YY: labels wrapped on the diag in a MxMnys matrix
        ETA_ETA: noise

        Returns
        -----------
        KtKp
        v: Kalpha -y +eta
        '''
        Nb = self.Nb
        lam = self.lam
        kernel_params = self.kernel_params
        Mnys = self.nyst_points
        ni = Xb.shape[1]
        W = self.W
        # print('sB, eB: ', sB, eB)
        
        ## cycle on omic centres, assuming one feature per centre
        for iff in range(ni,0,- 1):
            Ki = kernel_params.build_kernel(Xb[:,iff-1:iff],W[:,iff-1:iff])
                  
            # Updating the gradient 
            if (iff == ni):
                Kalpha=np.matmul(Ki,np.diag(alpha))
                Kp=np.matmul(Ki, np.diag(p))

            else:
                Kalpha=np.multiply(Ki,Kalpha) #  aggregate different omic centres info 
                Kp=np.multiply(Ki,Kp)
        
        ## back on the server
        Kp=np.sum(Kp,axis=1)
        Ymasked = YY.copy()
        Ymasked[0:sB,:] = 0
        Ymasked[eB: , :] = 0
    

        #print("YY before masking:", YY)
        #print("Ymasked after masking:", Ymasked)

        Kalpha_embed = np.zeros((M,Mnys))         # build the embedding of the true Kalpha
        Kalpha_embed[sB:eB,:]=Kalpha.copy()
        #print('Kalpha_embed', Kalpha_embed)
        
        ETA = ETA_ETA.copy()      # mask the noise
        ETA[0:sB,:] = 0
        ETA[eB: , :] = 0
        # print('ETA', ETA)

        
        v = np.sum((Kalpha_embed - Ymasked + ETA), axis=1)       # defining v as reduction of
        ## second cycle on clients
        KtKp = []
        
        # simulates features federation backward
        for iff in range(ni,0,- 1):
            Ki = kernel_params.build_kernel(Xb[:,iff-1:iff],W[:,iff-1:iff]) #ottimizza questo per non calcolarlo cento volte
            if (iff == ni):
                KtKp=np.matmul(Ki.T,np.diag(Kp))
            else:
                KtKp=np.multiply(Ki.T,KtKp)
  
        ## back on the server
        KtKp = np.sum(KtKp,axis=1) + (lam/ Nb*p)
        return   KtKp, v
    
        
    def fit(self, X_train, y_train, X_test, y_test, W=None, alpha_init=None, silent=False):
        if alpha_init is None:
            alpha_init = self.initialize_alpha()  # Call initialize_alpha here to set alpha_init if None
        else:
             #print("alpha_init is already provided, ", alpha_init)
             #print("alpha_init is already provided.")
             pass
        if W is not None:
            self.W = W  # Store W as a class attribute
        else:
            raise ValueError("W must be provided for this method.")
 
        self.localResidue, self.central_test_accuracy, self.test_accuracy, self.elapsed, self.alpha = self.simulated_federated_cycles(X_train, y_train, X_test, y_test, alpha_init=alpha_init, silent=silent)
        return self.localResidue, self.central_test_accuracy, self.test_accuracy, self.elapsed, self.alpha


    def predict(self, X_pred):
        if self.alpha is None:
            raise ValueError("Alpha has not been trained. Please train the model first.")
        
        try:
            prediction = np.matmul(self.kernel_params.build_kernel(X_pred, self.W), self.alpha)
            return prediction
        except Exception as e:
            # Print or handle the error as needed
            print(f"An error occurred during prediction: {e}")
            return None  # or return an error code or default value
        
    def solve_linear(self, X, y, lam):
        # Compute the kernel matrix K
        K = self.kernel_params.build_kernel(X, self.W)
        
        # Compute K^T K
        KTK = np.matmul(K.T, K)
        
        # Compute K^T y
        KTy = np.matmul(K.T, y)
        
        # Add regularization term
        KTK += lam * np.eye(K.shape[1])  # Adding lambda times the identity matrix
        
        # Solve the linear system
        alphadir = np.linalg.solve(KTK, KTy)
        
        return alphadir, K



      
# kernel_params = Kernel(kernel_type=1, kernel_sigma = 1, poly_degree = None, binary = None)  
# fed_CG_model = FedCG(kernel_params=kernel_params, nyst_points = 50, lam=1e-6, Nb=10, toll=1e-3)
