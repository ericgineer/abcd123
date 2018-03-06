import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct
import scipy.stats as st

class gmmhmm:
    #This class converted with modifications from https://code.google.com/p/hmm-speech-recognition/source/browse/Word.m
    def __init__(self, n_states):
        self.n_states = n_states
        self.random_state = np.random.RandomState(0)
        
        #Normalize random initial state
        self.prior = self._normalize(self.random_state.rand(self.n_states, 1))
        # Initialize transition matrix with random probabilities
        self.A = np.zeros((self.n_states, self.n_states))
        for i in range(0,self.n_states):
            for j in range(0,self.n_states):
                if j < i:
                    self.A[i,j] = 0
                elif i == self.n_states-1 and j == self.n_states-1:
                    self.A[i,j] = 1
                elif i == 0 and j == 0:
                    self.A[i,j] = 0
                elif j > i + 1:
                    self.A[i,j] = 0
                else:
                    self.A[i,j] = np.random.rand()
        
        self.mu = None
        self.covs = None
        self.n_dims = None
           
    def _forward(self, B):
        log_likelihood = 0.
        T = B.shape[1]
        alpha = np.zeros(B.shape)
        for t in range(T):
            if t == 0:
                alpha[:, t] = B[:, t] * self.prior.ravel()
            else:
                alpha[:, t] = B[:, t] * np.dot(self.A.T, alpha[:, t - 1])
         
            alpha_sum = np.sum(alpha[:, t])
            alpha[:, t] /= alpha_sum
            log_likelihood = log_likelihood + np.log(alpha_sum)
        return log_likelihood, alpha
    
    def _backward(self, B):
        T = B.shape[1]
        beta = np.zeros(B.shape);
           
        beta[:, -1] = np.ones(B.shape[0])
            
        for t in range(T - 1)[::-1]:
            beta[:, t] = np.dot(self.A, (B[:, t + 1] * beta[:, t + 1]))
            beta[:, t] /= np.sum(beta[:, t])
        return beta
    
    def _state_likelihood(self, obs):
        obs = np.atleast_2d(obs)
        B = np.zeros((self.n_states, obs.shape[1]))
        for s in range(self.n_states):
            #Needs scipy 0.14
            np.random.seed(self.random_state.randint(1))
            B[s, :] = st.multivariate_normal.pdf(
                obs.T, mean=self.mu[:, s].T, cov=self.covs[:, :, s].T)
            #This function can (and will!) return values >> 1
            #See the discussion here for the equivalent matlab function
            #https://groups.google.com/forum/#!topic/comp.soft-sys.matlab/YksWK0T74Ak
            #Key line: "Probabilities have to be less than 1,
            #Densities can be anything, even infinite (at individual points)."
            #This is evaluating the density at individual points...
        return B
    
    def _normalize(self, x):
        return (x + (x == 0)) / np.sum(x)
    
    def _stochasticize(self, x):
        return (x + (x == 0)) / np.sum(x, axis=1)
    
    def _em_init(self, obs):
        #Using this _em_init function allows for less required constructor args
        if self.n_dims is None:
            self.n_dims = obs.shape[0]
        if self.mu is None:
            subset = self.random_state.choice(np.arange(self.n_dims), size=self.n_states, replace=False)
            self.mu = obs[:, subset]
        if self.covs is None:
            self.covs = np.zeros((self.n_dims, self.n_dims, self.n_states))
            self.covs += np.diag(np.diag(np.cov(obs)))[:, :, None]
        return self
    
    def _em_step(self, obs): 
        obs = np.atleast_2d(obs)
        B = self._state_likelihood(obs)
        T = obs.shape[1]
        
        log_likelihood, alpha = self._forward(B)
        beta = self._backward(B)
        
        xi_sum = np.zeros((self.n_states, self.n_states))
        gamma = np.zeros((self.n_states, T))
        
        for t in range(T - 1):
            partial_sum = self.A * np.dot(alpha[:, t], (beta[:, t] * B[:, t + 1]).T)
            xi_sum += self._normalize(partial_sum)
            partial_g = alpha[:, t] * beta[:, t]
            gamma[:, t] = self._normalize(partial_g)
              
        partial_g = alpha[:, -1] * beta[:, -1]
        gamma[:, -1] = self._normalize(partial_g)
        
        expected_prior = gamma[:, 0]
        expected_A = self._stochasticize(xi_sum)
        
        expected_mu = np.zeros((self.n_dims, self.n_states))
        expected_covs = np.zeros((self.n_dims, self.n_dims, self.n_states))
        
        gamma_state_sum = np.sum(gamma, axis=1)
        #Set zeros to 1 before dividing
        gamma_state_sum = gamma_state_sum + (gamma_state_sum == 0)
        
        for s in range(self.n_states):
            gamma_obs = obs * gamma[s, :]
            expected_mu[:, s] = np.sum(gamma_obs, axis=1) / gamma_state_sum[s]
            partial_covs = np.dot(gamma_obs, obs.T) / gamma_state_sum[s] - np.dot(expected_mu[:, s], expected_mu[:, s].T)
            #Symmetrize
            partial_covs = np.triu(partial_covs) + np.triu(partial_covs).T - np.diag(partial_covs)
        
        #Ensure positive semidefinite by adding diagonal loading
        expected_covs += .01 * np.eye(self.n_dims)[:, :, None]
        
        # Zero out elements in matrix A to make the state transitions only
        # left to right
        for i in range(0,self.n_states):
            for j in range(0,self.n_states):
                if j > i + 1 or j < i:
                    self.A[i,j] = 0
        
        self.prior = expected_prior
        self.mu = expected_mu
        self.covs = expected_covs
        self.A = expected_A
        return log_likelihood
    
    def fit(self, obs, n_iter=15):
        #Support for 2D and 3D arrays
        #2D should be n_features, n_dims
        #3D should be n_examples, n_features, n_dims
        #For example, with 6 features per speech segment, 105 different words
        #this array should be size
        #(105, 6, X) where X is the number of frames with features extracted
        #For a single example file, the array should be size (6, X)
        if len(obs.shape) == 2:
            for i in range(n_iter):
                self._em_init(obs)
                log_likelihood = self._em_step(obs)
        elif len(obs.shape) == 3:
            count = obs.shape[0]
            for n in range(count):
                for i in range(n_iter):
                    self._em_init(obs[n, :, :])
                    log_likelihood = self._em_step(obs[n, :, :])
        return self
    
    def transform(self, obs):
        #Support for 2D and 3D arrays
        #2D should be n_features, n_dims
        #3D should be n_examples, n_features, n_dims
        #For example, with 6 features per speech segment, 105 different words
        #this array should be size
        #(105, 6, X) where X is the number of frames with features extracted
        #For a single example file, the array should be size (6, X)
        if len(obs.shape) == 2:
            B = self._state_likelihood(obs)
            log_likelihood, _ = self._forward(B)
            return log_likelihood
        elif len(obs.shape) == 3:
            count = obs.shape[0]
            out = np.zeros((count,))
            for n in range(count):
                B = self._state_likelihood(obs[n, :, :])
                log_likelihood, _ = self._forward(B)
                out[n] = log_likelihood
            return out
        
    def hmm(self, obs):
        B = self._state_likelihood(obs)
        log_likelihood, alpha1 = self._forward(B)
        beta1 = self._backward(B)
        p = np.sum(beta1*alpha1,0)
        return p

if __name__ == "__main__":
    """ MFCC parameters """
    frameSize = 25 # Length of the frame in milliseconds
    skipSize  = 10 # Time difference in milliseconds between the start of one frame 
                   # and the start of the next frame
    numCoef   = 13 # Number of MFCC coefficients
    
    ydim = 5
    xdim = 5
    
    numDataSets   = 1 
    
    numStates = 5
    
    # Load "Odessa" training data
    #Dw1 = loadWavData("odessa", frameSize, skipSize, numCoef, numDataSets)
    #OdessaMfcc = Dw1[:,:,0]
    np.random.seed(0)
    Dw1 = np.random.rand(xdim, ydim)
    
    # Initialize the "Odessa" HMM
    hmm1 = gmmhmm(numStates)
    
    # Train the "Odessa" HMM
    hmm1.fit(Dw1)
    
    
    # Load "What time is it" training data
    #Dw2 = loadWavData("WhatTimeIsIt", frameSize, skipSize, numCoef, numDataSets)
    #WhatTimeIsItMfcc = Dw2[:,:,0]
    np.random.seed(1)
    Dw2 = np.random.rand(xdim, ydim)
    
    # Initialize the "What time is it" HMM
    hmm2 = gmmhmm(numStates)
    
    # Train the "What time is it" HMM
    hmm2.fit(Dw2)
    
    
    # Load "Play music" training data
    #Dw3 = loadWavData("PlayMusic", frameSize, skipSize, numCoef, numDataSets)
    #PlayMusicMfcc = Dw3[:,:,0]
    np.random.seed(3)
    Dw3 = np.random.rand(xdim, ydim)
    
    # Initialize the "Play music" HMM
    hmm3 = gmmhmm(numStates)
    
    # Train the "Play music" HMM
    hmm3.fit(Dw3)
    
    
    """ Test HMMs """
    
    mfccIn = Dw1
    
     # Test with "Odessa"
    probOdessa = hmm1.transform(mfccIn)
    
    # Test with "What time is it"
    probWhatTimeIsIt = hmm2.transform(mfccIn)
    
    # Test with "Play music"
    probPlayMusic = hmm3.transform(mfccIn)
    
    print("p(Odessa | Odessa): ",probOdessa)
    print("p(What time is it | Odessa): ",probWhatTimeIsIt)
    print("p(Play music | Odessa): ",probPlayMusic)
    print("")
    
    likelihoodArray = np.array([probOdessa,probWhatTimeIsIt,probPlayMusic])
    
    plt.figure()
    plt.plot(likelihoodArray,'o')
    plt.title('Odessa HMM')