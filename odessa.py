import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct
import scipy.stats as st

""" A class for implementing the MFCC and silence detector functions """
class mfcc:
    def __init__(self):
        """ Initialize silence detector variables """
        self.idx = 0
        self.thresholdEs = 0 
        self.frameIdx = 0 # A counter to keep track of how many frames we have processed
        self.frameAv = 10 # Number of frames to average for threshold calculation
        self.pastFrames = 15 # Number of past frames to use for threshold comparison
        self.pastEs = np.zeros(self.pastFrames)
        self.silenceDetectVect = []
        self.dataVect = np.zeros((self.frameSamples,1))
    
    """ Silence detection function """
    
    def silenceDetect(self, frame):        
        Es = sum(abs(frame)) # Calculate the energy in the frame
        # Update shift register that holds previous values of Es
        for i in range(0,self.pastFrames): 
            if i == self.pastFrames-1:
                self.pastEs[i] = Es
            else:
                self.pastEs[i] = self.pastEs[i+1]
        # Average the first <frameAv> frames with the assumption that there is 
        # no speech in them to get the threshold value
        if self.frameIdx < self.frameAv:
            self.thresholdEs += Es/10
            self.frameIdx += 1 # Update frame index counter
            return 0
        elif (self.pastEs > self.thresholdEs+self.thresholdEs).any() and self.frameIdx > self.pastFrames: 
            self.frameIdx += 1 # Update frame index counter
            return 1
        else:
            self.frameIdx += 1 # Update frame index counter
            return 0
    
    
    """ A function to compute delta features of an MFCC vector """
    def deltaFeature(x, M, numCoef):
        xhat = np.zeros((2 * numCoef,int(x[0,:].size)))
        for j in range(M,int(x[0,:].size)-M):
            xhat[:,j] = np.concatenate((x[:,j],np.sum(x[:,j-M:j+M+1],1)/sum(np.arange(-M,M+1)**2)))
        return xhat
        

    
    """ A function that computes Mel-frequency ceptral coefficients """
    def mfcc(x, fs, frameSize, skipSize, numCoef):
        """ Calculate MFCCs """
        startFreq = 0 # start frequency for Mel-scale calculation (Hz)
        stopFreq  = 8000 # stop frequency for Mel-scale calculation (Hz)
        numFilt   = 26   # Number of filter banks to use for the Mel-scale freq warping 
        
        # Compute the FFT size based on the frame size in ms
        nfft = int(float(fs)*float(frameSize)/1000.0)
        nSkip = int(float(fs)*float(skipSize)/1000.0)
        
        c = np.zeros((numCoef,int(x.size/nSkip)))
        count = 0
        # Iterate over the input signal
        for n in range(0, x.size-nfft, nSkip):
            # Window input signal
            xw = x[n:n+nfft] * np.hamming(nfft)
            # Compute the next power of 2 size for the fft
            next_power = 1
            my_pad = int(np.power(2,(next_power-1)+ np.ceil(np.log2(nfft))))
            # Transform to frequency domain and convert to magnitude
            X = np.abs(np.fft.rfft(xw, my_pad))
            # Compute frequency axis
            freq = np.arange(0,X.size)/X.size * fs/2
            # Convert start frequency to Mel-scale
            startMel = 1127 * np.log(1 + startFreq/700)
            # Convert the stop frequency to Mel-scale
            stopMel = 1127 * np.log(1 + stopFreq/700)
            mel = np.arange(startMel,stopMel,int((stopMel-startMel)/numFilt))
            # Convert the Mel-scale vector to frequency
            freqBins = 700 * (np.exp(mel/1127)-1)
            
            # Calculate Mel-scale filters
            filtBins = np.zeros(numFilt)
            idx = 0
            for m in range(1,numFilt+1):
                startBin = np.argmin(abs(freq-freqBins[m - 1]))
                if m == numFilt:
                    stopBin = freq.size
                else:
                    stopBin  = np.argmin(abs(freq-freqBins[m + 1]))
                # Average the FFT bins with a triangle window to get the Mel-scale bins
                # (Actually use a Bartlett window, which is a traingle window with the
                # first and last points set to 0)
                filtBins[idx] = np.log(np.sum(X[startBin:stopBin]*np.bartlett(stopBin-startBin))/(stopBin-startBin))    
                idx = idx + 1
                
                
            # Take the Inverse FFT of the Mel-scale vector to get MFCCs
            c[:,count] = dct(filtBins[0:numCoef])
            count += 1
            
        return c
    
    """ Call this function to get the mfcc data in the correct array form """
    def getMfcc(x, fs, frameSize, skipSize, numCoef):
        c = mfcc.mfcc(x, fs, frameSize, skipSize, numCoef)
        c = mfcc.deltaFeature(c, 2, numCoef)
        return c[:,2:-2]

class hmm:
    """ A class to implement a HMM for speech recognition """
    def __init__(self, numStates, mfcc):
        
        """ Initialize HMM variables """
        self.N = numStates  # Number of states to use in the HMM
                            # (should be the number of phonenems in the phrase)
                            
        self.n_states = numStates
        self.randState = np.random.RandomState(0)
        
        self.n_dims = []
        
        self.testGamma = []
        self.testXi = []
        
        self.covs = []
        
        # Initialize transition matrix with random probabilities
        self.A = np.zeros((self.N, self.N))
        for i in range(0,self.N):
            for j in range(0,self.N):
                if j < i:
                    self.A[i,j] = 0
                elif i == self.N-1 and j == self.N-1:
                    self.A[i,j] = 1
                elif i == 0 and j == 0:
                    self.A[i,j] = 0
                elif j > i + 1:
                    self.A[i,j] = 0
                else:
                    self.A[i,j] = np.random.rand()
        
        # Initialize state distribution vector
        self.prior = np.random.rand(self.N,1)
        self.prior /= max(self.prior)
        
        subset = self.randState.choice(np.arange(mfcc.shape[0]), size=self.N, replace=True)
        self.mu = mfcc[:, subset]
        
        # Initialize covariance matrix
        self.C = np.zeros((mfcc.shape[0], mfcc.shape[0], self.N))
        self.C += np.diag(np.diag(np.cov(mfcc)))[:, :, None]
        
        # Initialize the state likelihoood matrix p(x_t | Q_t = q)
        self.B = np.zeros((self.N, mfcc.shape[1]))

    """ Function to calculate the state likelihood matrix p(x_t | Q_t = q) """
    def stateLikelihood(self, obs):
        obs = np.atleast_2d(obs)
        for s in range(self.N):
            #Needs scipy 0.14
            np.random.seed(self.randState.randint(1))
            self.B[s, :] = st.multivariate_normal.pdf(
                obs.T, mean=self.mu[:, s].T, cov=self.C[:, :, s].T)
            #This function can (and will!) return values >> 1
            #See the discussion here for the equivalent matlab function
            #https://groups.google.com/forum/#!topic/comp.soft-sys.matlab/YksWK0T74Ak
            #Key line: "Probabilities have to be less than 1,
            #Densities can be anything, even infinite (at individual points)."
            #This is evaluating the density at individual points...
            
        return self.B
    
    """ Calculate the probability of evidence p(x_1:T) """
    def probEvidence(self, mfcc):
        p = hmm.stateLikelihood(self, mfcc)
        #alpha, beta, gamma, xi, logLikelihood = hmm.recursion(self, p)
        #pEvidence = np.log(np.sum(alpha*beta,axis=0))
        logLikelihood, alpha = hmm._forward(self, p)
        return logLikelihood


    """ Expectation-Maximization (EM) algorithm initializiation function """
    def emInit(self, mfcc):
        # Initialize transition matrix to be random with each row summing to 1
        M = np.random.rand(self.N,self.N)
        self.A = M/M.sum(axis=1)[:,None]
        
        subset = self.randState.choice(np.arange(mfcc.shape[0]), size=self.N, replace=True)
        self.mu = mfcc[:, subset]
    
        C = np.zeros((mfcc.shape[0], mfcc.shape[0], self.N))
        C += np.diag(np.diag(np.cov(mfcc)))[:, :, None]
        
        self.covs = np.zeros((mfcc.shape[0], mfcc.shape[0], self.N))
        self.covs += np.diag(np.diag(np.cov(mfcc)))[:, :, None]


    
    """ Calculate the forward recursion, backward recursion, gamma, and xi """
    def recursion(self, B):
        logLikelihood = 0.
        T = B.shape[1]
        alpha = np.zeros(B.shape)
        beta = np.zeros(B.shape);
        gamma = np.zeros(B.shape)
        xi = np.zeros((self.N,self.N))
        
        # First calculate beta
        beta[:, T-1] = np.ones(self.N)
        for t in range(T-2,-1,-1):
            beta[:, t] = np.sum(beta[:, t + 1] * B[:, t + 1] * self.A,axis=1)
            beta[:, t] /= np.sum(beta[:, t])
        
        # Then compute the rest of the parameters
        for t in range(T):
            if t == 0:
                alpha[:, t] = B[:, t] * self.prior.ravel()
            else:
                #alpha[:, t] = B[:, t] * np.sum(self.A * alpha[:, t - 1],axis=1)
                alpha[:, t] = B[:, t] * np.dot(self.A.T, alpha[:, t - 1])
            
            logLikelihood += np.log(np.sum(alpha[:, t]))
            alpha[:,t] /= np.sum(alpha[:,t])
            
            gamma[:,t] = alpha[:,t] * beta[:,t] / np.sum(alpha[:,t]*beta[:,t])
            
            #xiTmp[:,:,t] = B[:,t] * beta[:,t] * self.A * alpha[:,t-1]
            if t == 0:
                xiTmp = self.A * np.dot(self.prior.ravel(), (beta[:, t] * B[:, t]).T)
            else:
                xiTmp = self.A * np.dot(alpha[:, t-1], (beta[:, t] * B[:, t]).T)
            xi += self._normalize(xiTmp)
        return alpha, beta, gamma, xi, logLikelihood
    
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
    
    """ Expectation-Maximization algorithm """
    def em(self, Dw):
        w = Dw.shape[0]
        T = Dw.shape[1]
        if np.array(Dw.shape).size > 2:
            numDataSets = Dw.shape[2]
        else:
            numDataSets = 1
            
        for L in range(0, numDataSets):
            if np.array(Dw.shape).size > 2:            
                p = hmm.stateLikelihood(self, Dw[:,:,L])
            else:
                p = hmm.stateLikelihood(self, Dw)
            alpha, beta, gamma, xi, logLikelihood = hmm.recursion(self, p)
            #self.A = xi / np.sum(gamma,axis=0)
            self.A = hmm._stochasticize(self, xi)
            
#            # Zero out elements in matrix A to make the state transitions only
#            # left to right
#            for i in range(0,self.N):
#                for j in range(0,self.N):
#                    if j > i + 1 or j < i:
#                        self.A[i,j] = 0
            
#            for q in range(0,self.N): # iterate over the number of states
#                if np.array(Dw.shape).size > 2:
#                    self.mu[:,q] += np.sum(Dw[:,:,L]*gamma[q,:])/np.sum(gamma[q,:])
#                else:
#                    self.mu[:,q] += np.sum(Dw*gamma[q,:],axis=1)/np.sum(gamma[q,:])
#                    #self.mu[:,q] /= np.sum(self.mu[:,q]) # Normalize mu to avoid underflow
#                
#                if np.array(Dw.shape).size > 2:
#                    self.C[:,:,q] += np.diag(np.sum((Dw[:,:,L]-np.reshape(self.mu[:,q],(w,1)))*
#                            (Dw[:,:,L]-np.reshape(self.mu[:,q],(w,1)))*gamma[q,:],axis=1)/
#                            np.sum(gamma[q,:]))
#                else:
#                    self.C[:,:,q] += np.diag(np.sum((Dw-np.reshape(self.mu[:,q],(w,1)))*
#                            (Dw-np.reshape(self.mu[:,q],(w,1)))*gamma[q,:],axis=1)/
#                            np.sum(gamma[q,:]))
            
            gamma_state_sum = np.sum(gamma, axis=1)
            for q in range(self.N):
                gamma_obs = Dw * gamma[q, :]
                self.mu[:, q] = np.sum(gamma_obs, axis=1) / gamma_state_sum[q]
                #self.C = np.dot(gamma_obs, Dw.T) / gamma_state_sum[q] - np.dot(self.mu[:, q], self.mu[:, q].T)
                #Symmetrize
                #self.C = np.triu(self.C) + np.triu(self.C).T - np.diag(self.C)
                if np.array(Dw.shape).size > 2:
                    self.C[:,:,q] += np.diag(np.sum((Dw[:,:,L]-np.reshape(self.mu[:,q],(w,1)))*
                            (Dw[:,:,L]-np.reshape(self.mu[:,q],(w,1)))*gamma[q,:],axis=1)/
                            np.sum(gamma[q,:]))
                else:
                    self.C[:,:,q] += np.diag(np.sum((Dw-np.reshape(self.mu[:,q],(w,1)))*
                            (Dw-np.reshape(self.mu[:,q],(w,1)))*gamma[q,:],axis=1)/
                            np.sum(gamma[q,:]))
            # Ensure positive semidefinite by adding diagonal loading
            self.C = self.C + .01 * np.eye(Dw.shape[0])[:, :, None]
                
    def _normalize(self, x):
        return (x + (x == 0)) / np.sum(x)
    
    def _stochasticize(self, x):
        return (x + (x == 0)) / np.sum(x, axis=1)
                
    def _em_step(self, obs): 
        obs = np.atleast_2d(obs)
        B = self.stateLikelihood(obs)
        T = obs.shape[1]
        
        self.n_dims = obs.shape[0]
        
        #log_likelihood, alpha = self._forward(B)
        #beta = self._backward(B)
        
        alpha, beta, gamma, xi, log_likelihood = hmm.recursion(self, B)
        
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
        expected_covs = partial_covs + .01 * np.eye(self.n_states)[:, :, None]
        
#        # Zero out elements in matrix A to make the state transitions only
#        # left to right
#        for i in range(0,self.n_states):
#            for j in range(0,self.n_states):
#                if j > i + 1 or j < i:
#                    expected_A[i,j] = 0
        
        self.prior = expected_prior
        self.mu = expected_mu
        self.covs = expected_covs
        self.A = expected_A
        
        self.testGamma = gamma
        self.testXi = xi_sum
        
        
        return log_likelihood
    
    """ A function to train the HMM on a sequence of data """
    def train(self, Dw):
        if np.array(Dw.shape).size > 2:
            hmm.emInit(self, Dw[:,:,0])
        else:
            hmm.emInit(self, Dw)
        for i in range(0,15):
            print("Iteration number " + str(i))
            #hmm._em_step(self, Dw)
            hmm.em(self, Dw)