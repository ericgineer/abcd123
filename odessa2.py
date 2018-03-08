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
        self.random_state = np.random.RandomState(0)
        
        """ Initialize HMM variables """
        self.Q = numStates  # Number of states to use in the HMM
                            # (should be the number of phonenems in the phrase)
                            
        self.numCoef = mfcc.shape[0] # Number of MFCC coefficients
        self.T       = mfcc.shape[1] # Number of MFCC vectors
        
        # Initialize transition matrix with random probabilities
        self.A = np.zeros((self.Q, self.Q))
        for i in range(0,self.Q):
            for j in range(0,self.Q):
                if j < i:
                    self.A[i,j] = 0
                elif i == self.Q-1 and j == self.Q-1:
                    self.A[i,j] = 1
                elif i == 0 and j == 0:
                    self.A[i,j] = 0
                elif j > i + 1:
                    self.A[i,j] = 0
                else:
                    self.A[i,j] = np.random.rand()
        
        # Initialize mu
        self.mu = np.zeros((self.numCoef,self.Q))
        self.randState = np.random.RandomState(0)
        subset = self.randState.choice(np.arange(self.numCoef), size=self.Q, replace=True)
        self.mu = mfcc[:, subset]
        
        self.prior = self.normalize(self.random_state.rand(self.Q, 1))
        
        # Initialize covariance matrix
#        self.C = np.zeros((self.numCoef, self.Q))
#        for q in range(self.Q):
#            for c in range(self.numCoef):
#                self.C[c,q] = np.mean(mfcc[c,:])/np.max(np.abs(mfcc[c,:])) + 0.01 * np.random.rand()
        self.C = np.zeros((self.numCoef, self.numCoef, self.Q))
        self.C += np.diag(np.diag(np.cov(mfcc)))[:, :, None]
        
        # Initialize the state likelihoood matrix p(x_t | Q_t = q)
        self.B = np.zeros((self.Q, mfcc.shape[1]))
        
        self.alphaPrev = np.random.rand(self.Q)
        self.alphaPrev /= np.sum(self.alphaPrev)
        self.alpha = np.zeros((self.Q,self.T))
        
        
        self.beta = np.zeros((self.Q,self.T))
        
        self.gamma = np.zeros((self.Q, self.T))
        
        self.xi = np.zeros((self.Q,self.Q,self.T))

    """ Function to calculate the path weight matrix p(x_t | Q_t = q) """
    def pathWeights(self, x):
        for t in range(self.T):
            for q in range(self.Q):
                exponent = 0
                for c in range(self.numCoef):
                    exponent += (x[c,t]-self.mu[c,q]) * 1/self.C[c,c,q] * (x[c,t]-self.mu[c,q])
                self.B[q,t] = 1/np.sqrt(np.linalg.det(2*np.pi*self.C[:,:,q]))*np.exp(-1/2 * exponent)
        return self.B
    
    """ Calculate the probability of evidence p(x_1:T) """
    def probEvidence(self, mfcc):
        B = hmm.pathWeights(self, mfcc)
        alpha, logLikelihood = hmm.alphaRecursion(self, B)
        beta = hmm.betaRecursion(self, B)
        pEvidence = np.zeros(self.T)
        for t in range(self.T):
            for q in range(self.Q):
                pEvidence[t] += alpha[q,t]*beta[q,t]
        return pEvidence,logLikelihood

   
    """ Calculate the forward recursion, backward recursion, gamma, and xi """
    def alphaRecursion(self, B):
        alpha = np.zeros((self.Q,self.T))
        alpha[:,0] = self.alphaPrev #np.random.rand(self.Q)
        alpha[:,0] /= np.sum(alpha[:,0])
        for t in range(1,self.T):
            alpha[:,t] = B[:,t] * np.dot(self.A, alpha[:,t-1])
            logLikelihood = np.sum(alpha[:,t])
            alpha[:,t] /= np.sum(alpha[:,t])
        return logLikelihood, alpha
    
    def betaRecursion(self, B):
        beta = np.zeros(B.shape)
        beta[:, self.T-1] = np.ones(self.Q)
        for t in range(self.T-2,-1,-1):
            beta[:, t] = np.sum(beta[:, t + 1] * B[:, t + 1] * self.A,axis=1)
            beta[:, t] /= np.sum(beta[:, t])
        return beta
    
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
    
        
    def normalize(self, x):
        return (x + (x == 0)) / np.sum(x)
    
    def stochasticize(self, x):
        return (x + (x == 0)) / np.sum(x, axis=1)
    
    """ Expectation-Maximization algorithm """
    def em(self, x):        
        B = hmm.pathWeights(self, x)    
        
        A = self.A
        #logLikelihood, alpha = hmm.alphaRecursion(self, B)
        #beta = hmm.betaRecursion(self, B)
        logLikelihood, alpha = hmm._forward(self, B)
        beta = hmm._backward(self, B)
        
        
        # Calculate gamma
        gamma = np.zeros((self.Q, self.T))
        for t in range(self.T):
            for q in range(self.Q):
                gammaSum = 0
                for qq in range(self.Q):
                    gammaSum += alpha[qq,t] * beta[qq,t]
                gamma[q,t] = alpha[q,t]*beta[q,t] / gammaSum
            gamma[:,t] = hmm.normalize(self, gamma[:,t])
        
        # Calculate xi
        pEvidence = np.zeros(self.T)
        for t in range(self.T):
            for q in range(self.Q):
                pEvidence[t] += alpha[q,t]*beta[q,t]
        
        xi = np.zeros((self.Q,self.Q,self.T))
        xi[:,:,0] = np.dot(self.alphaPrev, (beta[:,t] * B[:,t]).T) * A / pEvidence[0]
        xi[:,:,0] = hmm.normalize(self, xi[:,:,0])
        for t in range(1,self.T):
            xi[:,:,t] = np.dot(alpha[:,t-1], (beta[:,t] * B[:,t]).T) * A / pEvidence[t]
            xi[:,:,t] = hmm.normalize(self, xi[:,:,t])
            
        # Update A
        for i in range(self.Q):
            for j in range(self.Q):
                numA = 0
                denA = 0
                for t in range(self.T):
                    numA += xi[i,j,t]
                    denA += gamma[i,t]
                A[i,j] = numA/denA
        
        # Zero out un-needed A elements
        for i in range(0,self.Q):
            for j in range(0,self.Q):
                if j < i:
                    A[i,j] = 0
                elif i == 0 and j == 0:
                    A[i,j] = 0
                elif j > i + 1:
                    A[i,j] = 0

                
                    
        # Update mu
        mu = np.zeros((self.numCoef,self.Q))
        for q in range(self.Q):
            numMu = 0
            denMu = 0
            for t in range(self.T):
                numMu += x[:,t] *  gamma[q,t]
                denMu += gamma[q,t]
            mu[:,q] = numMu / denMu
            
        # Update C
        C = np.zeros((self.numCoef, self.numCoef, self.Q))
        for q in range(self.Q):
            numC = 0
            denC = 0
            for t in range(self.T):
                numC += (x[:,t]-mu[:,q]) * (x[:,t]-mu[:,q]) * gamma[q,t]
                denC += gamma[q,t]
            for c in range(self.numCoef):
                C[c,c,q] = numC[c] / denC
                
        # Update state variables
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.xi   = xi
        self.A     = A
        self.mu    = mu
        self.C     = C
        
        return logLikelihood
    
    
                
   
    
    """ A function to train the HMM on a sequence of data """
    def train(self, Dw, numIter):
        conv = np.zeros(numIter)
        for i in range(0,numIter):
            conv[i] = hmm.em(self, Dw)
        return conv
            