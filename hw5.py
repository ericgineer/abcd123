import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct
import scipy.stats as st

plt.close('all')

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
        self.dataVect = np.zeros((frameSamples,1))
    
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

class hmm:
    """ A class to implement a HMM for speech recognition """
    def __init__(self, numStates, mfcc):
        
        """ Initialize HMM variables """
        self.N = numStates  # Number of states to use in the HMM
                            # (should be the number of phonenems in the phrase)
        self.randState = np.random.RandomState(0)
        
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
        for s in range(N):
            #Needs scipy 0.14
            np.random.seed(self.random_state.randint(1))
            self.B[s, :] = st.multivariate_normal.pdf(
                obs.T, mean=self.mu[:, s].T, self.cov=C[:, :, s].T)
            #This function can (and will!) return values >> 1
            #See the discussion here for the equivalent matlab function
            #https://groups.google.com/forum/#!topic/comp.soft-sys.matlab/YksWK0T74Ak
            #Key line: "Probabilities have to be less than 1,
            #Densities can be anything, even infinite (at individual points)."
            #This is evaluating the density at individual points...
            
        return B



    """ Expectation-Maximization (EM) algorithm initializiation function """
    def emInit(self, mfcc):
        global N
        global A
        global prior
        global mu
        global C
    
        # Initialize transition matrix to be random with each row summing to 1
        M = np.random.rand(N,N)
        A = M/M.sum(axis=1)[:,None]
        # Initialize state distribution vector
        prior = np.random.rand(N,1)
        prior /= max(prior)
        
        subset = self.random_state.choice(np.arange(mfcc.shape[0]), size=N, replace=True)
        mu = mfcc[:, subset]
    
        C = np.zeros((mfcc.shape[0], mfcc.shape[0], N))
        C += np.diag(np.diag(np.cov(mfcc)))[:, :, None]


    
    """ Calculate the forward recursion, backward recursion, gamma, and xi """
    def recursion(self, p):
        T = p.shape[1]
        alpha = np.zeros(p.shape)
        beta = np.zeros(p.shape);
        gamma = np.zeros(p.shape)
        xi = np.zeros((self.N,self.N,T))
        for t in range(T):
            if t == 0:
                alpha[:, t] = p[:, t] * self.prior.ravel()
            else:
                alpha[:, t] = p[:, t] * np.dot(self.A.T, alpha[:, t - 1])
                
            alpha[:,t] /= np.sum(alpha[:,t])
            
            beta[:, t] = np.dot(A, (p[:, t + 1] * beta[:, t + 1]))
            beta[:, t] /= np.sum(beta[:, t])
            
            gamma[:,t] = alpha[:,t] * beta[:,t] / np.sum(alpha[:,t]*beta[:,t])
            
            for i in range(0,self.N):
                for j in range(0,self.N):
                    xi[i,j,t] = p[:,t] * beta[:,t] * self.A[i,j] * 
            
        return alpha, beta, gamma
    
    def normalize(x):
        return (x + (x == 0)) / np.sum(x)
    
    def stochasticize(x):
        return (x + (x == 0)) / np.sum(x, axis=1)
        
    

if __name__ == "__main__":
    """ WAV file parameters """
    #fs = 16000 # Sample rate (Hz)
    frameSize = 10 # Frame size in ms
    frameSamples = int(frameSize/1000 * fs)
    CHANNELS = 1
    
    """ MFCC parameters """
    frameSize = 25 # Length of the frame in milliseconds
    skipSize  = 10 # Time difference in milliseconds between the start of one frame 
                   # and the start of the next frame
    numCoef   = 13 # Number of MFCC coefficients
    
    # Load some training MFCC data
    trainData = np.loadtxt('MFCC/odessa.csv',delimiter=',')
    trainData = trainData[:,2:trainData.shape[1]-2]
    
    # Load a wav file
    FILENAME = "audio/odessa.wav" # Name of wav file
    fs, wavData = scipy.io.wavfile.read(FILENAME)
    
    mfccVect = mfcc.mfcc(wavData, fs, frameSize, skipSize, numCoef)
    
#    emInit(data)
#    
#    B = stateLikelihood(data)
#    
#    log_likelihood, alpha = forwardRec(B)
#    beta = backwardRec(B)
    
    
    #log_likelihood = train(data)
    
    hmm1 = hmm(10, mfccVect)
    A = hmm1.A
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    