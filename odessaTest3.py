import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct
import scipy.stats as st
import odessa2

plt.close('all')
            
def loadWavData(phrase, frameSize, skipSize, numCoef, numDataSets):
    # Load some training wav files to get MFCC training data
    FILENAME = "audio/" + phrase + "/" + phrase + "1.wav" # Name of wav file
    fs, wavData = scipy.io.wavfile.read(FILENAME)    
    mfccVect = odessa2.mfcc.getMfcc(wavData, fs, frameSize, skipSize, numCoef)
    if numDataSets > 1:
        Dw = np.zeros((mfccVect.shape[0],mfccVect.shape[1],numDataSets))
        Dw[:,:,0] = mfccVect
        for i in range(1,numDataSets+1):
            FILENAME = "audio/" + phrase + "/" + phrase + str(i) + ".wav" # Name of wav file
            print("Reading wave file " + FILENAME)
            fs, wavData = scipy.io.wavfile.read(FILENAME)
            mfccVect = odessa2.mfcc.getMfcc(wavData, fs, frameSize, skipSize, numCoef)
            Dw[:,:,i-1] = mfccVect
    else:
        Dw = mfccVect
    return Dw
        
if __name__ == "__main__":
    """ MFCC parameters """
    frameSize = 25 # Length of the frame in milliseconds
    skipSize  = 10 # Time difference in milliseconds between the start of one frame 
                   # and the start of the next frame
    numCoef   = 13 # Number of MFCC coefficients
    
    numDataSets   = 1 
    
    ydim = 10
    xdim = 10
    
    numDataSets   = 1 
    
    numStates = 10
    
    numIter = 10 # number of EM algorithm iterations
    
    
    """ HMM 1 """
    
    # Load "Odessa" training data
    #Dw1 = loadWavData("odessa", frameSize, skipSize, numCoef, numDataSets)
    #OdessaMfcc = Dw1[:,:,0]
    np.random.seed(0)
    d1 = np.random.rand(xdim, ydim)
    
    # Initialize the "Odessa" HMM
    hmm1 = odessa2.hmm(numStates, d1)
    
    # Train the "Odessa" HMM
    conv1 = hmm1.train(d1, numIter)
    
    """ HMM 1 """
    
    # Load "Odessa" training data
    #Dw1 = loadWavData("odessa", frameSize, skipSize, numCoef, numDataSets)
    #OdessaMfcc = Dw1[:,:,0]
    np.random.seed(0)
    d2 = np.random.rand(xdim, ydim)
    
    # Initialize the "Odessa" HMM
    hmm2 = odessa2.hmm(numStates, d2)
    
    # Train the "Odessa" HMM
    conv2 = hmm2.train(d2, numIter)
    
    
    data = d1
    
    prob1, ll1 = hmm1.probEvidence(data)
    prob2, ll2 = hmm2.probEvidence(data)
    
    print("hmm1: ",ll1)
    print("hmm2: ",ll2)
    
    
    
#    plt.figure()
#    plt.plot(conv)
#    plt.title('EM algorithm convergence')
#    
#    alpha = hmm1.alpha
#    beta  = hmm1.beta
#    gamma = hmm1.gamma
#    xi    = hmm1.xi
#    A     = hmm1.A
#    mu    = hmm1.mu
#    C     = hmm1.C
    