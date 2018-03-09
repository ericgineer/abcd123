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
#    """ MFCC parameters """
#    frameSize = 25 # Length of the frame in milliseconds
#    skipSize  = 10 # Time difference in milliseconds between the start of one frame 
#                   # and the start of the next frame
#    numCoef   = 13 # Number of MFCC coefficients
#    
#    numDataSets   = 1 
#    
#    ydim = 5
#    xdim = 5
#    
#    numDataSets   = 1 
#    
#    numStates = 10
#    
#    numIter = 2 # number of EM algorithm iterations
#    
#    
#    """ HMM 1 """
#    
#    # Load "Odessa" training data
#    #Dw1 = loadWavData("odessa", frameSize, skipSize, numCoef, numDataSets)
#    #OdessaMfcc = Dw1[:,:,0]
#    np.random.seed(0)
#    d1 = np.random.rand(xdim, ydim)
#    
#    # Initialize the "Odessa" HMM
#    hmm1 = odessa2.hmm(numStates, d1)
#    
#    # Train the "Odessa" HMM
#    conv1 = hmm1.train(d1, numIter)
#    
#    """ HMM 1 """
#    
#    # Load "Odessa" training data
#    #Dw1 = loadWavData("odessa", frameSize, skipSize, numCoef, numDataSets)
#    #OdessaMfcc = Dw1[:,:,0]
#    d2 = np.random.rand(xdim, ydim)
#    
#    # Initialize the "Odessa" HMM
#    hmm2 = odessa2.hmm(numStates, d2)
#    
#    # Train the "Odessa" HMM
#    conv2 = hmm2.train(d2, numIter)
#    
#    
#    data = d1
#    
#    prob1, ll1 = hmm1.probEvidence(data)
#    prob2, ll2 = hmm2.probEvidence(data)
#    
#    print("hmm1: ",ll1)
#    print("hmm2: ",ll2)
    
    numStates = 2
    
    numIter = 15
    
    rstate = np.random.RandomState(0)
    t1 = np.ones((4, 40)) + .001 * rstate.rand(4, 40)
    t1 /= t1.sum(axis=0)
    t2 = rstate.rand(*t1.shape)
    t2 /= t2.sum(axis=0)
    
    m1 = odessa2.hmm(numStates, t1)
    conv1 = m1.train(t1, numIter)
    m2 = odessa2.hmm(numStates, t2)
    conv2 = m2.train(t2, numIter)
    
    p11, m1t1 = m1.probEvidence(t1)
    p21, m2t1 = m2.probEvidence(t1)
    print("Likelihoods for test set 1")
    print("M1:", m1t1)
    print("M2:", m2t1)
    print("Prediction for test set 1")
    print("Model", np.argmax([m1t1, m2t1]) + 1)
    print()
    
    p12, m1t2 = m1.probEvidence(t2)
    p22, m2t2 = m2.probEvidence(t2)
    print("Likelihoods for test set 2")
    print("M1:", m1t2)
    print("M2:", m2t2)
    print("Prediction for test set 2")
    print("Model", np.argmax([m1t2, m2t2]) + 1)
    
    plt.figure()
    plt.plot(conv1[1:conv1.size])
    plt.title('HMM1 convergence')
    
    plt.figure()
    plt.plot(conv2)
    plt.title('HMM2 convergence')
    
    alpha = m1.alpha
    beta  = m1.beta
    gamma = m1.gamma
    xi    = m1.xi
    A     = m1.A
    mu    = m1.mu
    C     = m1.C
    xiSum = m1.xiSum
    