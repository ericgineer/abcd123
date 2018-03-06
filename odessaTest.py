import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct
import scipy.stats as st
import odessa

plt.close('all')
            
def loadWavData(phrase, frameSize, skipSize, numCoef, numDataSets):
    # Load some training wav files to get MFCC training data
    FILENAME = "audio/" + phrase + "/" + phrase + "1.wav" # Name of wav file
    fs, wavData = scipy.io.wavfile.read(FILENAME)    
    mfccVect = odessa.mfcc.getMfcc(wavData, fs, frameSize, skipSize, numCoef)
    Dw = np.zeros((mfccVect.shape[0],mfccVect.shape[1],numDataSets))
    Dw[:,:,0] = mfccVect
    for i in range(1,numDataSets+1):
        FILENAME = "audio/" + phrase + "/" + phrase + str(i) + ".wav" # Name of wav file
        print("Reading wave file " + FILENAME)
        fs, wavData = scipy.io.wavfile.read(FILENAME)
        mfccVect = odessa.mfcc.getMfcc(wavData, fs, frameSize, skipSize, numCoef)
        Dw[:,:,i-1] = mfccVect
    return Dw
        
if __name__ == "__main__":
    """ MFCC parameters """
    frameSize = 25 # Length of the frame in milliseconds
    skipSize  = 10 # Time difference in milliseconds between the start of one frame 
                   # and the start of the next frame
    numCoef   = 13 # Number of MFCC coefficients
    
    numDataSets   = 10 
    
    
    # Load "Odessa" training data
    Dw1 = loadWavData("odessa", frameSize, skipSize, numCoef, numDataSets)
    OdessaMfcc = Dw1[:,:,0]
    
    # Initialize the "Odessa" HMM
    hmm1 = odessa.hmm(6, Dw1[:,:,0])
    
    # Train the "Odessa" HMM
    hmm1.train(Dw1)
    
    
    # Load "What time is it" training data
    Dw2 = loadWavData("WhatTimeIsIt", frameSize, skipSize, numCoef, numDataSets)
    WhatTimeIsItMfcc = Dw2[:,:,0]
    
    # Initialize the "What time is it" HMM
    hmm2 = odessa.hmm(40, Dw2[:,:,0])
    
    # Train the "What time is it" HMM
    hmm2.train(Dw2)
    
    
    # Load "Play music" training data
    Dw3 = loadWavData("PlayMusic", frameSize, skipSize, numCoef, numDataSets)
    PlayMusicMfcc = Dw3[:,:,0]
    
    # Initialize the "Play music" HMM
    hmm3 = odessa.hmm(40, Dw3[:,:,0])
    
    # Train the "Play music" HMM
    hmm3.train(Dw3)
    
    
    
    """ Use the "Odessa" HMM """
    probOdessa = []
    probWhatTimeIsIt = []
    probPlayMusic = []
    
    # Test with "Odessa"
    probOdessa = hmm1.probEvidence(OdessaMfcc)
    
    # Test with "What time is it"
    probWhatTimeIsIt = hmm1.probEvidence(WhatTimeIsItMfcc)
    
    # Test with "Play music"
    probPlayMusic = hmm1.probEvidence(PlayMusicMfcc)
    
    print("p(Odessa | Odessa): ",probOdessa)
    print("p(What time is it | Odessa): ",probWhatTimeIsIt)
    print("p(Play music | Odessa): ",probPlayMusic)
    print("")
    
    likelihoodArray = np.array([probOdessa,probWhatTimeIsIt,probPlayMusic])
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(likelihoodArray,'o')
    plt.title('Odessa HMM')
    
    
    """ Use the "What time is it" HMM """
    probOdessa = []
    probWhatTimeIsIt = []
    probPlayMusic = []
    
    # Test with "Odessa"
    probOdessa = hmm2.probEvidence(OdessaMfcc)
    
    # Test with "What time is it"
    probWhatTimeIsIt = hmm2.probEvidence(WhatTimeIsItMfcc)
    
    # Test with "Play music"
    probPlayMusic = hmm2.probEvidence(PlayMusicMfcc)
    
    print("p(Odessa | What time is it): ",probOdessa)
    print("p(What time is it | What time is it): ",probWhatTimeIsIt)
    print("p(Play music | What time is it): ",probPlayMusic)
    print("")
    
    likelihoodArray = np.array([probOdessa,probWhatTimeIsIt,probPlayMusic])
    
    plt.subplot(3,1,2)
    plt.plot(likelihoodArray,'o')
    plt.title('What time is it HMM')
    
    
    """ Use the "Play music" HMM """
    probOdessa = []
    probWhatTimeIsIt = []
    probPlayMusic = []
    
    # Test with "Odessa"
    probOdessa = hmm3.probEvidence(OdessaMfcc)
    
    # Test with "What time is it"
    probWhatTimeIsIt = hmm3.probEvidence(WhatTimeIsItMfcc)
    
    # Test with "Play music"
    probPlayMusic = hmm3.probEvidence(PlayMusicMfcc)
    
    print("p(Odessa | Play music): ",probOdessa)
    print("p(What time is it | Play music): ",probWhatTimeIsIt)
    print("p(Play music | Play music): ",probPlayMusic)
    print("")
    
    likelihoodArray = np.array([probOdessa,probWhatTimeIsIt,probPlayMusic])

    plt.subplot(3,1,3)
    plt.plot(likelihoodArray,'o')
    plt.title('Play music HMM')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    