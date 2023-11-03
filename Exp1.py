import numpy as np
import math
import matplotlib.pyplot as plt
import numpy as np

def fft(data, nn, isign):
    n = nn
    data = np.array(data, dtype=complex) #For imaginary numbers

    if isign == -1:  #Forward FFT
        if n > 1:
            evenPoints = fft(data[0::2], n // 2, isign) #Recursive split of data points for even
            oddPoints = fft(data[1::2], n // 2, isign)  #and odd data points

            w = np.exp(-2j * np.pi * np.arange(n) / n) # e^(-2pij * k/N)
            half = n // 2   #Truncate decimals

            return np.concatenate([evenPoints + w[:half] * oddPoints, evenPoints + w[half:] * oddPoints])
        else:
            return data
    
    elif isign == 1:  #Inverse FFT
        data = np.conj(data)        #Perform Complex conjugate 
        result = fft(data, nn, -1)  #Recall FFT

        result = np.round(np.conj(result) / n, 10) #Normalize with N, round to 10 so we get our original data back in our real numbers

        return result
    
def plotDFT(data, title, xlabel, ylabel, filename):
    graph = plt.figure()    #Different graph every time
    plt.title(title)
    plt.plot(np.arange(data.size), data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename + ".png")

def magShift(data):
    n = len(data)
    half = n // 2
    return np.concatenate((data[half:], data[:half]))

def partA():
    data = [2, 3, 4, 4]
    nn = len(data)

    forward = fft(data, nn, -1)    #For FFT
    magnitude = np.abs(forward)
    #print("Data going in:", data)    
    #print("Data after fft:", forward)
    #print("Magnitude of the DFT:", magnitude)

    #Plot DFT components REAL, IMAGINARY, and MAGNITUDE
    plotDFT(forward.real, "FFT Real Component", "Index", "Real", "fftReal")
    plotDFT(forward.imag, "FFT Imaginary Component", "Index", "Imaginary", "fftImaginary")
    plotDFT(magnitude, "FFT Manitude", "Index", "Manitude", "fftManitude")
    inverse = fft(forward, nn, 1)    #For IFFT
    print("Data after ifft:", inverse)

def partB():
    nn = 128    # N = 128 samples
    u = 8       # u = 8 cycles per period
    x = np.arange(nn)
    cosWave = np.cos(2 * np.pi * u * x / nn)    #Our cosine wave

    #Verify that we have a cosine wave:
    plotDFT(cosWave, "Original Cosine Wave", "Samples(128)", "Amplitude", "originalCos")

    #Compute fft of cosine wave
    forward = fft(cosWave, nn, -1)

    #Verify that we get the same wave back:
    inverse = fft(forward, nn, 1)
    plotDFT(inverse.real, "Inverse Cosine Wave", "Samples(128", "Amplitude", "inverseCos")

    #Shift the magnitude to center of frequency domain:
    forward = magShift(forward)

    #Phase and Magnitude of forward:
    magnitude = np.abs(forward)
    phase = np.angle(forward)

    #Plot the REAL, IMAGINARY, MANITUDE, and PHASE components of cosine wave:
    plotDFT(forward.real, "Cosine Real Component", "Samples(128)", "Real", "cosRealComp")
    plotDFT(forward.imag, "Cosine Imaginary Component", "Samples(128)", "Imaginary", "cosImaginaryComp")
    plotDFT(magnitude, "Cosine Magnitude Component", "Samples(128)", "Magnitude", "cosMagnitudeComp")
    plotDFT(phase, "Cosine Phase Component", "Samples(128)", "Phase", "cosPhaseComp")

def partC():
    #Read in Rect_128.dat file:
    with open("Rect_128.dat", 'r') as file:
        rect = [int(float(line.strip())) for line in file]   #File holds floats for no reason so we strip them
    
    #Perform same experiment as PartB:
    rect = np.array(rect)
    nn = len(rect)

    #Verify that we have the data from Rect_128:
    plotDFT(rect, "Original Rect_128.dat", "Samples(128)", "Value", "originalRectPlot")

    #Compute fft of Rect_128
    forward = fft(rect, nn, -1)

    #Verify that we get the same Rect_128 data back:
    inverse = fft(forward, nn, 1)
    plotDFT(inverse.real, "Inverse Rect_128.dat", "Samples(128", "Value", "inverseRectPlot")

    #Shift the magnitude to center of frequency domain:
    forward = magShift(forward)

    #Phase and Magnitude of forward:
    magnitude = np.abs(forward)
    phase = np.angle(forward)

    #Plot the REAL, IMAGINARY, MANITUDE, and PHASE components of Rect_128:
    plotDFT(forward.real, "Rect_128.dat Real Component", "Samples(128)", "Real", "rect128RealComp")
    plotDFT(forward.imag, "Rect_128.dat Imaginary Component", "Samples(128)", "Imaginary", "rect128ImaginaryComp")
    plotDFT(magnitude, "Rect_128.dat Magnitude Component", "Samples(128)", "Magnitude", "rect128MagnitudeComp")
    plotDFT(phase, "Rect_128.dat Phase Component", "Samples(128)", "Phase", "rect128PhaseComp")
     
# partA()
# partB()
# partC()