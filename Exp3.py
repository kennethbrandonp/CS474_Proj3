from Exp1 import np, fft
from pgm import PGM
        

def fft2D(n, m, real, imag, isign):
    #Combine to put into our 1D DFT:
    data = real + 1j * imag

    #FFT on rows:
    for i in range(n):
        data[i, :] = fft(data[i, :], m, isign)

    #FFT on columns:
    for j in range(m):
        data[:, j] = fft(data[:, j], n, isign)
    
    if isign == -1:
        data /= n * m   #Normlization

    real = data.real
    imag = data.imag

    return real, imag

def partA(real, imag, pgm):
    #Set the phase to zero:
    magnitude = np.abs(real)
    zero = np.zeros_like(magnitude)

    #Combine real and imag components:
    phase = magnitude + 1j * zero   #Imaginary portion set to zero

    #Perform inverse 2D FFT:
    inverseReal, inverseImag = fft2D(image.x, image.y, phase.real, phase.imag, 1)

    #Round our real numbers to the nearest positive integer:
    inverseReal = np.clip(np.ceil(inverseReal).astype(int), 0, 255)
    #Flip it back to original orientation:
    inverseReal = inverseReal[::-1, ::1]

    pgm.pixels = inverseReal
    pgm.save("NoPhase") #Save zero phase image
    
def partB(real, imag, pgm):
    originalPhase = np.angle(real + 1j * imag) #atan2
    cosine = np.cos(originalPhase)
    sinc = np.sin(originalPhase)

    #Combine consine and sinc
    magOne = cosine + 1j * sinc

    #Inverse DFT
    inverseReal, inverseImag = fft2D(image.x, image.y, magOne.real, magOne.imag, 1)

    #Rescale pixel values 0-225:
    inverseReal = np.clip(np.ceil(inverseReal).astype(int), 0, 255)

    #Flip to original orientation:
    inverseReal = inverseReal[::-1, ::1]

    pgm.pixels = inverseReal
    pgm.save("OriginalPhaseMagOne")


###Computing the DFT for the lenna image###
#Read in lenna image:
image = PGM("lenna")

dataReal = np.array(image.pixels)
dataImag = np.zeros_like(dataReal)

#Computed lenna DFT:
forwardReal, forwardImag = fft2D(image.x, image.y, dataReal, dataImag, -1) #Forward transformation

#Verify we get the same result back:
inverseReal, inverseImag = fft2D(image.x, image.y, forwardReal, forwardImag, 1) #Inverse transformation

#Round our real numbers to the nearest positive integer:
inverseReal = np.ceil(inverseReal).astype(int)
#Flip it back to original orientation:
inverseReal = inverseReal[::-1, ::1]

image.pixels = inverseReal
image.save("InverseDFT") #Save our image to verify 2D FFT

###Experiments###
# partA(forwardReal, forwardImag, image)
# partB(forwardReal, forwardImag, image)
