import math
from Exp1 import np, fft
from pgm import PGM

def clamp(a, b, x):
    if x > b:
        return b
    if x < a:
        return a
    return x

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

def visDFT(fname, centered):
    image = PGM(fname)

    if centered == 1:
        for i in range(0,image.y):
            for k in range(0,image.x):
                image.pixels[i][k] = image.pixels[i][k] * math.pow(-1, i+k)

    dataReal = np.array(image.pixels)
    dataImag = np.zeros_like(dataReal)

    r, c = fft2D(image.x, image.y, dataReal, dataImag, -1) #Forward transformation

    for i in range(0,image.y):
        for k in range(0,image.x):
            idx = i * image.x + k
            m = r[i][k] * r[i][k] + c[i][k] * c[i][k]
            image.pixels[i][k] = clamp(0,255, 120 * math.log2(1 + math.sqrt(m)))

    if centered == 1:
        image.save("_centered_dft")
    else:
        image.save("_dft")

visDFT("square32", 0)
visDFT("square64", 0)
visDFT("square128", 0)

visDFT("square32", 1)
visDFT("square64", 1)
visDFT("square128", 1)


