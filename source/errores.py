import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import librosa
import librosa.display
import soundfile
from sklearn.metrics import mean_squared_error

def nmf_instrumentos(V,tol,maxit,N,M):
    W = np.abs(np.random.normal(0, 2.5, (N,1)))    
    H = np.abs(np.random.normal(0, 2.5, (1,M)))

    error = 1 # inicializamos error
    m_error = [] # inicializamos matriz para graficar
    it = 0
    while it <= maxit and error >= tol: # criterio de parada: tolerancia para el error y máximo de iteraciones
        H *= (W.T@V)/(W.T@(W@H) + 1e-12) # 1e-12 es un error agregado para evitar divisiones por cero
        W *= (V@H.T)/((W@H)@H.T + 1e-12)
        error = mean_squared_error(W@H,V) # obtengo el error entre la factorización W*H y el audio original
        m_error.append(error) # añado a la matriz de error
        it += 1
    return H,W,m_error

def errores(window,overlap,comb,inst1,inst2):
    sf = librosa.stft(comb,n_fft = window, hop_length = overlap)   
    sf_i1 = librosa.stft(inst1,n_fft = window, hop_length = overlap)
    sf_i2 = librosa.stft(inst2,n_fft = window, hop_length = overlap)

    comb_mag = np.abs(sf) # obtengo magnitud de la mezcla para trabajar con NMF
    inst1_mag = np.abs(sf_i1) # obtengo magnitud del instrumento 1 para trabajar con NMF
    inst2_mag = np.abs(sf_i2) # obtengo magnitud del instrumento 2 para trabajar con NMF
    comb_fase = np.angle(sf) # guardo fase para reconstrucción

    maxit = 1000; # máximo de iteraciones
    tol = 0.00005 # tolerancia de error
    V = comb_mag + 1e-12
    N, M = np.shape(V) # dimension de la matriz de magnitud 

    V1 = inst1_mag
    H1,W1,m_error1 = nmf_instrumentos(V1,tol,maxit,N,M)
    V2 = inst2_mag
    H2,W2,m_error2 = nmf_instrumentos(V2,tol,maxit,N,M)

    W = np.concatenate((W1,W2),axis=1) # armo W grande con las W de los elementos separados
    H = np.abs(np.random.normal(0, 2.5, (2,M))) # H aleatorizada
    error = 1 # inicializamos error
    m_error = [] # inicializamos matriz para graficar
    it = 0
    while it <= maxit: # criterio de parada: tolerancia para el error y máximo de iteraciones
        H *= (W.T@V)/(W.T@(W@H) + 1e-12) # 1e-12 es un error agregado para evitar divisiones por cero
        # W *= (V@H.T)/((W@H)@H.T + 1e-12) # mantengo fija mi W
        error = mean_squared_error(W@H,V) # obtengo el error entre la factorización W*H y el audio original
        m_error.append(error) # añado a la matriz de error
        it += 1
    plt.plot(m_error,label=f"Ventana de {window} - Overlap de {overlap}")
    plt.legend()
    plt.ylim(0.03,0.04)

sample_rate = 5512
comb, sr = librosa.load('data/prueba.wav', sr = sample_rate)
inst1, sr = librosa.load('data/prueba_piano.wav', sr = sample_rate)
inst2, sr = librosa.load('data/prueba_flauta.wav', sr = sample_rate)

plt.figure()
errores(256,255,comb,inst1,inst2)
errores(256,200,comb,inst1,inst2)
errores(256,128,comb,inst1,inst2)
errores(256,100,comb,inst1,inst2)
errores(256,60,comb,inst1,inst2)
errores(256,20,comb,inst1,inst2)
plt.savefig('plots/error.png')