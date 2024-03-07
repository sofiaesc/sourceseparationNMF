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

# Carga de audio
sample_rate = 5512
comb, sr = librosa.load('data/prueba_3.wav', sr = sample_rate)
inst1, sr = librosa.load('data/prueba_3_piano.wav', sr = sample_rate)
inst2, sr = librosa.load('data/prueba_3_flauta.wav', sr = sample_rate)
inst3, sr = librosa.load('data/prueba_3_bajo.wav', sr = sample_rate)

# Comentar/descomentar: Añado ruido a la mezcla para probar robustez
""" ruido_blanco = np.random.normal(0,0.02,len(comb))
comb = comb + ruido_blanco """

# Plot de la señal original
fig, ax = plt.subplots(4,1)

librosa.display.waveshow(comb, sr=sr,x_axis='time',ax=ax[0])
ax[0].set_title('Mezcla original')
librosa.display.waveshow(inst1, sr=sr,x_axis='time',ax=ax[1])
ax[1].set_title('Instrumento 1 original')
librosa.display.waveshow(inst2, sr=sr,x_axis='time',ax=ax[2])
ax[2].set_title('Instrumento 2 original')
librosa.display.waveshow(inst3, sr=sr,x_axis='time',ax=ax[3])
ax[3].set_title('Instrumento 3 original')
fig.tight_layout()
plt.savefig('plots/tres instrumentos/señales_originales.png')

# Transformada de Fourier de Tiempo Corto
window = 256; # ancho de ventana
overlap = 128; # superposición de ventanas
sf = librosa.stft(comb,n_fft = window, hop_length = overlap)
sf_i1 = librosa.stft(inst1,n_fft = window, hop_length = overlap)
sf_i2 = librosa.stft(inst2,n_fft = window, hop_length = overlap)
sf_i3 = librosa.stft(inst3,n_fft = window, hop_length = overlap)

comb_mag = np.abs(sf) # obtengo magnitud de la mezcla para trabajar con NMF
comb_fase = np.angle(sf) # guardo fase para reconstrucción
inst1_mag = np.abs(sf_i1) # obtengo magnitud del instrumento 1 para trabajar con NMF
inst2_mag = np.abs(sf_i2) # obtengo magnitud del instrumento 2 para trabajar con NMF
inst3_mag = np.abs(sf_i3) # obtengo magnitud del instrumento 2 para trabajar con NMF

# Grafico espectrogramas:
fig, ax = plt.subplots(4,1)

spec = librosa.amplitude_to_db(comb_mag, ref = np.max)
librosa.display.specshow(spec, y_axis = 'hz',sr=sr,hop_length=overlap,x_axis ='time',cmap= matplotlib.cm.jet,ax=ax[0])
ax[0].set_title('Espectograma de mezcla original')
spec_i1 = librosa.amplitude_to_db(inst1_mag, ref = np.max)
librosa.display.specshow(spec_i1, y_axis = 'hz',sr=sr,hop_length=overlap,x_axis ='time',cmap= matplotlib.cm.jet,ax=ax[1])
ax[1].set_title('Espectograma de Instrumento 1 original')
spec_i2 = librosa.amplitude_to_db(inst2_mag, ref = np.max)
librosa.display.specshow(spec_i2, y_axis = 'hz',sr=sr,hop_length=overlap,x_axis ='time',cmap= matplotlib.cm.jet,ax=ax[2])
ax[2].set_title('Espectograma de Instrumento 2 original')
spec_i3 = librosa.amplitude_to_db(inst3_mag, ref = np.max)
librosa.display.specshow(spec_i3, y_axis = 'hz',sr=sr,hop_length=overlap,x_axis ='time',cmap= matplotlib.cm.jet,ax=ax[3])
ax[3].set_title('Espectograma de Instrumento 3 original')

fig.tight_layout()
plt.savefig('plots/tres instrumentos/espectrogramas_originales.png')

# Algoritmo NMF (Non-Negative Matrix Factorization)
maxit = 1000; # máximo de iteraciones
tol = 0.00005 # tolerancia de error
V = comb_mag + 1e-12
N, M = np.shape(V) # dimension de la matriz de magnitud 

# INSTRUMENTO 1:
V1 = inst1_mag
H1,W1,m_error1 = nmf_instrumentos(V1,tol,maxit,N,M)

# INSTRUMENTO 2:
V2 = inst2_mag
H2,W2,m_error2 = nmf_instrumentos(V2,tol,maxit,N,M)

# INSTRUMENTO 3:
V3 = inst3_mag
H3,W3,m_error3 = nmf_instrumentos(V3,tol,maxit,N,M)

# MEZCLA:
W = np.concatenate((W1,W2,W3),axis=1) # armo W grande con las W de los elementos separados
H = np.abs(np.random.normal(0, 2.5, (3,M))) # H aleatorizada
error = 1 # inicializamos error
m_error = [] # inicializamos matriz para graficar
it = 0
while it <= maxit: # criterio de parada: tolerancia para el error y máximo de iteraciones
    H *= (W.T@V)/(W.T@(W@H) + 1e-12) # 1e-12 es un error agregado para evitar divisiones por cero
    # W *= (V@H.T)/((W@H)@H.T + 1e-12) # mantengo fija mi W
    error = mean_squared_error(W@H,V) # obtengo el error entre la factorización W*H y el audio original
    m_error.append(error) # añado a la matriz de error
    it += 1
fig = plt.figure() 
print(error)
plt.plot(m_error)
plt.title("Error cuadrático entre W*H y V")
plt.savefig('plots/tres instrumentos/error.png')

# Multiplico matrices de la factorización para cada instrumento:
sf_i1 = (W[:,[0]]@H[[0],:]) /(W@H) * V;
sf_i2 = (W[:,[1]]@H[[1],:]) /(W@H) * V;
sf_i3 = (W[:,[2]]@H[[2],:]) /(W@H) * V;

# ESPECTROGRAMAS:
sf_i1_db = librosa.amplitude_to_db(sf_i1, ref = np.max)
sf_i2_db = librosa.amplitude_to_db(sf_i2, ref = np.max)
sf_i3_db = librosa.amplitude_to_db(sf_i3, ref = np.max)

# Espectrogramas del audio original de la mezcla y de los instrumentos separados por NMF:
fig = plt.figure()
fig, ax = plt.subplots(3,1)

librosa.display.specshow(sf_i1_db,y_axis = 'hz', sr=sr,hop_length=overlap,x_axis ='time',cmap= matplotlib.cm.jet, ax = ax[0])
ax[0].set_title('Espectrograma de instrumento 1 separado')
librosa.display.specshow(sf_i2_db,y_axis = 'hz', sr=sr,hop_length=overlap,x_axis ='time',cmap= matplotlib.cm.jet, ax = ax[1])
ax[1].set_title('Espectrograma de instrumento 2 separado')
librosa.display.specshow(sf_i3_db,y_axis = 'hz', sr=sr,hop_length=overlap,x_axis ='time',cmap= matplotlib.cm.jet, ax = ax[2])
ax[2].set_title('Espectrograma de instrumento 3 separado')

fig.tight_layout()
plt.savefig('plots/tres instrumentos/espectrogramas_comparacion.png')

# RECONSTRUCCIONES:
# Vuelvo a añadir fase que guardé antes:
i1_con_fase = sf_i1 * np.exp(1j*comb_fase)
i2_con_fase = sf_i2 * np.exp(1j*comb_fase)
i3_con_fase = sf_i3 * np.exp(1j*comb_fase)
rec_i1 = librosa.istft(i1_con_fase, n_fft = window, hop_length = overlap)
rec_i2 = librosa.istft(i2_con_fase, n_fft = window, hop_length = overlap)
rec_i3 = librosa.istft(i3_con_fase, n_fft = window, hop_length = overlap)

# Grafico las reconstrucciones:
fig = plt.figure()
fig, ax = plt.subplots(3,1)
ax[0].set_title('Instrumento 1 separado')
librosa.display.waveshow(rec_i1, sr=sr, ax=ax[0],x_axis='time')
ax[1].set_title('Instrumento 2 separado')
librosa.display.waveshow(rec_i2, sr=sr, ax=ax[1],x_axis='time')
ax[2].set_title('Instrumento 3 separado')
librosa.display.waveshow(rec_i3, sr=sr, ax=ax[2],x_axis='time')
fig.tight_layout()
plt.savefig('plots/tres instrumentos/señales_comparacion.png')

# Guardo resultado en un archivo de audio .wav:
soundfile.write(f'resultados/tres instrumentos/instrumento1.wav',rec_i1,sr)
soundfile.write(f'resultados/tres instrumentos/instrumento2.wav',rec_i2,sr)
soundfile.write(f'resultados/tres instrumentos/instrumento3.wav',rec_i3,sr)