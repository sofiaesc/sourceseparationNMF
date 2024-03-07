import numpy as np
import librosa
import math

def powSignal(signal):
    sq = np.square(signal)
    power = np.mean(sq)
    return power

def SNRsystem(inputSig, outputSig):
    noise = outputSig-inputSig
    inputSig = inputSig - np.mean(inputSig)
    outputSig = inputSig - np.mean(outputSig)
    powS = powSignal(outputSig)
    powN = powSignal(noise)
    return 10*math.log10(powS/powN)

sample_rate = 5512

print("------------------------\n Dos instrumentos:\n------------------------")
inst1, sr = librosa.load('data/prueba_2_piano.wav', sr = sample_rate)
inst2, sr = librosa.load('data/prueba_2_flauta.wav', sr = sample_rate)
rec_i1, sr = librosa.load('resultados/dos instrumentos/instrumento1.wav', sr = sample_rate)
rec_i2, sr = librosa.load('resultados/dos instrumentos/instrumento2.wav', sr = sample_rate)
snr_i1 = SNRsystem(inst1[0:54144], rec_i1[0:54144])
print(f"SNR del Instrumento 1: {snr_i1}")
snr_i2 = SNRsystem(inst2[0:54144], rec_i2[0:54144])
print(f"SNR del Instrumento 2: {snr_i2}")

print("------------------------\n Dos instrumentos con ruido:\n------------------------")
rec_i1, sr = librosa.load('resultados/ruido/instrumento1.wav', sr = sample_rate)
rec_i2, sr = librosa.load('resultados/ruido/instrumento2.wav', sr = sample_rate)
snr_i1 = SNRsystem(inst1[0:54144], rec_i1[0:54144])
print(f"SNR del Instrumento 1: {snr_i1}")
snr_i2 = SNRsystem(inst2[0:54144], rec_i2[0:54144])
print(f"SNR del Instrumento 2: {snr_i2}")

print("------------------------\n Tres instrumentos:\n------------------------")
inst1, sr = librosa.load('data/prueba_3_piano.wav', sr = sample_rate)
inst2, sr = librosa.load('data/prueba_3_flauta.wav', sr = sample_rate)
inst3, sr = librosa.load('data/prueba_3_bajo.wav', sr = sample_rate)
rec_i1, sr = librosa.load('resultados/tres instrumentos/instrumento1.wav', sr = sample_rate)
rec_i2, sr = librosa.load('resultados/tres instrumentos/instrumento2.wav', sr = sample_rate)
rec_i3, sr = librosa.load('resultados/tres instrumentos/instrumento3.wav', sr = sample_rate)
snr_i1 = SNRsystem(inst1[0:71552], rec_i1[0:71552])
print(f"SNR del Instrumento 1: {snr_i1}")
snr_i2 = SNRsystem(inst2[0:71552], rec_i2[0:71552])
print(f"SNR del Instrumento 2: {snr_i2}")
snr_i3 = SNRsystem(inst3[0:71552], rec_i3[0:71552])
print(f"SNR del Instrumento 3: {snr_i3}")