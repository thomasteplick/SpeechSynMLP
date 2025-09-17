'''
Created on Aug 19, 2025
Generate synthetic speech using sine waves and noise
@author: tjteplick
'''
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

# sample frequency in Hz
fs = 8000.0
nyquist = fs/2
fft_size = 256
nffts = 64
# samples in a frame, with period = 25ms
frame_size = 200 
nsamples = fft_size*nffts 
nframes = math.ceil(nsamples/frame_size)
# min and max pitch in Hz
f1_min = 200.0
f1_max = 800.0
# min and max amplitues of pitch
ampl_f1_min = 1.0
ampl_f1_max = 2.0
maxsubfreq = 3
synSpeech = [0.0]*nsamples
freqs = [0.0]*(maxsubfreq+1)
amps = [0.0]*(maxsubfreq+1)
outfile = "speech.csv"

# patterns with periodic structure are voiced
percent_voiced = 50
# patterns with noise only are unvoiced
sigma_noise = 1.0
# time step
tstep = 1.0/fs

def create_patterns():
    '''
    Create sinusoidal frequencies and amplitudes as patterns for synthetic speech
    '''
    with (open(outfile, 'w')) as fout:
        # loop over frames in the speech   
        for _ in range(nframes):
            # voiced speech consisting of sinusoids
            if random.randint(0,100) < percent_voiced:
                # select a fundamental frequency in (200,800) Hz and amplitude in (1.0,2.0)
                freqs[0] = random.randrange(int(f1_min), int(f1_max))
                amps[0] = ampl_f1_min + random.random()*(ampl_f1_max-ampl_f1_min)
                # select 1-3 higher sub-frequencies,  f1 < f2 < f3 < f4
                # select sub-amplitudes in (1.0, 2.0), a1 > a2 > a3 > a4
                nsubfreq = random.randint(1,maxsubfreq)
                for sf in list(range(1,nsubfreq+1)):
                    freqs[sf] = freqs[sf-1] + (nyquist-freqs[sf-1])*random.random()
                    amps[sf] = amps[sf-1]*random.random()
            # unvoiced consisting of uniform noise
            else:
                nsubfreq = 0
                freqs[0] = 0.0
                amps[0] = sigma_noise*random.random()
            print(*freqs[:nsubfreq+1], sep=',', end=',', file=fout)
            print(*amps[:nsubfreq+1], sep=',', end='\n', file=fout)
    return

def create_speech():
    ''' 
    Create synthetic speech by summing sinusoids and normal noise using
    frames of 25ms duration.
    '''
    start = 0
    k1 = .01
    k2 = .01
    k3 = 25
    # read speech file to get the frequencies and amplitudes for the pattern
    with (open(outfile, 'r')) as fin:
        # loop over frames and parse the items in each line
        remain = nsamples
        for i, line in enumerate(fin):
            # frequencies, amplitudes
            items = line.split(',')
            nfreq = len(items)//2
            # set frequency delta, duration delta, amplitude delta
            for i in range(nfreq):
                # frequencies
                f = float(items[i])
                if random.randint(0,1) == 0:
                    freqs[i] = (1.0 - k1)*f
                else:
                    freqs[i] = (1.0 + k1)*f
                # amplitudes
                a = float(items[i+nfreq])
                if random.randint(0,1) == 0:
                    amps[i] = (1.0 - k2)*a
                else:
                    amps[i] = (1.0 + k2)*a
                    
            if random.randint(0,1) == 0:
                duration = frame_size - random.randrange(0, k3)
            else:
                duration = frame_size + random.randrange(0, k3)
            
            if i < nframes-1:
                duration = min(remain, duration)
            else:
                duration = remain
                
            # call synthesize with start, stop, freq, ampl
            synthesize(start, start+duration, freqs[:nfreq], amps[:nfreq])
            start += duration
            remain -= duration
    return   

def synthesize(start, stop, freqs, amps):
    '''
    Create a frame in start to stop of synthetic speech using the frequencies and amplitudes
    '''
    t = 0
    step = 1.0/fs
    # loop over the samples in the frame
    for n in range(start, stop):
        val = 0.0
        # loop over the frequencies and add their values
        if freqs[0] == 0:
            val += random.gauss(mu=0, sigma=amps[0])
        else:
            for i, fr in enumerate(freqs):
                val += amps[i]*math.sin(math.tau*fr*t)
        t += step
        synSpeech[n] = val
    
    return
     
def plot_speech():
    '''
    Plot the synthetic speech as amplitude vs time and the spectrogram.
    '''
    # amplitude/time
    ep = nsamples/fs
    t = np.linspace(0, ep, nsamples, endpoint=False)
    _, ax = plt.subplots(1,1)
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Synthetic Speech")
    ax.plot(t, synSpeech)
    plt.show()
    # spectrogram, magnitude/time
    f,t,Sxx = signal.spectrogram(x=np.array(synSpeech), fs=fs, window='boxcar', mode='magnitude', nperseg=256, nfft=256, noverlap=0)
    plt.pcolormesh(t, f, Sxx, shading='auto')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (sec)')
    plt.title('Spectrogram Synthetic Speech')
    plt.show()
    

def run_main():
    create_patterns()
    create_speech()
    plot_speech()

if __name__ == '__main__':
    run_main()