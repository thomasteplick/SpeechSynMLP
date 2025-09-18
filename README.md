<h3>
Synthetic Speech Classification using a Multilayer Perceptron (MLP) Neural Network with Backpropagation
</h3>
<p>
This is a web application written in Go that make use of the html/template package to dynamically
create the web page. Start the web server at bin\mlpspeech.exe and connect to it from your web browser
at http://127.0.0.1:8080/speechMLPtrain. The program creates synthetic speech and generates spectrograms from them. 
Spectrograms are 3-dimensional plots of the time-frequency content of the audio. The spectral power at a 
particular frequency and time is shown as a grayscale color, with black having the greatest power and white 
having the least. Short-time Fourier transforms (STFT) are used in 32 ms time frames with no overlap. 
between FFTs. Time domain plots of the audio waveform can be displayed as well as the spectrogram. 
The user can generate new synthetic speech of 2.048 seconds (16,384 samples) by selecting the New Speech radio button.  
Upon selecting the <i>Submit</i> button, the user will hear the speech on their computer's audio device. 
To hear the synthetic speech requires the <b>fmedia</b> program to be in the PATH environmental variable.
A WAV file is created from the synthetic speach and submitted to the fmedia program to be played.</p>

<p>
The FFT size is 256. The sampling rate is 8,000 Hz which produces a Nyquist critical frequency of 4,000 Hz. 
The grayscale consists of five colors. As stated above, black RGB(0,0,0), has the greatest power in the 
Power Spectral Density (PSD) bin.  There are 256 bins in the 256-pt FFT and each bin is 8,000/256 = 31.25 Hz wide.
</p>

<p>
The spectrogram's PSD maximum bin for each 32ms frame is used as input to the Multilayer Perceptron Neural Network. 
That is, the frequency bin with the maximum power of each frame is submitted serially to the input layer of the MLP. 
Backpropagation is used to train the weights in the network. The Train page allows the speech and frequency 
domain parameters to be chosen. Upon sumitting the form in the HTML, the synthetic speech patterns are converted 
to spectrograms and submitted to the MLP. Next the Test page will generate variation of the synthetic speech 
patterns. In the <i>Display</i> page, you can plot the time domain or the spectrogram of one of the synthetic speech patterns.
</p>

<p>
The synthetic speech is generated with a sum of sinusoids (voiced) or gaussian noise (unvoiced) in
20-30 ms frames.  If voiced, the fundamental frequency is randomly chosen from between 200 and 800 Hz. Each voiced
speech has 1-5 subfrequencies with a smaller amplitude than the fundamental.  The amplitudes are randomly
chosen and can be varied.  The duration of each frame can also be varied.  The variation of these parameters
will test the generalization capabilities of the Neural Network.  The testing phase varies the parameters
based upon the user input.  The percentage of correct classification is presented in graphical and tabular
forms upon completion of the testing.
</p>

<p>
There are two phases of operation:  the training phase and the testing phase.  
Epochs consising of a sequence of examples are used to train the Neural Network.  
Each example consists of a spectrogram (STFT) of synthetic speech and a desired class output.  The MLP
itself consists of an input layer of nodes, one or more hidden layers containing nodes,
and an output layer of nodes.  The nodes are fully connected by weighted links.  The
weights are trained by back propagating the output layer errors forward to the
input layer.  The chain rule of differential calculus is used to assign credit
for the errors in the output to the weights in the hidden layers.
The output layer outputs are subtracted from the desired to obtain the error.
The user trains first and then tests.  The MLP Neural Network uses Rectified
Linear Unit (ReLU) as the activation function in the hidden layers and Softmax function
(normalized exponential) in the output layer.  Cross-entropy loss is used to compute
the error in the ouput layer with one-hot vector as the target or desired output.
This is a classification problem and only one of the ouputs is one, the rest are zero.
Therefore the outputs are probabilities with values between 0 and 1.  The maximum probability
is declared the MLP's class for the corresponding synthetic speech input.
</p>
<h4>Training Learning Curve, Cross Entropy vs Epoch</h4>
<img width="1143" height="901" alt="image" src="https://github.com/user-attachments/assets/f0c97116-630a-41d3-b86c-2f63cb4b5ea0" />
<h4>Test Results</h4>
<img width="1319" height="956" alt="image" src="https://github.com/user-attachments/assets/57c6aeb3-93cb-4902-b03e-f234d64977db" />
<h4>Display Time Domain, Speech Pattern 1</h4>
<img width="1296" height="962" alt="image" src="https://github.com/user-attachments/assets/af987c0e-a63a-45c1-849f-fe426ec493ab" />
<h4>Display Spectrogram, Speech Pattern 1</h4>
<img width="1317" height="923" alt="image" src="https://github.com/user-attachments/assets/4a7561b6-3dd7-4f59-b2a7-a190a9343870" />




