/*
Multilayer Perceptron with Back-propagation Architecture.
This is a web application that uses the html/template package to create the HTML.
The URL is http://127.0.0.1:8080/SpeechSynMLP.  There are two phases of
operation:  the training phase and the testing phase.  Epochs consising of
a sequence of examples are used to train the Neural Network.  Each example consists
of a spectrogram of synthetic speech and a desired class output.  The MLP
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
Therefore the outputs are probabilities with values between 0 and 1.

This application classifies synthetic speech patterns.  The spectrogram of
each speech pattern file is calculated and the spectrogram is the input to the MLP.
The MLP classifies the speech pattern based on its spectral content versus time. The test
results are shown.  The user can plot the time domain or the spectrogram
(frequency versus time) of the synthetic speech.  The spectrogram is a three-dimentional
plot of the spectral power versus time.  The third dimension is a grayscale color.
Short-time Fourier Transforms (STFT) are used to compute the FFT from 20-30 ms blocks
of synthetic speech data.

The synthetic speech is generated with a sum of sinusoids (voiced) or gaussian noise (unvoiced) in
20-30 ms frames.  If voiced, the fundamental is randomly chosen from between 200 and 800 Hz. Each voiced
speech has 1-5 subfrequencies with a smaller amplitude than the fundamental.  The amplitudes are randomly
chosen and can be varied.  The duration of each frame can also be varied.  The variation of these parameters
will test the generalization capabilities of the Neural Network.  The testing phase varies the parameters
based upon the user input.  The percentage of correct classification is presented in graphical and tabular
forms upon completion of the testing.
*/

package main

import (
	"bufio"
	"fmt"
	"html/template"
	"log"
	"math"
	"math/cmplx"
	"math/rand"
	"net/http"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/go-audio/audio"
	"github.com/go-audio/wav"
	"github.com/mjibson/go-dsp/fft"
)

const (
	addr               = "127.0.0.1:8080"             // http server listen address
	fileTrainingMLP    = "templates/trainingMLP.html" // html for training MLP
	fileTestingMLP     = "templates/testingMLP.html"  // html for testing MLP
	fileDisplayMLP     = "templates/displayMLP.html"  // html for speech time or spectrogram plots
	patternTrainingMLP = "/speechMLPtrain"            // http handler for training the MLP
	patternTestingMLP  = "/speechMLPtest"             // http handler for testing the MLP
	patternDisplayMLP  = "/speechMLPdisplay"          // http handler for displaying the MLP
	xlabels            = 11                           // # labels on x axis
	ylabels            = 11                           // # labels on y axis
	fileweights        = "weights.csv"                // mlp weights
	synSpeech          = "synSpeech.wav"              // synthetic speech wav file
	dataDir            = "data/"                      // directory for the weights and synthetic speech files
	rows               = 300                          // rows in canvas
	cols               = 300                          // columns in canvas
	sampleRate         = 8000                         // Hz or samples/sec
	bitDepth           = 16                           // audio wav encoder/decoder sample size
	ncolors            = 5                            // number of grayscale colors in spectrogram
	nffts              = 64                           // number of ffts in the spectrograms
	avgDuration        = 200                          // average duration in samples of the speech frame size
	npatterns          = 16                           // number of synthetic speech patterns
	classes            = 16                           // number of classes is the number of speech patterns
	maxSubFreq         = 5                            // max number of sub-frequencies
)

// test statistics that are tabulated in HTML
type Results struct {
	Class   string // int
	Correct string // int      percent correct
	Count   string // int      number of training examples in the class
}

// Type to contain all the HTML template actions
type PlotT struct {
	Grid          []string  // plotting grid
	Status        string    // status of the plot
	Xlabel        []string  // x-axis labels
	Ylabel        []string  // y-axis labels
	HiddenLayers  string    // number of hidden layers
	LayerDepth    string    // number of Nodes in hidden layers
	LearningRate  string    // size of weight update for each iteration
	Momentum      string    // previous weight update scaling factor
	Epochs        string    // number of epochs
	FFTSize       string    // 8192, 4098, 2048, 1024
	FFTWindow     string    // Bartlett, Welch, Hamming, Hanning, Rectangle
	Domain        string    // plot time or spectrogra domain
	TestResults   []Results // tabulated statistics of testing
	TotalCount    string    // Results tabulation
	TotalCorrect  string
	DelDuration   string // delta of the speech frame duration
	DelPitch      string // delta of the speech frame frequencies
	DelAmpl       string // delta of the speech frame frequency amplitudes
	PercentVoiced string // percentage of the speech frames voiced
	Classes       string // number of speech patterns
	SpeechPattern string // speech pattern to display
}

// Type to hold the minimum and maximum data values of the MSE in the Learning Curve
type Endpoints struct {
	xmin float64
	xmax float64
	ymin float64
	ymax float64
}

// graph node
type Node struct {
	y     float64 // output of this node for forward prop
	delta float64 // local gradient for backward prop
}

// graph links used to connect last Feature Map layer to output layer
type Link struct {
	wgt      float64 // weight
	wgtDelta float64 // previous weight update used in momentum
}

type Stats struct {
	correct    []int // % correct classifcation
	classCount []int // #samples in each class
}

// training examples
type Sample struct {
	desired int    // numerical class of the synthetic speech pattern
	data    []Node //  frequency bins from the STFT and deltas from backprop
}

// Speech frame attributes
type SpeechFrame struct {
	freqs []float64
	amps  []float64
}

// Primary data structure for holding the MLP state
type MLP struct {
	plot          *PlotT          // data to be distributed in the HTML template
	Endpoints                     // embedded struct
	link          [][]Link        // links in the graph
	node          [][]Node        // nodes in graph
	nsamples      int             // number of synthetic speech pattern
	domain        string          // time or spectrogram plot
	data          []float64       // cross-entropy Loss in output layer per epoch used in Learning Curve
	epochs        int             // number of epochs
	learningRate  float64         // learning rate parameter
	momentum      float64         // delta weight scale constant
	hiddenLayers  int             // number of hidden layers
	desired       []float64       // desired output of the sample
	layerDepth    int             // hidden layer number of nodes
	words         []string        // classified words in test message
	grayscale     map[int]string  // grayscale for spectrogram
	fftSize       int             // FFT size for spectrogram
	fftWindow     string          // FFT window
	speechPat     [][]SpeechFrame // speech pattern
	delPitch      int             // pitch delta
	delDuration   int             // duration delta in samples
	delAmpl       float64         // amplitude delta of the frequencies
	percentVoiced int             // percent voiced
	synSpeech     []float64       // synthetic speech
	statistics    Stats
	freqs         []float64 // speech frame frequencies
	amps          []float64 // speech frame amplitudes
}

// Window function type
type Window func(n int, m int) complex128

// global variables for parse and execution of the html template
var (
	tmplTrainingMLP *template.Template
	tmplTestingMLP  *template.Template
	tmplDisplayMLP  *template.Template
	winType         = []string{"Bartlett", "Welch", "Hamming", "Hanning", "Rectangle"}
)

// init parses the html template files
func init() {
	tmplTrainingMLP = template.Must(template.ParseFiles(fileTrainingMLP))
	tmplTestingMLP = template.Must(template.ParseFiles(fileTestingMLP))
	tmplDisplayMLP = template.Must(template.ParseFiles(fileDisplayMLP))
}

// Bartlett window
func bartlett(n int, m int) complex128 {
	real := 1.0 - math.Abs((float64(n)-float64(m))/float64(m))
	return complex(real, 0)
}

// Welch window
func welch(n int, m int) complex128 {
	x := math.Abs((float64(n) - float64(m)) / float64(m))
	real := 1.0 - x*x
	return complex(real, 0)
}

// Hamming window
func hamming(n int, m int) complex128 {
	return complex(.54-.46*math.Cos(math.Pi*float64(n)/float64(m)), 0)
}

// Hanning window
func hanning(n int, m int) complex128 {
	return complex(.5-.5*math.Cos(math.Pi*float64(n)/float64(m)), 0)
}

// Rectangle window
func rectangle(n int, m int) complex128 {
	return 1.0
}

// calculateCrossEntropy Loss finds the error at the output layer every epoch
func (mlp *MLP) calculateCrossEntropy(epoch int) {
	// loop over the output layer nodes
	var err float64 = 0.0
	mlp.data[epoch] = 0.0
	outputLayer := len(mlp.node) - 1
	for n := 0; n < len(mlp.node[outputLayer]); n++ {
		// Calculate -Sum(desired(i)*log(y(i))) and store in mlp.data[n]
		// yi = exp(vi)/sum(exp(vj)), normalized exponential of output layer component i
		err = -mlp.desired[n] * math.Log(mlp.node[outputLayer][n].y)
		mlp.data[epoch] += err
	}

	// calculate min/max cross entropy
	if mlp.data[epoch] < mlp.ymin {
		mlp.ymin = mlp.data[epoch]
	}
	if mlp.data[epoch] > mlp.ymax {
		mlp.ymax = mlp.data[epoch]
	}
}

// determineClass determines testing example class given sample number and sample
func (mlp *MLP) determineClass(sample *Sample) error {
	// At output layer, classify example and increment class/correct count

	// convert node outputs to the class; one-hot vector
	maxy := 0.0
	class := 0
	for i, output := range mlp.node[mlp.hiddenLayers+1] {
		if output.y > maxy {
			maxy = output.y
			class = i
		}
	}

	// Assign Stats.correct, Stats.classCount
	mlp.statistics.classCount[sample.desired]++
	if class == sample.desired {
		mlp.statistics.correct[class]++
	}

	return nil
}

// class2desired constructs the desired output from the given class
func (mlp *MLP) class2desired(class int) {
	// tranform int to slice with one location equal one, all others zero
	// the so-called one-hot vector
	for i := 0; i < len(mlp.desired); i++ {
		if i == class {
			mlp.desired[i] = 1.0
		} else {
			mlp.desired[i] = 0.0
		}
	}
}

func (mlp *MLP) propagateForward(samp *Sample) error {
	// Assign sample to input layer, i=0 is the bias equal to one
	layer := 0
	for i, val := range samp.data {
		mlp.node[layer][i+1].y = val.y
	}

	// calculate desired from the class
	mlp.class2desired(samp.desired)

	// Loop over layers: input + hiddenLayers + output layer
	// input->first hidden, then hidden->hidden,..., then hidden->output
	for layer := 1; layer <= mlp.hiddenLayers; layer++ {
		// Loop over FMs in the layer, d1 is the layer depth of current
		d1 := len(mlp.node[layer])
		for i1 := 1; i1 < d1; i1++ { // this layer loop
			// The network is fully connected.  d2 is the layer depth of previous
			d2 := len(mlp.node[layer-1])
			// Loop over weights to get v
			v := 0.0
			for i2 := range d2 { // previous layer loop
				v += mlp.link[layer-1][i2*(d1-1)+i1-1].wgt * mlp.node[layer-1][i2].y
			}
			// compute output y = Phi(v) for the ReLU activation function
			mlp.node[layer][i1].y = max(0, v)
		}
	}

	// last layer is different because there is no bias node, so the indexing is different
	layer = mlp.hiddenLayers + 1
	sum := 0.0
	d1 := len(mlp.node[layer])
	for i1 := range d1 { // this layer loop
		// Each node in previous layer is connected to current node because
		// the network is fully connected.  d2 is the layer depth of previous
		d2 := len(mlp.node[layer-1])
		// Loop over weights to get v
		v := 0.0
		for i2 := 0; i2 < d2; i2++ { // previous layer loop
			v += mlp.link[layer-1][i2*d1+i1].wgt * mlp.node[layer-1][i2].y
		}
		// compute output using Softmax, normalized exponential
		mlp.node[layer][i1].y = math.Exp(v)
		sum += mlp.node[layer][i1].y
	}
	// normalize the exponential to make a probability in (0, 1)
	for i := range mlp.node[layer] {
		mlp.node[layer][i].y /= sum
	}
	return nil
}

func (mlp *MLP) propagateBackward() error {

	// output layer is different, no bias node, so the indexing is different
	// Loop over nodes in output layer
	layer := mlp.hiddenLayers + 1
	d1 := len(mlp.node[layer])
	for i1 := range d1 { // this layer loop
		//compute error e=d-y, where y is the normalized exponential or probability,
		// d is 0 or 1, the one-hot vector
		mlp.node[layer][i1].delta = mlp.desired[i1] - mlp.node[layer][i1].y
		// Send this node's local gradient to previous layer nodes through corresponding link.
		// Each node in previous layer is connected to current node because the network
		// is fully connected.  d2 is the previous layer depth
		d2 := len(mlp.node[layer-1])
		for i2 := range d2 { // previous layer loop
			mlp.node[layer-1][i2].delta += mlp.link[layer-1][i2*d1+i1].wgt * mlp.node[layer][i1].delta
			// Compute weight delta, Update weight with momentum, y, and local gradient
			wgtDelta := mlp.learningRate * mlp.node[layer][i1].delta * mlp.node[layer-1][i2].y
			mlp.link[layer-1][i2*d1+i1].wgt +=
				wgtDelta + mlp.momentum*mlp.link[layer-1][i2*d1+i1].wgtDelta
			// update weight delta
			mlp.link[layer-1][i2*d1+i1].wgtDelta = wgtDelta
		}
		// Reset this local gradient to zero for next training example
		mlp.node[layer][i1].delta = 0.0
	}

	// Loop over layers in backward direction, starting at the last hidden layer
	for layer := mlp.hiddenLayers; layer > 0; layer-- {
		// Loop over nodes in this layer, d1 is the current layer depth
		d1 := len(mlp.node[layer])
		for i1 := 1; i1 < d1; i1++ { // this layer loop
			// Multiply error by this node's Phi'(v) to get local gradient.
			// ReLU derivative = 1 if y > 0, else 0
			if mlp.node[layer][i1].y <= 0 {
				mlp.node[layer][i1].delta = 0
			}
			// Send this node's local gradient to previous layer nodes through corresponding link.
			// Each node in previous layer is connected to current node because the network
			// is fully connected.  d2 is the previous layer depth
			d2 := len(mlp.node[layer-1])
			for i2 := range d2 { // previous layer loop
				mlp.node[layer-1][i2].delta += mlp.link[layer-1][i2*(d1-1)+i1-1].wgt * mlp.node[layer][i1].delta
				// Compute weight delta, Update weight with momentum, y, and local gradient
				// anneal learning rate parameter: mlp.learnRate/(epoch*layer)
				// anneal momentum: momentum/(epoch*layer)
				wgtDelta := mlp.learningRate * mlp.node[layer][i1].delta * mlp.node[layer-1][i2].y
				mlp.link[layer-1][i2*(d1-1)+i1-1].wgt +=
					wgtDelta + mlp.momentum*mlp.link[layer-1][i2*(d1-1)+i1-1].wgtDelta
				// update weight delta
				mlp.link[layer-1][i2*(d1-1)+i1-1].wgtDelta = wgtDelta

			}
			// Reset this local gradient to zero for next training example
			mlp.node[layer][i1].delta = 0.0
		}
	}
	return nil
}

// runTrainingEpochs performs forward and backward propagation over each sample
func (mlp *MLP) runTrainingEpochs() error {

	// number of samples per pattern, number of frames in the pattern
	nsamples := mlp.fftSize * nffts
	nframes := int(math.Ceil(float64(nsamples) / float64(avgDuration)))

	// Initialize the weights

	// input layer
	// initialize the wgt and wgtDelta randomly, zero mean, normalize by fan-in
	for i := range mlp.link[0] {
		mlp.link[0][i].wgt = 2.0 * (rand.ExpFloat64() - .5) / float64(nframes+1)
		mlp.link[0][i].wgtDelta = 2.0 * (rand.ExpFloat64() - .5) / float64(nframes+1)
	}

	// output layer links
	for i := range mlp.link[mlp.hiddenLayers] {
		mlp.link[mlp.hiddenLayers][i].wgt = 2.0 * (rand.Float64() - .5) / float64(mlp.layerDepth)
		mlp.link[mlp.hiddenLayers][i].wgtDelta = 2.0 * (rand.Float64() - .5) / float64(mlp.layerDepth)
	}

	// hidden layers
	for layer := 1; layer < len(mlp.link)-1; layer++ {
		for link := 0; link < len(mlp.link[layer]); link++ {
			mlp.link[layer][link].wgt = 2.0 * (rand.Float64() - .5) / float64(mlp.layerDepth)
			mlp.link[layer][link].wgtDelta = 2.0 * (rand.Float64() - .5) / float64(mlp.layerDepth)
		}
	}

	// Create a sample for containing the spectrogram to propagate forward
	samp := Sample{data: make([]Node, nframes)}

	for n := 0; n < mlp.epochs; n++ {
		// create speech for one pattern
		// Randomly choose a pattern
		pattern := rand.Intn(npatterns)
		err := mlp.createSpeech(pattern)
		for err != nil {
			if err.Error() == "repeat" {
				err = mlp.createSpeech(pattern)
			} else {
				fmt.Printf("createSpeech error: %v\n", err.Error())
				return fmt.Errorf("createSpeech error: %v", err.Error())
			}
		}

		samp.desired = pattern

		// create spectrogram
		err = mlp.createSpectrogram(&samp)
		if err != nil {
			fmt.Printf("createSpectrogram error: %v\n", err.Error())
			return fmt.Errorf("createSpectrogram error: %v", err.Error())
		}

		// Forward Propagation
		err = mlp.propagateForward(&samp)
		if err != nil {
			return fmt.Errorf("forward propagation error: %s", err.Error())
		}

		// Backward Propagation
		err = mlp.propagateBackward()
		if err != nil {
			return fmt.Errorf("backward propagation error: %s", err.Error())
		}

		// At the end of each epoch, loop over the output nodes and calculate cross entropy loss
		mlp.calculateCrossEntropy(n)

	}
	return nil
}

// createSpectrogram creates spectrograms from the synthetic speech used in runEpochs
func (mlp *MLP) createSpectrogram(samp *Sample) error {

	// Power Spectral Density, PSD[N/2] is the Nyquist critical frequency
	// It is (sampling frequency)/2, the highest non-aliased frequency
	PSD := make([]float64, mlp.fftSize/2)

	mlp.nsamples = mlp.fftSize * nffts

	// Create the spectrogram, no overlap
	// loop over the samples, with fftSize jump
	i := 0
	for smpl := 0; smpl < len(mlp.synSpeech); smpl += mlp.fftSize {
		i = smpl / mlp.fftSize
		binMax, _, err := mlp.calculatePSD(mlp.synSpeech[smpl:smpl+mlp.fftSize], PSD, mlp.fftWindow, mlp.fftSize)
		if err != nil {
			fmt.Printf("calculatePSD error: %v\n", err)
			return fmt.Errorf("calculatePSD error: %v", err.Error())
		}
		samp.data[i].y = float64(binMax)
	}
	return nil
}

// create synthetic speech patterns consisting of 25ms frames of voiced or unvoiced input
func (mlp *MLP) createPatterns() error {
	nsamples := mlp.fftSize * nffts
	// a block consists of avgDuration samples = 25ms of speech sampled at 8,000 Hz
	nframes := int(math.Ceil(float64(nsamples) / float64(avgDuration)))
	const (
		amplMinf1  = 500.0
		amplMaxf1  = 1000.0
		f1Min      = 200   // Hz = cycles/sec
		f1Max      = 800   // Hz = cycles/sec
		sigmaNoise = 200.0 // unvoiced speech
		nyquist    = sampleRate / 2
		frameScale = 2 // frame extender
	)

	nsubfreq := 0

	// make SpeechFrames for the speech patterns
	mlp.speechPat = make([][]SpeechFrame, npatterns)
	for i := range mlp.speechPat {
		mlp.speechPat[i] = make([]SpeechFrame, nframes)
	}

	// Create the speech patterns for voiced or unvoiced frames
	// loop over number of speech patterns
	for pat := 0; pat < npatterns; pat++ {
		speechfile := filepath.Join(dataDir, fmt.Sprintf("speech%d.csv", pat))
		fspeech, err := os.Create(speechfile)
		if err != nil {
			return fmt.Errorf("createPatterns could not create file %s error: %s", speechfile, err.Error())
		}
		// loop over the number of frames
		for fr := 0; fr < nframes; fr++ {
			// select frequencies and amplitudes depending on voiced/unvoiced
			if fr%frameScale == 0 {
				if 100.0*rand.Float64() < float64(mlp.percentVoiced) {
					// voiced, cycles/sec = Hz
					// 1-5 subfrequencies
					nsubfreq = rand.Intn(maxSubFreq) + 1
					mlp.speechPat[pat][fr].freqs = make([]float64, nsubfreq+1)
					mlp.speechPat[pat][fr].amps = make([]float64, nsubfreq+1)
					// fundamental frequency
					mlp.speechPat[pat][fr].freqs[0] = f1Min + (f1Max-f1Min)*rand.Float64()
					mlp.speechPat[pat][fr].amps[0] = amplMinf1 + (amplMaxf1-amplMinf1)*rand.Float64()
					// subfrequencies
					for sf := 0; sf < nsubfreq; sf++ {
						mlp.speechPat[pat][fr].freqs[sf+1] =
							mlp.speechPat[pat][fr].freqs[sf] + (nyquist-mlp.speechPat[pat][fr].freqs[sf])*rand.Float64()
						mlp.speechPat[pat][fr].amps[sf+1] = mlp.speechPat[pat][fr].amps[sf] * 0.9
					}
				} else {
					// unvoiced is gaussian noise
					nsubfreq = 0
					mlp.speechPat[pat][fr].freqs = make([]float64, nsubfreq+1)
					mlp.speechPat[pat][fr].amps = make([]float64, nsubfreq+1)
					mlp.speechPat[pat][fr].freqs[0] = 0.0
					mlp.speechPat[pat][fr].amps[0] = sigmaNoise * rand.Float64()
				}
			} else {
				// Use previous frame parameters so that the frame is extended
				mlp.speechPat[pat][fr].freqs = make([]float64, nsubfreq+1)
				mlp.speechPat[pat][fr].amps = make([]float64, nsubfreq+1)
				// fundamental frequency
				mlp.speechPat[pat][fr].freqs[0] = mlp.speechPat[pat][fr-1].freqs[0]
				mlp.speechPat[pat][fr].amps[0] = mlp.speechPat[pat][fr-1].amps[0]
				// subfrequencies
				for sf := 0; sf < nsubfreq; sf++ {
					mlp.speechPat[pat][fr].freqs[sf+1] = mlp.speechPat[pat][fr-1].freqs[sf+1]
					mlp.speechPat[pat][fr].amps[sf+1] = mlp.speechPat[pat][fr-1].amps[sf+1]
				}
			}
			// Save the speech frame to disk file
			// frequencies
			for _, freq := range mlp.speechPat[pat][fr].freqs {
				_, err = fmt.Fprintf(fspeech, "%.16f,", freq)
				if err != nil {
					return fmt.Errorf("createPatterns file: %s, frame: %d, freq: %f write error: %s", speechfile, fr, freq, err.Error())
				}
			}
			// amplitudes
			for _, amp := range mlp.speechPat[pat][fr].amps[0 : len(mlp.speechPat[pat][fr].amps)-1] {
				_, err = fmt.Fprintf(fspeech, "%.16f,", amp)
				if err != nil {
					return fmt.Errorf("createPatterns file: %s, frame: %d, ampl: %f write error: %s", speechfile, fr, amp, err.Error())
				}
			}
			_, err = fmt.Fprintf(fspeech, "%.16f\n", mlp.speechPat[pat][fr].amps[len(mlp.speechPat[pat][fr].amps)-1])
			if err != nil {
				return fmt.Errorf("createPatterns file: %s, frame: %d write error: %s", speechfile, fr, err.Error())
			}
		}
		fspeech.Close()
	}
	return nil
}

// synthesize creates synthetic speech using frequencies and amplitudes of sinusoids or gaussian noise
func (mlp *MLP) synthesize(nfreqs int, start int, stop int) error {

	t := 0.0
	step := 1.0 / float64(sampleRate)
	var sum float64
	// calculate speech over the interval
	if start == 0 {
		sum = 0.0
		if nfreqs == 1 {
			sum += mlp.amps[0] * rand.NormFloat64()
		} else {
			for j := 0; j < nfreqs; j++ {
				sum += mlp.amps[j] * math.Sin(2.0*math.Pi*mlp.freqs[j]*t)
			}
		}
		t += step
		mlp.synSpeech[0] = sum
		start++
	}
	for i := start; i < stop; i++ {
		sum = 0.0
		if nfreqs == 1 {
			sum += mlp.amps[0] * rand.NormFloat64()
		} else {
			for j := 0; j < nfreqs; j++ {
				sum += mlp.amps[j] * math.Sin(2.0*math.Pi*mlp.freqs[j]*t)
			}
		}
		t += step
		mlp.synSpeech[i] = 0.5 * (sum + mlp.synSpeech[i-1])
	}
	return nil
}

// createSpeech creates a slice of training/testing synthetic speech based on the speech patterns
func (mlp *MLP) createSpeech(pattern int) error {
	// use the speech patterns and apply deltas for the frequencies, amplitude, and duration so the MLP generalizes
	nsamples := mlp.fftSize * nffts
	remain := nsamples
	nframes := len(mlp.speechPat[pattern])
	// starting sample for current frame
	samp := 0

	const (
		margin1       int = avgDuration - 40
		margin2       int = 2 * margin1
		delSigmaNoise     = 1.0
	)

	// loop over the frame of the pattern and retrieve the attributes of the speech frame
	// add delta for pitch and duration
	for frame := 0; frame < nframes-1; frame++ {
		if remain < margin2 {
			return fmt.Errorf("repeat")
		}
		nfreqs := len(mlp.speechPat[pattern][frame].freqs)
		// unvoiced pattern, frequency = 0
		if nfreqs == 1 {
			if rand.Intn(2) > 0 {
				mlp.amps[0] = mlp.speechPat[pattern][frame].amps[0] + delSigmaNoise*mlp.delAmpl
			} else {
				mlp.amps[0] = mlp.speechPat[pattern][frame].amps[0] - delSigmaNoise*mlp.delAmpl
			}
			mlp.freqs[0] = mlp.speechPat[pattern][frame].freqs[0]
		} else {
			// voiced pattern
			for i := 0; i < nfreqs; i++ {
				if rand.Intn(2) > 0 {
					mlp.freqs[i] =
						mlp.speechPat[pattern][frame].freqs[i] + float64(mlp.delPitch)
				} else {
					mlp.freqs[i] =
						mlp.speechPat[pattern][frame].freqs[i] - float64(mlp.delPitch)
				}

				if rand.Intn(2) > 0 {
					mlp.amps[i] = mlp.speechPat[pattern][frame].amps[i] * (1.0 + float64(mlp.delAmpl))
				} else {
					mlp.amps[i] = mlp.speechPat[pattern][frame].amps[i] * (1.0 - float64(mlp.delAmpl))
				}
			}
		}
		duration := avgDuration
		// add or subtract delta
		if rand.Intn(2) > 0 {
			duration += mlp.delDuration
		} else {
			duration -= mlp.delDuration
		}
		remain -= duration

		err := mlp.synthesize(nfreqs, samp, samp+duration)
		if err != nil {
			return fmt.Errorf("synthesize error: %v", err.Error())
		}
		samp += duration
	}

	if remain < margin1 {
		return fmt.Errorf("repeat")
	}

	nfreqs := len(mlp.speechPat[pattern][nframes-1].freqs)
	// unvoiced pattern, frequency = 0
	if nfreqs == 1 {
		if rand.Intn(2) > 0 {
			mlp.amps[0] = mlp.speechPat[pattern][nframes-1].amps[0] + delSigmaNoise*mlp.delAmpl
		} else {
			mlp.amps[0] = mlp.speechPat[pattern][nframes-1].amps[0] - delSigmaNoise*mlp.delAmpl
		}
		mlp.freqs[0] = mlp.speechPat[pattern][nframes-1].freqs[0]
	} else {
		// voiced pattern
		for i := 1; i < nfreqs; i++ {
			if rand.Intn(2) > 0 {
				mlp.freqs[i] =
					mlp.speechPat[pattern][nframes-1].freqs[i] + float64(mlp.delPitch)/mlp.speechPat[pattern][nframes-1].freqs[i]
			} else {
				mlp.freqs[i] =
					mlp.speechPat[pattern][nframes-1].freqs[i] - float64(mlp.delPitch)/mlp.speechPat[pattern][nframes-1].freqs[i]
			}

			if rand.Intn(2) > 0 {
				mlp.amps[i] = mlp.speechPat[pattern][nframes-1].amps[i] * (1.0 + float64(mlp.delAmpl))
			} else {
				mlp.amps[i] = mlp.speechPat[pattern][nframes-1].amps[i] * (1.0 - float64(mlp.delAmpl))
			}
		}
	}

	err := mlp.synthesize(nfreqs, samp, samp+remain)
	if err != nil {
		return fmt.Errorf("synthesize error: %v", err.Error())
	}

	return nil
}

// newMLP constructs an MLP instance for training
func newMLP(r *http.Request, hiddenLayers int, plot *PlotT) (*MLP, error) {
	// Read the training parameters in the HTML Form

	txt := r.FormValue("layerdepth")
	layerDepth, err := strconv.Atoi(txt)
	if err != nil {
		fmt.Printf("layerdepth int conversion error: %v\n", err)
		return nil, fmt.Errorf("layerdepth int conversion error: %s", err.Error())
	}

	txt = r.FormValue("learningrate")
	learningRate, err := strconv.ParseFloat(txt, 64)
	if err != nil {
		fmt.Printf("learningrate float conversion error: %v\n", err)
		return nil, fmt.Errorf("learningrate float conversion error: %s", err.Error())
	}

	txt = r.FormValue("momentum")
	momentum, err := strconv.ParseFloat(txt, 64)
	if err != nil {
		fmt.Printf("momentum float conversion error: %v\n", err)
		return nil, fmt.Errorf("momentum float conversion error: %s", err.Error())
	}

	txt = r.FormValue("epochs")
	epochs, err := strconv.Atoi(txt)
	if err != nil {
		fmt.Printf("epochs int conversion error: %v\n", err)
		return nil, fmt.Errorf("epochs int conversion error: %s", err.Error())
	}

	fftWindow := r.FormValue("fftwindow")

	txt = r.FormValue("fftsize")
	fftSize, err := strconv.Atoi(txt)
	if err != nil {
		fmt.Printf("fftsize int conversion error: %v\n", err)
		return nil, err
	}

	// Get delta pitch, delta duration, delta amplitude, and percent voiced speech
	txt = r.FormValue("delpitch")
	delPitch, err := strconv.Atoi(txt)
	if err != nil {
		fmt.Printf("delta Pitch conversion error: %v\n", err)
		return nil, err
	}

	txt = r.FormValue("delduration")
	delDuration, err := strconv.Atoi(txt)
	if err != nil {
		fmt.Printf("delta Duration conversion error: %v\n", err)
		return nil, err
	}

	txt = r.FormValue("delampl")
	delAmpl, err := strconv.ParseFloat(txt, 64)
	if err != nil {
		fmt.Printf("delta Amplitude conversion error: %v\n", err)
		return nil, err
	}

	txt = r.FormValue("percentvoiced")
	percentVoiced, err := strconv.Atoi(txt)
	if err != nil {
		fmt.Printf("percent Voiced conversion error: %v\n", err)
		return nil, err
	}

	mlp := MLP{
		hiddenLayers: hiddenLayers,
		layerDepth:   layerDepth,
		epochs:       epochs,
		learningRate: learningRate,
		momentum:     momentum,
		plot:         plot,
		Endpoints: Endpoints{
			ymin: math.MaxFloat64,
			ymax: -math.MaxFloat64,
			xmin: 0,
			xmax: float64(epochs - 1)},
		words:         make([]string, 0),
		fftSize:       fftSize,
		fftWindow:     fftWindow,
		delDuration:   delDuration,
		delPitch:      delPitch,
		delAmpl:       delAmpl,
		percentVoiced: percentVoiced,
		freqs:         make([]float64, maxSubFreq+1),
		amps:          make([]float64, maxSubFreq+1),
	}
	// make SpeechFrames for the speech patterns
	nsamples := mlp.fftSize * nffts
	nframes := int(math.Ceil(float64(nsamples) / float64(avgDuration)))

	// number of outer layer nodes
	olnodes := classes

	// input layer nodes are largest PSD bins for each STFT, plus bias node
	ilnodes := nframes + 1

	// construct link that holds the weights and weight deltas
	mlp.link = make([][]Link, hiddenLayers+1)

	// input layer
	mlp.link[0] = make([]Link, ilnodes*layerDepth)

	// output layer links
	mlp.link[len(mlp.link)-1] = make([]Link, olnodes*(layerDepth+1))

	// hidden layer links
	for i := 1; i < len(mlp.link)-1; i++ {
		mlp.link[i] = make([]Link, (layerDepth+1)*layerDepth)
	}

	// construct node, init node[i][0].y to 1.0 (bias)
	mlp.node = make([][]Node, hiddenLayers+2)

	// input layer
	mlp.node[0] = make([]Node, ilnodes)
	// set first node in the layer (bias) to 1
	mlp.node[0][0].y = 1.0

	// output layer, which has no bias node
	mlp.node[hiddenLayers+1] = make([]Node, olnodes)

	// hidden layers
	for i := 1; i <= hiddenLayers; i++ {
		mlp.node[i] = make([]Node, layerDepth+1)
		// set first node in the layer (bias) to 1
		mlp.node[i][0].y = 1.0
	}

	// construct desired from classes, binary representation
	mlp.desired = make([]float64, olnodes)

	// construct desired from classes, one-hot vector
	mlp.desired = make([]float64, olnodes)

	// cross-entropy loss
	mlp.data = make([]float64, epochs)

	// synthetic speech for creating speech with synthesize
	mlp.synSpeech = make([]float64, nsamples)

	return &mlp, nil
}

// gridFillInterp inserts the data points in the grid and draws a straight line between points
func (mlp *MLP) gridFillInterp() error {
	var (
		x            float64 = 0.0
		y            float64 = mlp.data[0]
		prevX, prevY float64
		xscale       float64
		yscale       float64
	)

	// Mark the data x-y coordinate online at the corresponding
	// grid row/column.

	// Calculate scale factors for x and y
	xscale = float64(cols-1) / (mlp.xmax - mlp.xmin)
	yscale = float64(rows-1) / (mlp.ymax - mlp.ymin)

	mlp.plot.Grid = make([]string, rows*cols)

	// This cell location (row,col) is on the line
	row := int((mlp.ymax-y)*yscale + .5)
	col := int((x-mlp.xmin)*xscale + .5)
	mlp.plot.Grid[row*cols+col] = "online"

	prevX = x
	prevY = y

	// Scale factor to determine the number of interpolation points
	lenEPy := mlp.ymax - mlp.ymin
	lenEPx := mlp.xmax - mlp.xmin

	// Continue with the rest of the points in the file
	for i := 1; i < len(mlp.data); i++ {
		x++
		// mse/epoch or percent-correct/pattern
		y = mlp.data[i]

		// This cell location (row,col) is on the line
		row := int((mlp.ymax-y)*yscale + .5)
		col := int((x-mlp.xmin)*xscale + .5)
		mlp.plot.Grid[row*cols+col] = "online"

		// Interpolate the points between previous point and current point

		/* lenEdge := math.Sqrt((x-prevX)*(x-prevX) + (y-prevY)*(y-prevY)) */
		lenEdgeX := math.Abs((x - prevX))
		lenEdgeY := math.Abs(y - prevY)
		ncellsX := int(float64(cols) * lenEdgeX / lenEPx) // number of points to interpolate in x-dim
		ncellsY := int(float64(rows) * lenEdgeY / lenEPy) // number of points to interpolate in y-dim
		// Choose the biggest
		ncells := max(ncellsY, ncellsX)

		stepX := (x - prevX) / float64(ncells)
		stepY := (y - prevY) / float64(ncells)

		// loop to draw the points
		interpX := prevX
		interpY := prevY
		for i := 0; i < ncells; i++ {
			row := int((mlp.ymax-interpY)*yscale + .5)
			col := int((interpX-mlp.xmin)*xscale + .5)
			mlp.plot.Grid[row*cols+col] = "online"
			interpX += stepX
			interpY += stepY
		}

		// Update the previous point with the current point
		prevX = x
		prevY = y
	}
	return nil
}

// insertLabels inserts x- an y-axis labels in the plot
func (mlp *MLP) insertLabels() {
	mlp.plot.Xlabel = make([]string, xlabels)
	mlp.plot.Ylabel = make([]string, ylabels)
	// Construct x-axis labels
	incr := (mlp.xmax - mlp.xmin) / (xlabels - 1)
	x := mlp.xmin
	// First label is empty for alignment purposes
	for i := range mlp.plot.Xlabel {
		mlp.plot.Xlabel[i] = fmt.Sprintf("%.2f", x)
		x += incr
	}

	// Construct the y-axis labels
	incr = (mlp.ymax - mlp.ymin) / (ylabels - 1)
	y := mlp.ymin
	for i := range mlp.plot.Ylabel {
		mlp.plot.Ylabel[i] = fmt.Sprintf("%.2f", y)
		y += incr
	}
}

// handleTraining performs forward and backward propagation to calculate the weights
func handleTrainingMLP(w http.ResponseWriter, r *http.Request) {

	var (
		plot PlotT
		mlp  *MLP
	)

	// Get the number of hidden layers
	txt := r.FormValue("hiddenlayers")
	// Need hidden layers to continue
	if len(txt) > 0 {
		hiddenLayers, err := strconv.Atoi(txt)
		if err != nil {
			fmt.Printf("Hidden Layers int conversion error: %v\n", err)
			plot.Status = fmt.Sprintf("Hidden Layers conversion to int error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplTrainingMLP.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// create MLP instance to hold state
		mlp, err = newMLP(r, hiddenLayers, &plot)
		if err != nil {
			fmt.Printf("newMLP() error: %v\n", err)
			plot.Status = fmt.Sprintf("newMLP() error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplTrainingMLP.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// create new synthetic speech patterns
		newPattern := r.FormValue("speechpattern")
		if newPattern == "new" {
			if err = mlp.createPatterns(); err != nil {
				fmt.Printf("createPatterns() error: %v\n", err)
				plot.Status = fmt.Sprintf("createPatterns() error: %v", err.Error())
				// Write to HTTP using template and grid
				if err := tmplTrainingMLP.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			}
			// Read the synthetic speech patterns
		} else {
			files, err := os.ReadDir(dataDir)
			if err != nil {
				fmt.Printf("ReadDir %s error: %v\n", dataDir, err)
				plot.Status = fmt.Sprintf("ReadDir %s error: %v", dataDir, err.Error())
				// Write to HTTP using template and grid
				if err := tmplTrainingMLP.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			}
			if len(files) == 0 {
				fmt.Printf("No synthetic speech files in %s\n", dataDir)
				plot.Status = fmt.Sprintf("No filter files in %s", dataDir)
				// Write to HTTP using template and grid
				if err := tmplTrainingMLP.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			} else {
				// make SpeechFrames for the speech patterns
				nsamples := mlp.fftSize * nffts
				// a frame consists of avgDuration samples = 25ms of speech sampled at 8,000 Hz
				nframes := int(math.Ceil(float64(nsamples) / float64(avgDuration)))
				mlp.speechPat = make([][]SpeechFrame, npatterns)
				for i := range mlp.speechPat {
					mlp.speechPat[i] = make([]SpeechFrame, nframes)
				}
				pattern := 0
				// Retrieve the speech files
				for _, dirEntry := range files {
					name := dirEntry.Name()
					if strings.Contains(name, "speech") && strings.Contains(name, "csv") {
						fspeech, err := os.Open(filepath.Join(dataDir, name))
						if err != nil {
							fmt.Printf("Open %s error: %v\n", name, err)
							plot.Status = fmt.Sprintf("Open %s error: %v", name, err.Error())
							// Write to HTTP using template and grid
							if err := tmplTrainingMLP.Execute(w, plot); err != nil {
								log.Fatalf("Write to HTTP output using template with error: %v\n", err)
							}
							return
						}
						scanner := bufio.NewScanner(fspeech)
						frame := 0
						for scanner.Scan() {
							line := scanner.Text()
							items := strings.Split(line, ",")
							nfreqs := len(items) / 2
							mlp.speechPat[pattern][frame].freqs = make([]float64, nfreqs)
							mlp.speechPat[pattern][frame].amps = make([]float64, nfreqs)
							for i := 0; i < nfreqs; i++ {
								freq, err := strconv.ParseFloat(items[i], 64)
								if err != nil {
									plot.Status = fmt.Sprintf("freq %d conversion error: %v", i, err.Error())
									// Write to HTTP using template and grid
									if err := tmplTrainingMLP.Execute(w, plot); err != nil {
										log.Fatalf("Write to HTTP output using template with error: %v\n", err)
									}
									return
								}
								ampl, err := strconv.ParseFloat(items[i+nfreqs], 64)
								if err != nil {
									plot.Status = fmt.Sprintf("ampl %d conversion error: %v", i, err.Error())
									// Write to HTTP using template and grid
									if err := tmplTrainingMLP.Execute(w, plot); err != nil {
										log.Fatalf("Write to HTTP output using template with error: %v\n", err)
									}
									return
								}
								mlp.speechPat[pattern][frame].amps[i] = ampl
								mlp.speechPat[pattern][frame].freqs[i] = freq
							}
							frame++
						}
						fspeech.Close()
						if err = scanner.Err(); err != nil {
							fmt.Printf("speech file scanner error: %s", err.Error())
							// Write to HTTP using template and grid
							if err := tmplTrainingMLP.Execute(w, plot); err != nil {
								log.Fatalf("Write to HTTP output using template with error: %v\n", err)
							}
							return
						}
						pattern++
					}
				}
			}
		}
		// Loop over the Epochs
		err = mlp.runTrainingEpochs()
		if err != nil {
			fmt.Printf("runTrainingEpochs() error: %v\n", err)
			plot.Status = fmt.Sprintf("runTrainingEpochs() error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplTrainingMLP.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// Put cross entropy vs Epoch in PlotT
		err = mlp.gridFillInterp()
		if err != nil {
			fmt.Printf("gridFillInterp() error: %v\n", err)
			plot.Status = fmt.Sprintf("gridFillInterp() error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplTrainingMLP.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// insert x-labels and y-labels in PlotT
		mlp.insertLabels()

		// At the end of all epochs, insert form previous control items in PlotT
		mlp.plot.HiddenLayers = strconv.Itoa(mlp.hiddenLayers)
		mlp.plot.LayerDepth = strconv.Itoa(mlp.layerDepth)
		mlp.plot.LearningRate = strconv.FormatFloat(mlp.learningRate, 'f', 4, 64)
		mlp.plot.Momentum = strconv.FormatFloat(mlp.momentum, 'f', 4, 64)
		mlp.plot.Epochs = strconv.Itoa(mlp.epochs)
		mlp.plot.DelDuration = strconv.Itoa(mlp.delDuration)
		mlp.plot.DelPitch = strconv.Itoa(mlp.delPitch)
		mlp.plot.PercentVoiced = strconv.Itoa(mlp.percentVoiced)
		mlp.plot.DelAmpl = strconv.FormatFloat(mlp.delAmpl, 'f', 3, 64)

		// Save hidden layers, hidden layer depth, classes, epochs, fft size, fft window,
		// window, percent voiced, delta duration, delta pitch and Filters/weights to csv file
		f, err := os.Create(path.Join(dataDir, fileweights))
		if err != nil {
			fmt.Printf("os.Create() file %s error: %v\n", path.Join(fileweights), err)
			plot.Status = fmt.Sprintf("os.Create() file %s error: %v", path.Join(fileweights), err.Error())
			// Write to HTTP using template and grid
			if err := tmplTrainingMLP.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}
		defer f.Close()
		// save MLP parameters
		fmt.Fprintf(f, "%d,%d,%d,%f,%f,%d,%s,%d,%d,%d,%f\n",
			mlp.epochs, mlp.hiddenLayers, mlp.layerDepth, mlp.learningRate, mlp.momentum, mlp.fftSize,
			mlp.fftWindow, mlp.delPitch, mlp.delDuration, mlp.percentVoiced, mlp.delAmpl)

		// save weights, layer by layer
		// save first layer, one weight per line because too long to scan in
		for _, node := range mlp.link[0] {
			fmt.Fprintf(f, "%.16f\n", node.wgt)
		}
		// save remaining layers one layer per line with csv
		for _, layer := range mlp.link[1:] {
			for _, node := range layer {
				fmt.Fprintf(f, "%.16f,", node.wgt)
			}
			fmt.Fprintln(f)
		}

		mlp.plot.Status = "Cross-Entropy Loss plotted"

		// Execute data on HTML template
		if err = tmplTrainingMLP.Execute(w, mlp.plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
	} else {
		plot.Status = "Enter MLP Neural Network training parameters."
		// Write to HTTP using template and grid
		if err := tmplTrainingMLP.Execute(w, plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
		return
	}
}

// Welch's Method and Bartlett's Method variation of the Periodogram
func (mlp *MLP) calculatePSD(audio []float64, PSD []float64, fftWindow string, fftSize int) (int, float64, error) {

	N := fftSize
	m := N / 2

	// map of window functions
	window := make(map[string]Window, len(winType))
	// Put the window functions in the map
	window["Bartlett"] = bartlett
	window["Welch"] = welch
	window["Hamming"] = hamming
	window["Hanning"] = hanning
	window["Rectangle"] = rectangle

	w, ok := window[fftWindow]
	if !ok {
		fmt.Printf("Invalid FFT window type: %v\n", fftWindow)
		return 0, 0, fmt.Errorf("invalid FFT window type: %v", fftWindow)
	}

	bufN := make([]complex128, N)

	for j := 0; j < len(audio); j++ {
		bufN[j] = complex(audio[j], 0)
	}

	// zero-pad the remaining samples
	for i := len(audio); i < N; i++ {
		bufN[i] = 0
	}

	// window the N samples with chosen window
	for k := 0; k < N; k++ {
		bufN[k] *= w(k, m)
	}

	// Perform N-point complex FFT and add squares to previous values in PSD
	fourierN := fft.FFT(bufN)
	x := cmplx.Abs(fourierN[0])
	PSD[0] = x * x
	psdMax := PSD[0]
	binMax := 0
	for j := 1; j < m; j++ {
		// Use positive and negative frequencies -> bufN[N-j] = bufN[-j]
		xj := cmplx.Abs(fourierN[j])
		xNj := cmplx.Abs(fourierN[N-j])
		PSD[j] = xj*xj + xNj*xNj
		if PSD[j] > psdMax {
			psdMax = PSD[j]
			binMax = j
		}
	}

	return binMax, psdMax, nil
}

// runTestingEpochs classifies test examples and tabulates test results
func (mlp *MLP) runTestingEpochs() error {

	// number of samples per pattern, number of frames in the pattern
	nsamples := mlp.fftSize * nffts
	nframes := int(math.Ceil(float64(nsamples) / float64(avgDuration)))

	// Create a sample for containing the spectrogram to propagate forward
	samp := Sample{data: make([]Node, nframes)}
	mlp.statistics =
		Stats{correct: make([]int, classes), classCount: make([]int, classes)}

	for n := 0; n < mlp.epochs; n++ {
		// create speech for one pattern
		// Randomly choose a pattern
		pattern := rand.Intn(npatterns)
		err := mlp.createSpeech(pattern)
		for err != nil {
			if err.Error() == "repeat" {
				err = mlp.createSpeech(pattern)
			} else {
				fmt.Printf("createSpeech error: %v\n", err.Error())
				return fmt.Errorf("createSpeech error: %v", err.Error())
			}
		}

		samp.desired = pattern

		// create spectrogram
		err = mlp.createSpectrogram(&samp)
		if err != nil {
			fmt.Printf("createSpectrogram error: %v\n", err.Error())
			return fmt.Errorf("createSpectrogram error: %v", err.Error())
		}

		// Forward Propagation
		err = mlp.propagateForward(&samp)
		if err != nil {
			return fmt.Errorf("forward propagation error: %s", err.Error())
		}

		// At the end of each epoch, classify the result
		err = mlp.determineClass(&samp)
		if err != nil {
			return fmt.Errorf("determineClass error: %s", err.Error())
		}
	}

	mlp.plot.TestResults = make([]Results, classes)

	totalCount := 0
	totalCorrect := 0
	classCount := 0
	// tabulate TestResults by converting numbers to string in Results
	for i := range mlp.plot.TestResults {
		classCount = mlp.statistics.classCount[i]
		totalCount += classCount
		totalCorrect += mlp.statistics.correct[i]
		if classCount > 0 {
			mlp.plot.TestResults[i] = Results{
				Class:   strconv.Itoa(i),
				Count:   strconv.Itoa(classCount),
				Correct: strconv.Itoa(mlp.statistics.correct[i] * 100 / classCount),
			}
		} else {
			mlp.plot.TestResults[i] = Results{
				Class:   strconv.Itoa(i),
				Count:   strconv.Itoa(classCount),
				Correct: "0",
			}
		}
	}
	mlp.plot.TotalCount = strconv.Itoa(totalCount)
	mlp.plot.TotalCorrect = strconv.Itoa(totalCorrect * 100 / totalCount)
	mlp.plot.LearningRate = strconv.FormatFloat(mlp.learningRate, 'f', 4, 64)
	mlp.plot.Epochs = strconv.Itoa(mlp.epochs)

	mlp.plot.Status = "Testing results completed."

	mlp.plot.LearningRate = strconv.FormatFloat(mlp.learningRate, 'f', -1, 64)
	mlp.plot.Momentum = strconv.FormatFloat(mlp.momentum, 'f', -1, 64)
	mlp.plot.HiddenLayers = strconv.Itoa(mlp.hiddenLayers)
	mlp.plot.LayerDepth = strconv.Itoa(mlp.layerDepth)
	mlp.plot.Epochs = strconv.Itoa(mlp.epochs)
	mlp.plot.FFTSize = strconv.Itoa(mlp.fftSize)
	mlp.plot.FFTWindow = mlp.fftWindow
	mlp.plot.Classes = strconv.Itoa(classes)
	mlp.plot.DelPitch = strconv.Itoa(mlp.delPitch)
	mlp.plot.DelDuration = strconv.Itoa(mlp.delDuration)
	mlp.plot.PercentVoiced = strconv.Itoa(mlp.percentVoiced)
	mlp.plot.DelAmpl = strconv.FormatFloat(mlp.delAmpl, 'f', -1, 64)

	return nil
}

// newTestingMLP constructs an MLP from the saved mlp weights and parameters
func newTestingMLP(plot *PlotT) (*MLP, error) {

	// Read in weights from csv file, ordered by layers, and MLP parameters
	f, err := os.Open(path.Join(dataDir, fileweights))
	if err != nil {
		fmt.Printf("Open file %s error: %v", fileweights, err)
		return nil, fmt.Errorf("open file %s error: %s", fileweights, err.Error())
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	// get the parameters
	scanner.Scan()
	line := scanner.Text()

	items := strings.Split(line, ",")
	if len(items) != 11 {
		fmt.Printf("Testing parameters missing, should be 9, is %d\n", len(items))
		return nil, fmt.Errorf("testing parameters missing, run Train first")
	}

	epochs, err := strconv.Atoi(items[0])
	if err != nil {
		fmt.Printf("Conversion to int of %s error: %v\n", items[0], err)
		return nil, err
	}

	hiddenLayers, err := strconv.Atoi(items[1])
	if err != nil {
		fmt.Printf("Conversion to int of %s error: %v\n", items[1], err)
		return nil, err
	}
	layerDepth, err := strconv.Atoi(items[2])
	if err != nil {
		fmt.Printf("Conversion to int of %s error: %v\n", items[2], err)
		return nil, err
	}

	learningRate, err := strconv.ParseFloat(items[3], 64)
	if err != nil {
		fmt.Printf("Conversion to float of %s error: %v\n", items[4], err)
		return nil, err
	}

	momentum, err := strconv.ParseFloat(items[4], 64)
	if err != nil {
		fmt.Printf("Conversion to float of %s error: %v\n", items[5], err)
		return nil, err
	}

	fftSize, err := strconv.Atoi(items[5])
	if err != nil {
		fmt.Printf("Conversion to int of 'fftSize' error: %v\n", err)
		return nil, err
	}

	fftWindow := items[6]

	delPitch, err := strconv.Atoi(items[7])
	if err != nil {
		fmt.Printf("Conversion to int of 'delPitch' error: %v\n", err)
		return nil, err
	}

	delDuration, err := strconv.Atoi(items[8])
	if err != nil {
		fmt.Printf("Conversion to int of delDuration' error: %v\n", err)
		return nil, err
	}

	percentVoiced, err := strconv.Atoi(items[9])
	if err != nil {
		fmt.Printf("Conversion to int of percentVoiced error: %v\n", err)
		return nil, err
	}

	delAmpl, err := strconv.ParseFloat(items[10], 64)
	if err != nil {
		fmt.Printf("Conversion to float of delAmpl error: %v\n", err)
		return nil, err
	}

	// construct the MLP
	mlp := MLP{
		epochs:       epochs,
		hiddenLayers: hiddenLayers,
		layerDepth:   layerDepth,
		plot:         plot,
		Endpoints: Endpoints{
			ymin: 0.0,
			ymax: 100.0,
			xmin: 0,
			xmax: float64(npatterns - 1)},
		learningRate:  learningRate,
		momentum:      momentum,
		words:         make([]string, 0),
		fftSize:       fftSize,
		fftWindow:     fftWindow,
		delDuration:   delDuration,
		delPitch:      delPitch,
		delAmpl:       delAmpl,
		percentVoiced: percentVoiced,
		freqs:         make([]float64, maxSubFreq+1),
		amps:          make([]float64, maxSubFreq+1),
	}

	// make SpeechFrames for the speech patterns
	nsamples := mlp.fftSize * nffts
	nframes := int(math.Ceil(float64(nsamples) / float64(avgDuration)))

	// construct link that holds the weights and weight deltas
	mlp.link = make([][]Link, hiddenLayers+1)

	// input layer nodes are largest PSD bins for each STFT, plus bias node
	ilnodes := nframes + 1
	nwgts := ilnodes * layerDepth

	layer := 0
	mlp.link[layer] = make([]Link, nwgts)
	for i := 0; i < nwgts; i++ {
		scanner.Scan()
		line := scanner.Text()
		wgt, err := strconv.ParseFloat(line, 64)
		if err != nil {
			fmt.Printf("ParseFloat error: %v\n", err.Error())
			continue
		}
		mlp.link[0][i] = Link{wgt: wgt, wgtDelta: 0}
	}
	layer++
	// Continue with remaining layers, one layer per line
	for scanner.Scan() {
		line = scanner.Text()
		weights := strings.Split(line, ",")
		weights = weights[:len(weights)-1]
		mlp.link[layer] = make([]Link, len(weights))
		for i, wtStr := range weights {
			wt, err := strconv.ParseFloat(wtStr, 64)
			if err != nil {
				fmt.Printf("ParseFloat of %s error: %v", wtStr, err)
				continue
			}
			mlp.link[layer][i] = Link{wgt: wt, wgtDelta: 0}
		}
		layer++
	}
	if err = scanner.Err(); err != nil {
		fmt.Printf("scanner error: %s\n", err.Error())
		return nil, fmt.Errorf("scanner error: %v", err)
	}

	// number of outer layer nodes
	olnodes := classes

	// construct node, init node[i][0].y to 1.0 (bias)
	mlp.node = make([][]Node, hiddenLayers+2)

	// input layer
	mlp.node[0] = make([]Node, ilnodes)
	// set first node in the layer (bias) to 1
	mlp.node[0][0].y = 1.0

	// output layer, which has no bias node
	mlp.node[hiddenLayers+1] = make([]Node, olnodes)

	// hidden layers
	for i := 1; i <= hiddenLayers; i++ {
		mlp.node[i] = make([]Node, layerDepth+1)
		// set first node in the layer (bias) to 1
		mlp.node[i][0].y = 1.0
	}

	// *********************************************************
	// construct desired from classes, one-hot vector
	mlp.desired = make([]float64, olnodes)

	// percent correct classification of speech patterns
	mlp.data = make([]float64, npatterns)

	// synthetic speech for creating speech with synthesize
	mlp.synSpeech = make([]float64, nsamples)

	// make SpeechFrames for the speech patterns
	// a frame consists of avgDuration samples = 25ms of speech sampled at 8,000 Hz
	mlp.speechPat = make([][]SpeechFrame, npatterns)
	for i := range mlp.speechPat {
		mlp.speechPat[i] = make([]SpeechFrame, nframes)
	}

	files, err := os.ReadDir(dataDir)
	if err != nil {
		fmt.Printf("ReadDir %s error: %v\n", dataDir, err)
		return nil, fmt.Errorf("ReadDir %s error: %v", dataDir, err)
	}
	if len(files) == 0 {
		fmt.Printf("No synthetic speech files in %s\n", dataDir)
		return nil, fmt.Errorf("no synthetic speech files in %s", dataDir)
	} else {
		pattern := 0
		// Retrieve the speech files
		for _, dirEntry := range files {
			name := dirEntry.Name()
			if strings.Contains(name, "speech") && strings.Contains(name, "csv") {
				fspeech, err := os.Open(filepath.Join(dataDir, name))
				if err != nil {
					fmt.Printf("Open %s error: %v\n", name, err)
					return nil, fmt.Errorf("open error: %s", err.Error())
				}
				scanner := bufio.NewScanner(fspeech)
				frame := 0
				for scanner.Scan() {
					line := scanner.Text()
					items := strings.Split(line, ",")
					nfreqs := len(items) / 2
					mlp.speechPat[pattern][frame].freqs = make([]float64, nfreqs)
					mlp.speechPat[pattern][frame].amps = make([]float64, nfreqs)
					for i := 0; i < nfreqs; i++ {
						freq, err := strconv.ParseFloat(items[i], 64)
						if err != nil {
							return nil, fmt.Errorf("freq %d conversion error: %v", i, err.Error())
						}
						ampl, err := strconv.ParseFloat(items[i+nfreqs], 64)
						if err != nil {
							return nil, fmt.Errorf("ampl %d conversion error: %v", i, err.Error())
						}
						mlp.speechPat[pattern][frame].amps[i] = ampl
						mlp.speechPat[pattern][frame].freqs[i] = freq
					}
					frame++
				}
				fspeech.Close()
				if err = scanner.Err(); err != nil {
					fmt.Printf("speech file scanner error: %s", err.Error())
					return nil, fmt.Errorf("speech file scanner error: %s", err.Error())
				}
				fspeech.Close()
				pattern++
			}
			if err = scanner.Err(); err != nil {
				return nil, fmt.Errorf("speech parameter scanner error: %s", err.Error())
			}
		}
	}
	return &mlp, nil
}

// handleTestingMLP performs pattern classification of the test data
func handleTestingMLP(w http.ResponseWriter, r *http.Request) {
	// create synthetic speech
	// loop over the synthetic speech and generate the spectrogram
	// propagate forward and classify the output
	// fill the grid with the percent correct

	var (
		plot PlotT
		mlp  *MLP
		err  error
	)

	// Construct MLP instance containing MLP state
	mlp, err = newTestingMLP(&plot)
	if err != nil {
		fmt.Printf("newTestingMLP() error: %v\n", err)
		plot.Status = fmt.Sprintf("newTestingMLP() error: %v", err.Error())
		// Write to HTTP using template and grid
		if err := tmplTestingMLP.Execute(w, plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
		return
	}

	// At end of all examples display TestingResults
	// Convert classification numbers to string in Results
	err = mlp.runTestingEpochs()
	if err != nil {
		fmt.Printf("runTestingEpochs() error: %v\n", err)
		plot.Status = fmt.Sprintf("runTestingEpochs() error: %v", err.Error())
		// Write to HTTP using template and grid
		if err := tmplTestingMLP.Execute(w, plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
		return
	}

	// Put the percent correct in data
	for pat := range mlp.data {
		mlp.data[pat] = float64(mlp.statistics.correct[pat]) / float64(mlp.statistics.classCount[pat]) * 100.0
	}
	// Put data in PlotT
	err = mlp.gridFillInterp()
	if err != nil {
		fmt.Printf("gridFillInterp() error: %v\n", err)
		plot.Status = fmt.Sprintf("gridFillInterp() error: %v", err.Error())
		// Write to HTTP using template and grid
		if err := tmplTrainingMLP.Execute(w, plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
		return
	}

	// insert x-labels and y-labels in PlotT
	mlp.insertLabels()

	// Execute data on HTML template
	if err = tmplTestingMLP.Execute(w, mlp.plot); err != nil {
		log.Fatalf("Write to HTTP output using template with error: %v\n", err)
	}
}

// findEndpoints finds the minimum and maximum data values
func (ep *Endpoints) findEndpoints(input []float64) {
	ep.ymax = -math.MaxFloat64
	ep.ymin = math.MaxFloat64
	for _, y := range input {

		if y > ep.ymax {
			ep.ymax = y
		}
		if y < ep.ymin {
			ep.ymin = y
		}
	}
}

// processTimeDomain plots the time domain data of the synthetic speech
func (mlp *MLP) processTimeDomain(pattern int) error {

	var (
		xscale    float64
		yscale    float64
		endpoints Endpoints
	)

	mlp.plot.Grid = make([]string, rows*cols)
	mlp.plot.Xlabel = make([]string, xlabels)
	mlp.plot.Ylabel = make([]string, ylabels)

	mlp.nsamples = len(mlp.synSpeech)

	endpoints.findEndpoints(mlp.synSpeech)
	// time starts at 0 and ends at #samples*sampling period
	endpoints.xmin = 0.0
	// #samples*sampling period, sampling period = 1/sampleRate
	endpoints.xmax = float64(mlp.nsamples) / float64(sampleRate)

	// EP means endpoints
	lenEPx := endpoints.xmax - endpoints.xmin
	lenEPy := endpoints.ymax - endpoints.ymin
	prevTime := 0.0
	prevAmpl := mlp.synSpeech[0]

	// Calculate scale factors for x and y
	xscale = float64(cols-1) / (endpoints.xmax - endpoints.xmin)
	yscale = float64(rows-1) / (endpoints.ymax - endpoints.ymin)

	// This previous cell location (row,col) is on the line (visible)
	row := int((endpoints.ymax-mlp.synSpeech[0])*yscale + .5)
	col := int((0.0-endpoints.xmin)*xscale + .5)
	mlp.plot.Grid[row*cols+col] = "online"

	// Store the amplitude in the plot Grid
	for n := 1; n < mlp.nsamples; n++ {
		// Current time
		currTime := float64(n) / float64(sampleRate)

		// This current cell location (row,col) is on the line (visible)
		row := int((endpoints.ymax-mlp.synSpeech[n])*yscale + .5)
		col := int((currTime-endpoints.xmin)*xscale + .5)
		mlp.plot.Grid[row*cols+col] = "online"

		// Interpolate the points between previous point and current point;
		// draw a straight line between points.
		lenEdgeTime := math.Abs((currTime - prevTime))
		lenEdgeAmpl := math.Abs(mlp.synSpeech[n] - prevAmpl)
		ncellsTime := int(float64(cols) * lenEdgeTime / lenEPx) // number of points to interpolate in x-dim
		ncellsAmpl := int(float64(rows) * lenEdgeAmpl / lenEPy) // number of points to interpolate in y-dim
		// Choose the biggest
		ncells := max(ncellsAmpl, ncellsTime)

		stepTime := float64(currTime-prevTime) / float64(ncells)
		stepAmpl := float64(mlp.synSpeech[n]-prevAmpl) / float64(ncells)

		// loop to draw the points
		interpTime := prevTime
		interpAmpl := prevAmpl
		for i := 0; i < ncells; i++ {
			row := int((endpoints.ymax-interpAmpl)*yscale + .5)
			col := int((interpTime-endpoints.xmin)*xscale + .5)
			// This cell location (row,col) is on the line (visible)
			mlp.plot.Grid[row*cols+col] = "online"
			interpTime += stepTime
			interpAmpl += stepAmpl
		}

		// Update the previous point with the current point
		prevTime = currTime
		prevAmpl = mlp.synSpeech[n]

	}

	// Set plot status if no errors
	if len(mlp.plot.Status) == 0 {
		mlp.plot.Status = fmt.Sprintf("speech pattern %d plotted from (%.3f,%.3f) to (%.3f,%.3f)",
			pattern, endpoints.xmin, endpoints.ymin, endpoints.xmax, endpoints.ymax)
	}

	// Construct x-axis labels
	incr := (endpoints.xmax - endpoints.xmin) / (xlabels - 1)
	x := endpoints.xmin
	// First label is empty for alignment purposes
	for i := range mlp.plot.Xlabel {
		mlp.plot.Xlabel[i] = fmt.Sprintf("%.2f", x)
		x += incr
	}

	// Construct the y-axis labels
	incr = (endpoints.ymax - endpoints.ymin) / (ylabels - 1)
	y := endpoints.ymin
	for i := range mlp.plot.Ylabel {
		mlp.plot.Ylabel[i] = fmt.Sprintf("%.2f", y)
		y += incr
	}

	return nil
}

// processSpectrogram creates a spectrogram of the speech waveform
func (mlp *MLP) processSpectrogram(pattern int, fftWindow string, fftSize int) error {

	// get speech samples from mlp.synSpeech
	var (
		endpoints Endpoints
		PSD       []float64 // power spectral density
		xscale    float64   // data to grid in x direction
		yscale    float64   // data to grid in y direction
	)
	fftSize2 := fftSize / 2

	mlp.plot.Grid = make([]string, rows*cols)
	mlp.plot.Xlabel = make([]string, xlabels)
	mlp.plot.Ylabel = make([]string, ylabels)

	// Power Spectral Density, PSD[N/2] is the Nyquist critical frequency
	// It is (sampling frequency)/2, the highest non-aliased frequency
	PSD = make([]float64, fftSize/2)

	mlp.nsamples = len(mlp.synSpeech)
	// x-axis is time or sample, y-axis is frequency
	endpoints.xmin = 0.0
	endpoints.xmax = float64(mlp.nsamples)
	endpoints.ymin = 0.0
	endpoints.ymax = float64(fftSize2) // equivalent to Nyquist critical frequency

	// Calculate scale factors to convert physical units to screen units
	xscale = float64(cols-1) / (endpoints.xmax - endpoints.xmin)
	yscale = float64(rows-1) / (endpoints.ymax - endpoints.ymin)

	// number of cells to interpolate in time and frequency, stepping by fftSize/2 in time for ncellst
	// round up so the cells in the plot grid are connected
	ncellst := int((math.Ceil(float64(cols) * float64(fftSize2) / float64(mlp.nsamples))))
	ncellsf := int(math.Ceil(float64(rows) / float64((fftSize2))))

	stepTime := float64((fftSize) / ncellst)
	stepFreq := 1.0 / float64(ncellsf)

	// for loop over samples, increment by fftSize/2, calculatePSD on the batch
	// Overlap by 50% due to non-rectangular window to avoid Gibbs phenomenon
	for smpl := 0; smpl < mlp.nsamples; smpl += fftSize2 {
		// calculate the PSD using Bartlett's or Welch's variant of the Periodogram
		end := smpl + fftSize
		if end > mlp.nsamples {
			end = mlp.nsamples
		}
		_, psdMax, err := mlp.calculatePSD(mlp.synSpeech[smpl:end], PSD, fftWindow, fftSize)
		if err != nil {
			fmt.Printf("calculatePSD error: %v\n", err)
			return fmt.Errorf("calculatePSD error: %v", err.Error())
		}

		// for loop over the frequency bins in the PSD
		for bin := 0; bin < fftSize2; bin++ {
			// find the grayscale color based on bin power
			// largest power is black, smallest power is white
			// shades of gray in-between black and white
			var gs string
			r := PSD[bin] / psdMax
			if r < .10 {
				gs = mlp.grayscale[4]
			} else if r < .25 {
				gs = mlp.grayscale[3]
			} else if r < .50 {
				gs = mlp.grayscale[2]
			} else if r < .80 {
				gs = mlp.grayscale[1]
			} else {
				gs = mlp.grayscale[0]
			}

			// interpolate in time
			interpTime := float64(smpl)
			for nct := 0; nct < ncellst; nct++ {
				col := int((interpTime-endpoints.xmin)*xscale + .5)
				if col >= cols {
					col = cols - 1
				}
				// interpolate in frequency
				interpFreq := float64(bin)
				for ncf := 0; ncf < ncellsf; ncf++ {
					row := int((endpoints.ymax-interpFreq)*yscale + .5)
					if row < 0 {
						row = 0
					}
					// Store the color in the plot Grid
					mlp.plot.Grid[row*cols+col] = gs
					interpFreq += stepFreq
				}
				interpTime += stepTime
			}
		}
	}

	// Set plot status if no errors
	if len(mlp.plot.Status) == 0 {
		mlp.plot.Status = fmt.Sprintf("spectrogram of pattern %d plotted from (%.3f,%.3f) to (%.3f,%.3f)",
			pattern, endpoints.xmin, endpoints.ymin, endpoints.xmax, endpoints.ymax)
	}

	// Construct x-axis labels
	incr := (endpoints.xmax - endpoints.xmin) / ((xlabels - 1) * sampleRate)
	x := endpoints.xmin / sampleRate
	// First label is empty for alignment purposes
	for i := range mlp.plot.Xlabel {
		mlp.plot.Xlabel[i] = fmt.Sprintf("%.2f", x)
		x += incr
	}

	// Apply the  sampling rate in Hz to the y-axis using a scale factor
	// Convert the fft size to sampleRate/2, the Nyquist critical frequency
	sf := 0.5 * sampleRate / endpoints.ymax

	// Construct y-axis labels
	incr = (endpoints.ymax - endpoints.ymin) / (ylabels - 1)
	y := endpoints.ymin
	// First label is empty for alignment purposes
	for i := range mlp.plot.Ylabel {
		mlp.plot.Ylabel[i] = fmt.Sprintf("%.0f", y*sf)
		y += incr
	}

	return nil
}

// newDisplayMLP creates a MLP instance for displaying speech in time or spectrogram
func newDisplayMLP(r *http.Request, plot *PlotT) (*MLP, error) {

	// Get from form percent voiced, pitch variation, duration variation
	txt := r.FormValue("delpitch")
	delPitch, err := strconv.Atoi(txt)
	if err != nil {
		fmt.Printf("delPitch int conversion error: %v\n", err)
		return nil, fmt.Errorf("delPitch conversion to int error: %v", err.Error())
	}

	txt = r.FormValue("delduration")
	delDuration, err := strconv.Atoi(txt)
	if err != nil {
		fmt.Printf("delDuration int conversion error: %v\n", err)
		return nil, fmt.Errorf("delDuration conversion to int error: %v", err.Error())
	}

	txt = r.FormValue("percentvoiced")
	percentVoiced, err := strconv.Atoi(txt)
	if err != nil {
		fmt.Printf("percentVoiced int conversion error: %v\n", err)
		return nil, fmt.Errorf("percentVoiced conversion to int error: %v", err.Error())
	}

	txt = r.FormValue("delampl")
	delAmpl, err := strconv.ParseFloat(txt, 64)
	if err != nil {
		fmt.Printf("delta Amplitude conversion error: %v\n", err)
		return nil, err
	}

	// Read in MLP parameters
	f, err := os.Open(path.Join(dataDir, fileweights))
	if err != nil {
		fmt.Printf("Open file %s error: %v", fileweights, err)
		return nil, fmt.Errorf("open file %s error: %s", fileweights, err.Error())
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	// get the parameters
	scanner.Scan()
	line := scanner.Text()

	items := strings.Split(line, ",")
	if len(items) != 11 {
		fmt.Printf("Display parameters missing, should be 11, is %d\n", len(items))
		return nil, fmt.Errorf("display parameters missing, should be 11, is %d", len(items))
	}

	fftSize, err := strconv.Atoi(items[5])
	if err != nil {
		fmt.Printf("Conversion to int of 'fftSize' error: %v\n", err)
		return nil, err
	}

	fftWindow := items[6]

	if err = scanner.Err(); err != nil {
		fmt.Printf("scanner error: %s\n", err.Error())
		return nil, fmt.Errorf("scanner error: %v", err)
	}

	mlp := MLP{
		plot:          plot,
		delPitch:      delPitch,
		delDuration:   delDuration,
		percentVoiced: percentVoiced,
		fftSize:       fftSize,
		fftWindow:     fftWindow,
		freqs:         make([]float64, maxSubFreq+1),
		amps:          make([]float64, maxSubFreq+1),
		delAmpl:       delAmpl,
	}

	// Determine if time or spectrogram domain plot
	mlp.domain = r.FormValue("domain")
	if mlp.domain == "spectrogram" {
		plot.Domain = "Spectrogram (Hz/sec)"
	} else {
		plot.Domain = "Time Domain (sec)"
	}

	// synthetic speech for creating speech with synthesize
	mlp.synSpeech = make([]float64, mlp.fftSize*nffts)

	return &mlp, nil
}

// handleDisplayMLP displays the selected speech pattern (time or spectrogram) and plays the audio
func handleDisplayMLP(w http.ResponseWriter, r *http.Request) {
	var (
		plot PlotT
		mlp  *MLP
	)

	// Get the speech pattern to display
	txt := r.FormValue("speechpattern")
	// Need speech pattern to continue
	if len(txt) > 0 {
		speechPattern, err := strconv.Atoi(txt)
		if err != nil {
			fmt.Printf("Speech Pattern int conversion error: %v\n", err)
			plot.Status = fmt.Sprintf("Speech Pattern conversion to int error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplTestingMLP.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// Construct MLP instance containing MLP state
		mlp, err = newDisplayMLP(r, &plot)
		if err != nil {
			fmt.Printf("newDisplayMLP() error: %v\n", err)
			plot.Status = fmt.Sprintf("newDisplayMLP() error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplDisplayMLP.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// make SpeechFrames for the speech patterns
		nsamples := mlp.fftSize * nffts
		// a frame consists of avgDuration samples = 25ms of speech sampled at 8,000 Hz
		nframes := int(math.Ceil(float64(nsamples) / float64(avgDuration)))

		// create new synthetic speech patterns
		newPattern := r.FormValue("newpattern")
		if len(newPattern) > 0 {
			if err = mlp.createPatterns(); err != nil {
				fmt.Printf("createPatterns() error: %v\n", err)
				plot.Status = fmt.Sprintf("createPatterns() error: %v", err.Error())
				// Write to HTTP using template and grid
				if err := tmplDisplayMLP.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			}
			// Read the synthetic speech patterns
		} else {
			files, err := os.ReadDir(dataDir)
			if err != nil {
				fmt.Printf("ReadDir %s error: %v\n", dataDir, err)
				plot.Status = fmt.Sprintf("ReadDir %s error: %v", dataDir, err.Error())
				// Write to HTTP using template and grid
				if err := tmplDisplayMLP.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			}
			if len(files) == 0 {
				fmt.Printf("No synthetic speech files in %s\n", dataDir)
				plot.Status = fmt.Sprintf("No synthetic speech files in %s, create new patterns", dataDir)
				// Write to HTTP using template and grid
				if err := tmplDisplayMLP.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			} else {
				mlp.speechPat = make([][]SpeechFrame, npatterns)
				for i := range mlp.speechPat {
					mlp.speechPat[i] = make([]SpeechFrame, nframes)
				}
				pattern := 0
				// Retrieve the speech files
				for _, dirEntry := range files {
					name := dirEntry.Name()
					if strings.Contains(name, "speech") && strings.Contains(name, "csv") {
						fspeech, err := os.Open(filepath.Join(dataDir, name))
						if err != nil {
							fmt.Printf("Open %s error: %v\n", name, err)
							plot.Status = fmt.Sprintf("Open %s error: %v", name, err.Error())
							// Write to HTTP using template and grid
							if err := tmplTrainingMLP.Execute(w, plot); err != nil {
								log.Fatalf("Write to HTTP output using template with error: %v\n", err)
							}
							return
						}
						scanner := bufio.NewScanner(fspeech)
						frame := 0
						for scanner.Scan() {
							line := scanner.Text()
							items := strings.Split(line, ",")
							nfreqs := len(items) / 2
							mlp.speechPat[pattern][frame].freqs = make([]float64, nfreqs)
							mlp.speechPat[pattern][frame].amps = make([]float64, nfreqs)
							for i := 0; i < nfreqs; i++ {
								freq, err := strconv.ParseFloat(items[i], 64)
								if err != nil {
									plot.Status = fmt.Sprintf("freq %d conversion error: %v", i, err.Error())
									// Write to HTTP using template and grid
									if err := tmplTrainingMLP.Execute(w, plot); err != nil {
										log.Fatalf("Write to HTTP output using template with error: %v\n", err)
									}
									return
								}
								ampl, err := strconv.ParseFloat(items[i+nfreqs], 64)
								if err != nil {
									plot.Status = fmt.Sprintf("ampl %d conversion error: %v", i, err.Error())
									// Write to HTTP using template and grid
									if err := tmplTrainingMLP.Execute(w, plot); err != nil {
										log.Fatalf("Write to HTTP output using template with error: %v\n", err)
									}
									return
								}
								mlp.speechPat[pattern][frame].amps[i] = ampl
								mlp.speechPat[pattern][frame].freqs[i] = freq
							}
							frame++
						}
						fspeech.Close()
						if err = scanner.Err(); err != nil {
							fmt.Printf("speech file scanner error: %s", err.Error())
							// Write to HTTP using template and grid
							if err := tmplTrainingMLP.Execute(w, plot); err != nil {
								log.Fatalf("Write to HTTP output using template with error: %v\n", err)
							}
							return
						}
						pattern++
					}
				}
			}
		}

		// generate speech, spectrogram if desired, create wav file,
		// using speech pattern, percent voiced, deltas duration, amplitude, and pitch

		err = mlp.createSpeech(speechPattern)
		for err != nil {
			if err.Error() == "repeat" {
				err = mlp.createSpeech(speechPattern)
			} else {
				fmt.Printf("createSpeech error: %v\n", err.Error())
				plot.Status = fmt.Sprintf("createSpeech error: %v", err.Error())
				// Write to HTTP using template and grid
				if err := tmplDisplayMLP.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			}
		}

		if mlp.domain == "spectrogram" {
			mlp.grayscale = make(map[int]string)
			for i := 0; i < ncolors; i++ {
				mlp.grayscale[i] = fmt.Sprintf("gs%d", i)
			}

			err := mlp.processSpectrogram(speechPattern, mlp.fftWindow, mlp.fftSize)
			if err != nil {
				fmt.Printf("proessSpectrogram error: %v\n", err)
				plot.Status = fmt.Sprintf("processSpectrogram error: %v", err.Error())
				// Write to HTTP using template and grid
				if err := tmplDisplayMLP.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			}
			plot.Status = fmt.Sprintf("Spectrogram of pattern %d plotted.", speechPattern)
		} else {
			err := mlp.processTimeDomain(speechPattern)
			if err != nil {
				fmt.Printf("processTimeDomain error: %v\n", err)
				plot.Status = fmt.Sprintf("processTimeDomain error: %v", err.Error())
				// Write to HTTP using template and grid
				if err := tmplDisplayMLP.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			}
			plot.Status = fmt.Sprintf("Time Domain of pattern %d plotted.", speechPattern)
		}

		// Create the wav file from the synthetic speech
		outF, err := os.Create(path.Join(dataDir, synSpeech))
		if err != nil {
			fmt.Printf("os.Create() file %s error: %v\n", synSpeech, err)
			plot.Status = fmt.Sprintf("os.Create() file %s error: %v", synSpeech, err.Error())
			// Write to HTTP using template and grid
			if err := tmplDisplayMLP.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}
		defer outF.Close()
		// create wav.Encoder
		enc := wav.NewEncoder(outF, sampleRate, bitDepth, 1, 1)

		// create audio.FloatBuffer
		float64Buf := &audio.FloatBuffer{Data: mlp.synSpeech, Format: &audio.Format{NumChannels: 1, SampleRate: sampleRate}}

		// create IntBuffer from FloatBuffer and pass to Encoder.Write()
		if err := enc.Write(float64Buf.AsIntBuffer()); err != nil {
			fmt.Printf("wav encoder write error: %v\n", err)
			plot.Status = fmt.Sprintf("wav encoder write error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplDisplayMLP.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// close the encoder
		if err := enc.Close(); err != nil {
			fmt.Printf("wav encoder close error: %v\n", err)
			plot.Status = fmt.Sprintf("wav encoder close error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplDisplayMLP.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// Play the audio wav if fmedia is available in the PATH environment variable
		fmedia, err := exec.LookPath("fmedia.exe")
		if err != nil {
			log.Fatal("fmedia is not available in PATH")
		} else {
			fmt.Printf("fmedia is available in path: %s\n", fmedia)
			cmd := exec.Command(fmedia, filepath.Join(dataDir, synSpeech))
			stdoutStderr, err := cmd.CombinedOutput()
			if err != nil {
				fmt.Printf("stdout, stderr error from running fmedia: %v\n", err)
			} else {
				fmt.Printf("fmedia output: %s\n", string(stdoutStderr))
			}
		}

		// set the speech parameters
		mlp.plot.DelDuration = strconv.Itoa(mlp.delDuration)
		mlp.plot.DelPitch = strconv.Itoa(mlp.delPitch)
		mlp.plot.DelAmpl = strconv.FormatFloat(mlp.delAmpl, 'f', -1, 64)
		mlp.plot.PercentVoiced = strconv.Itoa(mlp.percentVoiced)
		mlp.plot.SpeechPattern = strconv.Itoa(speechPattern)

		// Execute plot on display HTML template
		if err = tmplDisplayMLP.Execute(w, mlp.plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
	} else {
		plot.Status = "Enter Display speech parameters:  pattern, percent voiced, pitch delta, duration delta."
		// Write to HTTP using template and grid
		if err := tmplDisplayMLP.Execute(w, plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
		return
	}
}

// executive creates the HTTP handlers, listens and serves
func main() {
	// Set up HTTP servers with handlers for training and testing the Multilayer Perceptron Neural Network

	// Create HTTP handler for training
	http.HandleFunc(patternTrainingMLP, handleTrainingMLP)
	// Create HTTP handler for testing
	http.HandleFunc(patternTestingMLP, handleTestingMLP)
	// Create HTTP handler for spectrogram generation
	http.HandleFunc(patternDisplayMLP, handleDisplayMLP)
	fmt.Printf("Speech MLP Neural Network Server listening on %v.\n", addr)
	http.ListenAndServe(addr, nil)
}
