# Motor Imagery Simulator

## Overview
The goal of this project is to simulate a motor-imagery BCI by streaming the BCI Competition III dataset IVa. 


## Pipeline

### Data Loading
The data is loaded from .mat files. Each file contains the continuous EEG signal, trial cues, class labels, sampling frequency, and class names.
### Stream Simulation
### Epoch Extraction
For training, features will be picked from a 1-second window start 2.5 seconds after the cue of each trial. 
### Feature Extraction
Discrete Fourier transform (DFT) features vs. Common Spatial Features (CSP) for classification.
### Classification
Random forest (RF) vs. shrinkage regularized linear discriminant analysis (sLDA). 
### Performance metrics
### Visualization 