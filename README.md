# Motor Imagery Classification

## Overview
The goal of this project is to correctly classify two-class motor imagery
from [Dataset 1](https://www.bbci.de/competition/iv/desc_1.html) of the BCI Competition IV.


## Protocol Summary (Per Run)
| Subject | Run         | Classes               | Trials / class | Fs (Hz) | Epoch Window (s) | Crop (s) | Band (Hz) | Ref | Channels Used | Artifacts |
|---------|-------------|-----------------------|----------------|---------|------------------|----------|-----------|-----|---------------|-----------|
| A       | Calibration | Left Hand, Foot       | 100            | 1000    | -1.5-4.5         | 0.5-3.5  | 8-30      |     |               |           |
| B       | Calibration | Left Hand, Right Hand | 100            | 1000    | -1.5-4.5         | 0.5-3.5  | 8-30      |     |               |           |
| F       | Calibration | Left Hand, Foot       | 100            | 1000    | -1.5-4.5         | 0.5-3.5  | 8-30      |     |               |           |
| G       | Calibration | Left Hand, Right Hand | 100            | 1000    | -1.5-4.5         | 0.5-3.5  | 8-30      |     |               |           |

## Modeling & Performance 

| Subject | Run         | Features     | CV scheme                      | Groups unit | CSP comps | Classifier | Metric       | Mean ± SD   | Class-wise acc | Permutation BA (mean ± SD) | p-value | Leakage guards |
|---------|-------------|--------------|--------------------------------|-------------|-----------|------------|--------------|-------------|----------------|----------------------------|---------|----------------|
| A       | Calibration | CSP(log-var) | Stratified Shuffle Split (SSS) | trial       | 4         | LDA        | Balanced Acc | 0.69 ± 0.13 | L: 78% F: 59%  |                            |         |                |
| B       | Calibration | CSP(log-var) | Stratified Shuffle Split (SSS) | trial       | 4         | LDA        | Balanced Acc | 0.80 ± 0.06 | L: 75% R: 85%  |                            |         |                |
| F       | Calibration | CSP(log-var) | Stratified Shuffle Split (SSS) | trial       | 4         | LDA        | Balanced Acc | 0.91 ± 0.04 | L: 87% F: 94%  |                            |         |                |
| G       | Calibration | CSP(log-var) | Stratified Shuffle Split (SSS) | trial       | 4         | LDA        | Balanced Acc | 0.88 ± 0.08 | L: 88% R: 88%  |                            |         |                |
