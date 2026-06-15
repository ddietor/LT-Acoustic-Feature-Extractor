# LT-Acoustic-Feature-Extractor

This project provides scripts to test Long-Term Acoustic Feature Extraction methods described in:

Diego-Tortosa, D.; Bonanno, D.; Bou-Cabo, M.; Di Mauro, L. S.; Idrissi, A.; Lara, G.; Riccobene, G.; Sanfilippo, S., & Viola, S. (2025). *Effective Strategies for Automatic Analysis of Acoustic Signals in Long-Term Monitoring*. Journal of Marine Science and Engineering, 13(3), 454. https://doi.org/10.3390/jmse13030454 

If you use this repository in your research, a citation to this article would be appreciated.


## Main notebook

The complete workflow is implemented in `LT_Acoustic_Feature_Extractor.ipyn`. 
This notebook serves as the main entry point of the project and executes all functionalities in a reproducible step-by-step manner.


## Workflow overview (Section 1 and Section 2)

The processing pipeline implemented in this repository follows the workflow shown below:

[![Proposed space-saving workflow for long-term acoustic monitoring data](https://www.mdpi.com/jmse/jmse-13-00454/article_deploy/html/images/jmse-13-00454-g002.png)](https://www.mdpi.com/2077-1312/13/3/454)






---


## Pipeline description

### Environment setup and synthetic signal generation

The notebook first imports external dependencies and internal modules from `src/`.

Then, a synthetic acoustic signal is generated using the script `scripts/generate_example_data.py` and saved as a WAV in the `outputs` directory.
 

This signal emulates an experimental 5-min hydrophone recording with low-frequency spike (occurring below the 31 Hz third-octave band and lasting around 15 s in duration) by SMO-O𝜈DE-2 station.

---

### - Section 1 and 2: High and Low frequency analysis

Starting from the synthetic WAV signal, the notebook reproduces the full feature extraction workflow step by step:

- PSD computation (`PSDdata`, `PSDdata_cum`)
- SPL computation (`SPLdata`, `SPLdata_cum`)

This follows the methodological pipeline described in the paper.

---

### - Section 3: Full frequency analysis

This section combines results from the previous analyses to generate:

- Full-band PSD representation (spectrogram-like visualization)
- Full-band SPL representation
- Combined `PSDdata_cum` visualization (equivalent utput to **[Figure 11](https://www.mdpi.com/jmse/jmse-13-00454/article_deploy/html/images/jmse-13-00454-g011.png)** of the paper)

---

### 4. - Section 4: Post-Processing. The Event Detection Phase from low-frequency SPL

The final section demonstrates an automatic event detection procedure based on low-frequency SPL data (`low_SPLdata`). This reproduces the concept shown in **[Figure 10](https://www.mdpi.com/jmse/jmse-13-00454/article_deploy/html/images/jmse-13-00454-g010.png)** of the paper.


## Repository structure

The project is organized as follows:

```
LT_Acoustic_Feature_Extractor
│
├── LICENSE
├── README.md
├── LT_Acoustic_Feature_Extractor.ipynb # Main example notebook (runs the full pipeline)
├── .gitignore
├── requirements.txt
│
├── scripts/
│ ├── init.py
│ ├── earthqueake_envelope.txt
│ └── generate_example_data.py
│
├── src/
│ ├── init.py
│ ├── spectrogram_analysis.py
│ ├── spl_utils.py
│ ├── wav_io.py
│ └── utils.py
│
…   # Default directory for the pipelina outputs
│
└── results/

```

---

## To do list:
```
- WAV creation for the decimated signal example (as suggested in the article)

- Improve the structure of LT_Acoustic_Feature_Extractor.ipynb to compute PSD and SPL (and derived metrics) for both high- and low-frequency analyses, and then display the figures side by side. This will reduce the vertical space required in the notebook output preview.
```