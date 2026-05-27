# 🎵 Surrogate Language Aligner

This project is a semi-automatic Python tool for segmenting and labeling paired spoken and drummed musical surrogate recordings. The program takes an audio recording and a list of elicitation sentences, detects relevant audio segments, labels them by their order in the experiment, and exports results for analysis. This project serves as my [Dartmouth COSC 72](https://dartmouth.smartcatalogiq.com/en/current/orc/departments-programs-undergraduate/computer-science/cosc-computer-science-undergraduate/cosc-72) final project.

## 📌 Project Overview

Musical surrogate languages use instruments (drums, flutes, etc.) to represent aspects of spoken language. This project is designed to make analysis of spoken/drummed data faster by helping identify where each spoken phrase and corresponding surrogate phrase occurs in a recording.

The current version focuses on:

- Loading an audio recording
- Detecting spoken and drummed segments
- Labeling segments using the order of experiment sentences
- Exporting segment information to Praat-readable TextGrid-style annotations
- Producing basic plots for checking the output

## 📁 Folder Structure

```text
surrogate-music-tone-aligner/
├── data/
│   ├── sample.wav
│   └── sentences.csv
├── outputs/
│   ├── segments.csv
│   └── annotation.TextGrid
├── screenshots/
├── .gitignore
├── aligner.py
├── LICENSE
├── README.md
└── requirements.txt
```
