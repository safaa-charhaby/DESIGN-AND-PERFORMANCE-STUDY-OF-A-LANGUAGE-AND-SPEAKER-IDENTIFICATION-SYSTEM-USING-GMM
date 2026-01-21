Here is the complete, raw Markdown code. You can copy everything inside the box below and paste it directly into your README.md file.

code
Markdown
download
content_copy
expand_less
<div align="center">

# 🎙️ GMM-Based Language & Speaker Identification System
**Design, Implementation, and Performance Evaluation**

[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-GMM--EM-orange?style=for-the-badge)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Project-Academic-success?style=for-the-badge)](https://github.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](https://opensource.org/licenses/MIT)

**📅 Academic Year: 2025–2026**  
**👨‍🏫 Supervisor: Prof. Jamal Kharroubi**

---

[**Overview**](#-overview) • [**Architecture**](#-system-architecture) • [**Performance**](#-performance-evaluation) • [**Setup**](#-installation--setup) • [**Structure**](#-project-structure)

</div>

---

## 📌 Overview

This project presents a robust system for **Language Identification (LID)** and **Speaker Identification/Verification (SID)**. By utilizing **Gaussian Mixture Models (GMM)** and **MFCC** acoustic features, the system achieves high-accuracy recognition and provides a full pipeline from audio signal to translated speech.

### 🎯 Core Capabilities
*   🌍 **Language ID:** Detects 5 languages (French, English, Dutch, Darija, Japanese).
*   🧑 **Speaker ID:** Identifies and verifies specific individual voices.
*   📝 **Transcription:** Converts speech to text in real-time.
*   🌐 **Translation:** Translates recognized text to target languages.
*   🔊 **Synthesis:** Generates synthesized speech (TTS) for the output.

---

## 🧠 System Architecture

The system implements a statistical pattern recognition pipeline:

1.  **Preprocessing:** Hybrid Silence Removal (K-Means + GMM + Energy Thresholding).
2.  **Feature Extraction:** MFCC (Mel-Frequency Cepstral Coefficients).
3.  **Modeling:** Statistical modeling using GMM with Expectation-Maximization (EM).
4.  **Selection:** Model optimization via **Bayesian Information Criterion (BIC)**.

```mermaid
graph TD
    A[Audio Input] --> B[Hybrid Silence Removal]
    B --> C[MFCC Extraction]
    C --> D[GMM/EM Training]
    D --> E{Decision Engine}
    E --> F[Language Classification]
    E --> G[Speaker Verification]
📊 Performance Evaluation

We conducted rigorous testing across different Gaussian components and test segment lengths.

🧪 Key Findings

Best Model: GMM with 256 Gaussians achieved the highest resolution.

Verification: Achieved an Equal Error Rate (EER) of 5.4%.

Reliability: 94.6% accuracy for speaker verification with 10s test segments.

📈 Comparison Table
Parameter	Optimal Value	Impact
Gaussian Components	32 (LID) / 256 (SID)	Balances precision vs. speed
Training Duration	60 - 120 Seconds	Essential for model convergence
Test Segment	10 Seconds	Minimum for stable log-likelihood
🛠️ Installation & Setup
1. Clone & Environment
code
Bash
download
content_copy
expand_less
git clone https://github.com/your-username/GMM-Speech-ID.git
cd GMM-Speech-ID
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
2. Dependencies
code
Bash
download
content_copy
expand_less
pip install numpy librosa scikit-learn speech_recognition pyttsx3 gTTS matplotlib pandas
3. Run the Application
code
Bash
download
content_copy
expand_less
python App.py
📁 Project Structure
code
Text
download
content_copy
expand_less
.
├── All_Gaussians/              # Trained models across different scales
├── trained_models/             # Production-ready BIC-selected models
├── data/                       # Dataset (Train/Test)
├── notebooks/                  # Analysis and Plotting scripts
├── src/
│   ├── silence_removal.py      # Hybrid thresholding logic
│   ├── features.py             # MFCC extraction scripts
│   └── classification.py       # GMM inference logic
├── App.py                      # GUI Application (Tkinter/PyQt)
└── README.md
⚙️ Technical Highlights
🔇 Hybrid Silence Removal

Unlike standard thresholding, our system uses a Hybrid Method combining K-Means and Energy analysis. This ensures that:

Speech integrity is preserved.

Background noise is effectively suppressed.

Word truncation is minimized.

🧮 Model Selection (BIC)

We don't just pick a random number of Gaussians. The system calculates the Bayesian Information Criterion (BIC) for multiple models and automatically selects the one that minimizes information loss while avoiding overfitting.

✅ Conclusion

This project demonstrates that GMM-MFCC architectures remains highly effective for speech tasks. The system is efficient, scalable, and accurate, providing a solid foundation for real-world biometric and linguistic applications.

<div align="center">


Developed for the 2025–2026 Academic Term.
If you find this research helpful, please consider giving it a ⭐!

</div>
```
