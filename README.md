#Robust Emotion Recognition Through Generative Adversarial Audio Enhancement and Multi-Branch Deep Learning Models

🎤 Robust-SER-GAN-CNN-BiLSTM

This repository contains the implementation of our research:

“Robust Emotion Recognition Through Generative Adversarial Audio Enhancement and Multi-Branch Deep Learning Models”

The project presents an end-to-end Speech Emotion Recognition (SER) pipeline that integrates:

🎧 GAN-based audio denoising (U-Net generator)
🎼 Multi-feature extraction (MFCC, Chroma, Spectral Contrast, Mel-Spectrogram)
🧠 Hybrid CNN–BiLSTM–Attention model
🔊 Optional Wav2Vec2 fine-tuning (raw audio)

The system is designed to improve robustness under noisy real-world conditions and achieves 98.86% accuracy on RAVDESS.

📂 Repository Structure
Robust-SER-GAN-CNN-BiLSTM/
│
├── preprocessing/        # Data loading & feature extraction
├── gan_denoiser/        # GAN model (U-Net Generator + Discriminator)
├── ser_model/           # CNN-BiLSTM-Attention classifier
├── wav2vec2_model/      # Optional transformer-based SER
├── training/            # Training scripts
├── evaluation/          # Metrics, confusion matrix, PR curves
├── visualization/       # Spectrograms, waveform plots
├── samples/             # Example noisy & denoised outputs (optional)
├── checkpoints/         # Saved models
└── README.md
⚙️ Requirements
Python ≥ 3.8
PyTorch
NumPy
Librosa
Scikit-learn
Matplotlib
Pandas
Torchaudio
Transformers (for Wav2Vec2)

Install dependencies:

pip install -r requirements.txt
📥 Dataset

We use the RAVDESS dataset:

24 actors
8 emotions:
neutral, calm, happy, sad, angry, fearful, disgust, surprised

Download from:

👉 https://zenodo.org/record/1188976

🧹 Data Preprocessing

Run preprocessing to extract features:

python preprocessing/run_preprocessing.py --data_dir <RAVDESS_PATH>
Output Structure:
processed_data/
├── audio/
├── mfcc/
├── chroma/
├── mel/
├── contrast/
├── labels/
Preprocessing includes:
Resampling & normalization
MFCC extraction (40 coefficients)
Feature fusion (MFCC + Chroma + Spectral Contrast + Mel)
Label encoding
Train/validation/test split
🤖 GAN-Based Audio Denoising

Train the GAN model:

python gan_denoiser/train_gan.py
Architecture:
Generator: U-Net (encoder–decoder with skip connections)
Discriminator: CNN-based binary classifier
Loss Function:
Reconstruction loss (MSE)
Adversarial loss (BCE)

After training, denoise audio:

python gan_denoiser/denoise.py --input_dir <noisy_audio> --output_dir <clean_audio>
🧠 SER Model (CNN–BiLSTM–Attention)

Train the emotion recognition model:

python training/train_ser.py
Architecture:
Conv1D layers → feature extraction
BiLSTM → temporal modeling
Multi-head attention → focus on emotional cues
Dense + Softmax → classification
🔄 Full Pipeline Training

To run the full pipeline:

bash training/full_pipeline.sh
Pipeline Steps:
Preprocess dataset
Train GAN denoiser
Denoise audio samples
Extract features
Train CNN–BiLSTM model
Evaluate results
📊 Evaluation

Run evaluation:

python evaluation/evaluate.py
Metrics:
Accuracy
Precision / Recall / F1-score
Confusion Matrix
Precision-Recall Curves
📈 Visualization

Generate plots:

python visualization/plot_results.py

Includes:

Accuracy vs noise level
Confusion matrix
Spectrogram comparison (noisy vs denoised)
Emotion probability distributions
🔊 Inference

Predict emotion from audio:

python inference/predict.py --audio <file.wav>

Output:

Predicted emotion
Confidence scores
Spectrogram visualization
🤖 Wav2Vec2 (Optional)

Fine-tune transformer-based model:

python wav2vec2_model/train_wav2vec2.py

This model:

Uses raw waveform input
Learns contextual representations
Improves performance in low-feature scenarios
🧪 Results
Model	Accuracy
CNN only	~92%
CNN + BiLSTM	~96%
CNN + BiLSTM + Attention	~98%
GAN + CNN-BiLSTM (Proposed)	98.86%
Key Findings:
GAN denoising significantly improves performance under noise
Feature fusion boosts robustness
Attention enhances emotional feature learning
🚀 Future Work
Cross-dataset generalization (IEMOCAP, TESS, SAVEE)
Real-time deployment
Multimodal emotion recognition (audio + video)
StarGAN-based emotional voice conversion
📜 Citation

If you use this work, please cite:

@article{bakri2025robust_ser,
  title={Robust Emotion Recognition Through Generative Adversarial Audio Enhancement and Multi-Branch Deep Learning Models},
  author={Bakri, Ahmed and Viriri, Serestina},
  journal={MDPI},
  year={2025}
}
📧 Contact

Bakri Ahmed
University of KwaZulu-Natal
📩 225083932@stu.ukzn.ac.za
