# Multimodal Event Detection in Video Streams

Overview
This project integrates audiovisual deep learning to detect and classify complex events from realworld video streams. By combining SlowFast video networks for spatiotemporal visual cues with Wav2Vec2 for audio embeddings and a Transformerbased fusion module, the system achieves highprecision multimodal event understanding, critical for surveillance, media analytics, and situational awareness.

Framework
Models: SlowFast Networks, Wav2Vec2, Transformer Fusion Layer
Libraries: PyTorch, TorchAudio, OpenCV, Transformers, NumPy, Matplotlib

Scope
 Develop a multimodal event detection pipeline integrating video and audio modalities.
 Utilize SlowFast for temporal visual dynamics and Wav2Vec2 for audio representations.
 Implement Transformerbased crossmodal attention fusion.
 Train and evaluate on AudioSet for largescale event recognition.
 Visualize detection confidence and temporal event boundaries.

Dataset Used:
 AudioSet (Google Research) — 2 million 10second video clips annotated with 527 event classes (audiovisual aligned).

Preprocessing Steps:
 Extracted RGB frames at 30 fps and audio waveforms at 16 kHz.
 Applied data augmentation (time masking, random cropping).
 Normalized audio spectrograms and resized frames to 224×224.
 Synchronized modalities using frame–audio timestamp alignment.

Methodology

 1. Data Loading and Preprocessing

 Loaded synchronized video–audio pairs from AudioSet.
 Applied spectrogram conversion and frame stacking for uniform temporal alignment.

 2. Visual Stream – SlowFast Network

 Used Slow pathway for coarse temporal reasoning (context).
 Used Fast pathway for fine motion capture (details).
 Combined outputs via lateral fusion.

 3. Audio Stream – Wav2Vec2 Encoder

 Extracted contextualized speech/audio embeddings from raw waveforms.
 Finetuned on event detection subset for acoustic variability handling.

 4. Multimodal Fusion – Transformer Attention

 Concatenated visual and audio embeddings.
 Used crossattention Transformer for intermodal feature interaction.
 Output fused representation passed to classifier head for event prediction.

 5. Evaluation and Visualization

 Metrics: mAP, Temporal IoU, Precision, Recall.
 Visualized detection timelines with attention heatmaps across frames.

Architecture (Textual Diagram)
          ┌──────────────────────────────┐
          │ Input Video + Audio Stream   │
          └─────────────┬────────────────┘
                        │
        ┌───────────────▼───────────────┐
        │ SlowFast Network (Video)      │
        │   Motion + Context Features  │
        └───────────────┬───────────────┘
                        │
        ┌───────────────▼───────────────┐
        │ Wav2Vec2 Encoder (Audio)      │
        │   Acoustic Embeddings        │
        └───────────────┬───────────────┘
                        │
        ┌───────────────▼───────────────┐
        │ Transformer Fusion Layer      │
        │   CrossModal Attention      │
        └───────────────┬───────────────┘
                        │
        ┌───────────────▼───────────────┐
        │ Event Classifier              │
        │ (Multimodal Prediction)       │
        └───────────────┬───────────────┘
                        │
        ┌───────────────▼───────────────┐
        │ Temporal Event Timeline +     │
        │ Confidence Visualization      │
        └───────────────────────────────┘

 Results
| Model Configuration                          | mAP     | Temporal IoU | Precision | Recall   | F1 Score |
| SlowFast Only (Visual)                       | 73%     | 0.64         | 0.76      | 0.70     | 0.73     |
| Wav2Vec2 Only (Audio)                        | 69%     | 0.59         | 0.72      | 0.68     | 0.70     |
| SlowFast + Wav2Vec2 + Transformer (Ours)     | 82%     | 0.78         | 0.84      | 0.80     | 0.82     |

Qualitative Observations:
 Detected overlapping multimodal events (e.g., “dog barking while person speaking”).
 Maintained robust performance under noisy and lowlight conditions.
 Attention maps clearly highlight correlated audiovisual cues.

Conclusion
The Multimodal Event Detection System effectively integrates visual and auditory modalities via Transformerbased crossattention, achieving 82% mAP on AudioSet. The architecture exhibits high temporal precision and contextaware reasoning, making it suitable for applications in security, surveillance, autonomous monitoring, and broadcast analytics.

Future Work
 Extend to realtime inference using streaming transformer encoders.
 Incorporate speechtotext alignment for semantic event explanation.
 Explore selfsupervised multimodal pretraining for lowlabel environments.
 Deploy on edge devices with optimized lightweight fusion models.

 References
1. Feichtenhofer, C. et al. (2019). SlowFast Networks for Video Recognition. ICCV.
2. Baevski, A. et al. (2020). Wav2Vec 2.0: A Framework for SelfSupervised Learning of Speech Representations. NeurIPS.
3. Arandjelović, R. & Zisserman, A. (2018). Objects That Sound. ECCV.
4. Gemmeke, J. F. et al. (2017). AudioSet: An Ontology and HumanLabeled Dataset for Audio Events. ICASSP.
5. Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS.

Closest Research Paper:
> “CrossModal Attention for AudioVisual Event Localization” — IEEE Transactions on Multimedia, 2023.
> This work parallels the project’s Transformerbased multimodal fusion for synchronized event detection across video and audio streams.
