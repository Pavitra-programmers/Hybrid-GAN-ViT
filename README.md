# Deepfake Detection with Hybrid GAN+ViT Models

This project implements a hybrid deepfake detection system that combines the strengths of Generative Adversarial Networks (GANs) and Vision Transformers (ViTs) to create robust and interpretable deepfake detection models.

## 🎯 Project Overview

The project addresses the critical challenge of detecting increasingly sophisticated deepfake content by implementing two innovative hybrid architectures:

1. **Sequential GAN+ViT Model**: A two-stage approach where GAN discriminator analyzes fine details first, then ViT processes the guided features for broader understanding.

2. **Parallel GAN+ViT Model**: A simultaneous processing approach where both models work in parallel with cross-attention mechanisms for feature fusion.

## 🏗️ Architecture Details

### Sequential Model Architecture
```
Input Image → GAN Discriminator → Attention Guidance → ViT → Feature Fusion → Classification
```

**Key Features:**
- **GAN Discriminator**: Detects fine details, low-level clues, and manipulation artifacts
- **Attention Guidance**: Uses GAN attention to focus ViT on suspicious regions
- **Vision Transformer**: Provides big-picture understanding and global context
- **Feature Fusion**: Combines both feature types for final decision

### Parallel Model Architecture
```
Input Image → [GAN Discriminator] → Cross-Attention → Feature Fusion → Classification
              [Vision Transformer] ↗
```

**Key Features:**
- **Parallel Processing**: Both models analyze the image simultaneously
- **Cross-Attention**: Bidirectional feature interaction between GAN and ViT
- **Enhanced Features**: Mutual enhancement through attention mechanisms
- **Confidence Estimation**: Built-in confidence scoring for predictions

## 🚀 Features

- **Smart Focus**: GAN discriminators guide attention to suspicious regions
- **Fine Detail Detection**: Texture analysis using Gram matrices for inconsistencies
- **Feature Fusion**: Intelligent combination of local and global features
- **Interpretability**: Attention maps and feature importance visualization
- **Cross-Platform**: Tested on multiple deepfake datasets
- **Efficient Processing**: Optimized for both speed and accuracy

## 📁 Project Structure

```
Research Paper/
├── models/
│   ├── __init__.py
│   ├── gan_discriminator.py      # GAN discriminator implementation
│   ├── vision_transformer.py     # Vision Transformer implementation
│   ├── sequential_model.py       # Sequential GAN+ViT model
│   └── parallel_model.py         # Parallel GAN+ViT model
├── utils/
│   ├── __init__.py
│   └── data_utils.py             # Data loading and preprocessing utilities
├── train.py                      # Training script for both models
├── demo.py                       # Demo and inference script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd Research\ Paper
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify installation:**
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

## 🎮 Usage

### Training the Models

1. **Train both models with synthetic data:**
```bash
python train.py
```

This will:
- Train the Sequential GAN+ViT model
- Train the Parallel GAN+ViT model
- Save checkpoints in `checkpoints/` directory
- Generate training history plots

### Running the Demo

1. **Test the models with synthetic data:**
```bash
python demo.py
```

This will:
- Initialize both models
- Run inference on synthetic test images
- Generate interpretability visualizations
- Save result plots

### Using Your Own Data

1. **Prepare your dataset:**
```
data/
├── train/
│   ├── real/     # Real images
│   └── fake/     # Fake/deepfake images
└── val/
    ├── real/     # Real images
    └── fake/     # Fake/deepfake images
```

2. **Modify the training script** to use your data directory:
```python
# In train.py, replace create_synthetic_data with:
train_loader, val_loader = create_dataloaders('path/to/your/data')
```

## 🔧 Model Configuration

### Sequential Model Parameters
```python
sequential_model = SequentialGANViT(
    img_size=224,              # Input image size
    gan_feature_dim=512,       # GAN feature dimension
    vit_embed_dim=768,         # ViT embedding dimension
    num_heads=8,               # Number of attention heads
    num_layers=6               # Number of transformer layers
)
```

### Parallel Model Parameters
```python
parallel_model = ParallelGANViT(
    img_size=224,              # Input image size
    gan_feature_dim=512,       # GAN feature dimension
    vit_embed_dim=768,         # ViT embedding dimension
    num_heads=8,               # Number of attention heads
    num_layers=6               # Number of transformer layers
)
```

## 📊 Model Performance

The models are designed to achieve:

- **High Accuracy**: Combines local and global feature analysis
- **Robust Generalization**: Works across different manipulation techniques
- **Fast Inference**: Optimized architecture for real-time detection
- **Interpretability**: Attention maps and feature importance visualization

## 🔍 Interpretability Features

### Attention Maps
- **GAN Attention**: Highlights suspicious regions and artifacts
- **ViT Attention**: Shows global context understanding
- **Cross-Attention**: Demonstrates feature interaction between models

### Feature Analysis
- **Texture Analysis**: Gram matrices for detecting inconsistencies
- **Feature Importance**: Visualization of learned representations
- **Confidence Scoring**: Model uncertainty estimation

## 🎯 Use Cases

- **Social Media**: Detect manipulated images and videos
- **News Verification**: Identify fake news imagery
- **Forensic Analysis**: Digital evidence authentication
- **Content Moderation**: Automated deepfake detection
- **Research**: Academic deepfake detection studies

## 🚧 Limitations and Future Work

### Current Limitations
- Requires significant computational resources for training
- Performance depends on training data quality
- May struggle with very high-quality deepfakes

### Future Improvements
- **Multi-modal Fusion**: Incorporate audio and video analysis
- **Adversarial Training**: Improve robustness against attacks
- **Real-time Processing**: Optimize for live video streams
- **Transfer Learning**: Adapt to new deepfake techniques

## 📚 Technical Details

### Loss Functions
- **Combined Loss**: Weighted combination of GAN, ViT, and final outputs
- **Attention Loss**: Encourages meaningful attention patterns
- **Confidence Loss**: Improves prediction reliability

### Training Strategy
- **Data Augmentation**: Rotation, flipping, color jittering
- **Learning Rate Scheduling**: Adaptive learning rate adjustment
- **Early Stopping**: Prevents overfitting
- **Model Checkpointing**: Saves best performing models

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for research and educational purposes. Please ensure compliance with local laws and regulations when using this technology.

## 🙏 Acknowledgments

- Research community for deepfake detection advancements
- PyTorch and TensorFlow communities
- Vision Transformer and GAN research papers

## 📞 Contact

For questions or collaboration opportunities, please open an issue in the repository.

---

**Note**: This implementation is for research and educational purposes. Always verify results and use responsibly in real-world applications. 