import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os

# Import our models
from models.sequential_model import SequentialGANViT
from models.parallel_model import ParallelGANViT
from utils.data_utils import create_synthetic_data, get_transforms

class DeepfakeDetector:
    """
    Deepfake detection demo class for both sequential and parallel models.
    """
    
    def __init__(self, model_path=None, model_type='sequential', device='auto'):
        """
        Initialize the deepfake detector.
        
        Args:
            model_path: Path to trained model checkpoint
            model_type: 'sequential' or 'parallel'
            device: Device to run inference on
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.model_type = model_type
        
        # Initialize model
        if model_type == 'sequential':
            self.model = SequentialGANViT(
                img_size=224,
                gan_feature_dim=512,
                vit_embed_dim=768,
                num_heads=8,
                num_layers=6
            )
        else:  # parallel
            self.model = ParallelGANViT(
                img_size=224,
                gan_feature_dim=512,
                vit_embed_dim=768,
                num_heads=8,
                num_layers=6
            )
        
        # Load trained weights if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            print(f"Loaded {model_type} model from {model_path}")
        else:
            print(f"Initialized {model_type} model with random weights")
        
        self.model.to(self.device)
        self.model.eval()
    
    def load_model(self, model_path):
        """Load trained model weights."""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from epoch {checkpoint['epoch']}")
        print(f"Validation accuracy: {checkpoint['val_acc']:.4f}")
    
    def preprocess_image(self, image_path, img_size=224):
        """
        Preprocess image for model input.
        
        Args:
            image_path: Path to image file
            img_size: Target image size
            
        Returns:
            preprocessed_image: Preprocessed image tensor
        """
        # Load image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        
        # Apply transforms
        transform = get_transforms(img_size, is_training=False)
        preprocessed_image = transform(image)
        
        # Add batch dimension
        preprocessed_image = preprocessed_image.unsqueeze(0)
        
        return preprocessed_image
    
    def detect_deepfake(self, image_path, threshold=0.5):
        """
        Detect deepfake in an image.
        
        Args:
            image_path: Path to image file
            threshold: Classification threshold
            
        Returns:
            result: Detection result dictionary
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model(image_tensor)
            
            # Get prediction
            fake_probability = predictions['output'].item()
            is_fake = fake_probability > threshold
            
            # Prepare result
            result = {
                'fake_probability': fake_probability,
                'is_fake': is_fake,
                'confidence': predictions.get('confidence', None),
                'gan_features': predictions.get('gan_features', None),
                'vit_features': predictions.get('vit_features', None),
                'attention_map': predictions.get('attention_map', None)
            }
            
            # Add model-specific results
            if self.model_type == 'sequential':
                result['gan_classification'] = predictions.get('gan_classification', None)
                result['vit_classification'] = predictions.get('vit_classification', None)
            else:  # parallel
                result['enhanced_gan_features'] = predictions.get('enhanced_gan_features', None)
                result['enhanced_vit_features'] = predictions.get('enhanced_vit_features', None)
                result['cross_attention'] = predictions.get('attention_weights', None)
        
        return result
    
    def get_interpretability_maps(self, image_path):
        """
        Get interpretability maps for model explanation.
        
        Args:
            image_path: Path to image file
            
        Returns:
            maps: Dictionary of interpretability maps
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)
        
        # Get interpretability maps
        with torch.no_grad():
            maps = self.model.get_interpretability_maps(image_tensor)
        
        return maps
    
    def visualize_results(self, image_path, result, save_path=None):
        """
        Visualize detection results and attention maps.
        
        Args:
            image_path: Path to input image
            result: Detection result from detect_deepfake
            save_path: Path to save visualization
        """
        # Load original image
        if isinstance(image_path, str):
            original_image = Image.open(image_path)
        else:
            original_image = image_path
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Deepfake Detection Results - {self.model_type.upper()} Model', fontsize=16)
        
        # Original image
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Detection result
        fake_prob = result['fake_probability']
        is_fake = result['is_fake']
        color = 'red' if is_fake else 'green'
        status = 'FAKE' if is_fake else 'REAL'
        
        axes[0, 1].text(0.5, 0.5, f'{status}\nProbability: {fake_prob:.4f}', 
                        ha='center', va='center', fontsize=20, color=color,
                        transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Detection Result')
        axes[0, 1].axis('off')
        
        # Confidence (if available)
        if result['confidence'] is not None:
            conf = result['confidence'].item()
            axes[0, 2].text(0.5, 0.5, f'Confidence: {conf:.4f}', 
                            ha='center', va='center', fontsize=16,
                            transform=axes[0, 2].transAxes)
            axes[0, 2].set_title('Model Confidence')
        else:
            axes[0, 2].text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=16,
                            transform=axes[0, 2].transAxes)
            axes[0, 2].set_title('Model Confidence')
        axes[0, 2].axis('off')
        
        # GAN attention map
        if result['attention_map'] is not None:
            gan_attn = result['attention_map'].cpu().numpy()[0, 0]  # First batch, first channel
            axes[1, 0].imshow(gan_attn, cmap='hot')
            axes[1, 0].set_title('GAN Attention Map')
            axes[1, 0].axis('off')
        else:
            axes[1, 0].text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=16,
                            transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('GAN Attention Map')
            axes[1, 0].axis('off')
        
        # Feature importance
        if result['gan_features'] is not None:
            gan_feat = result['gan_features'].cpu().numpy()[0]
            axes[1, 1].plot(gan_feat[:50])  # Plot first 50 features
            axes[1, 1].set_title('GAN Features (First 50)')
            axes[1, 1].set_xlabel('Feature Index')
            axes[1, 1].set_ylabel('Feature Value')
        else:
            axes[1, 1].text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=16,
                            transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('GAN Features')
        
        # ViT features
        if result['vit_features'] is not None:
            vit_feat = result['vit_features'].cpu().numpy()[0]
            axes[1, 2].plot(vit_feat[:50])  # Plot first 50 features
            axes[1, 2].set_title('ViT Features (First 50)')
            axes[1, 2].set_xlabel('Feature Index')
            axes[1, 2].set_ylabel('Feature Value')
        else:
            axes[1, 2].text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=16,
                            transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('ViT Features')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def batch_detection(self, image_paths, threshold=0.5):
        """
        Perform batch detection on multiple images.
        
        Args:
            image_paths: List of image paths
            threshold: Classification threshold
            
        Returns:
            results: List of detection results
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            print(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            try:
                result = self.detect_deepfake(image_path, threshold)
                results.append({
                    'image_path': image_path,
                    'result': result
                })
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results

def main():
    """Main demo function."""
    print("Deepfake Detection Demo")
    print("=" * 50)
    
    # Create synthetic data for demonstration
    print("Creating synthetic test data...")
    test_images, test_labels = create_synthetic_data(num_samples=10, img_size=224)
    
    # Test Sequential Model
    print("\nTesting Sequential GAN+ViT Model...")
    sequential_detector = DeepfakeDetector(model_type='sequential')
    
    # Test on synthetic data
    for i in range(min(3, len(test_images))):
        print(f"\nTest image {i+1}:")
        result = sequential_detector.detect_deepfake(test_images[i])
        print(f"Fake probability: {result['fake_probability']:.4f}")
        print(f"Predicted: {'FAKE' if result['is_fake'] else 'REAL'}")
        print(f"Actual: {'FAKE' if test_labels[i] == 1 else 'REAL'}")
    
    # Test Parallel Model
    print("\nTesting Parallel GAN+ViT Model...")
    parallel_detector = DeepfakeDetector(model_type='parallel')
    
    # Test on synthetic data
    for i in range(min(3, len(test_images))):
        print(f"\nTest image {i+1}:")
        result = parallel_detector.detect_deepfake(test_images[i])
        print(f"Fake probability: {result['fake_probability']:.4f}")
        print(f"Predicted: {'FAKE' if result['is_fake'] else 'REAL'}")
        print(f"Actual: {'FAKE' if test_labels[i] == 1 else 'REAL'}")
    
    # Demonstrate interpretability
    print("\nDemonstrating interpretability...")
    test_image = test_images[0]
    
    # Sequential model interpretability
    sequential_maps = sequential_detector.get_interpretability_maps(test_image)
    print("Sequential model interpretability maps generated")
    
    # Parallel model interpretability
    parallel_maps = parallel_detector.get_interpretability_maps(test_image)
    print("Parallel model interpretability maps generated")
    
    # Visualize results
    print("\nGenerating visualizations...")
    sequential_result = sequential_detector.detect_deepfake(test_image)
    sequential_detector.visualize_results(test_image, sequential_result, 'sequential_results.png')
    
    parallel_result = parallel_detector.detect_deepfake(test_image)
    parallel_detector.visualize_results(test_image, parallel_result, 'parallel_results.png')
    
    print("\nDemo completed!")
    print("Check the current directory for result visualizations.")

if __name__ == "__main__":
    main() 