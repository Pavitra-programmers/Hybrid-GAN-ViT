import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import cv2
import random
import shutil

class DeepfakeDataset(Dataset):
    """
    Custom dataset for deepfake detection.
    Supports both real and fake images/videos with labels.
    """
    
    def __init__(self, data_dir, transform=None, img_size=224, extract_frames=True, frames_per_video=10):
        self.data_dir = data_dir
        self.transform = transform
        self.img_size = img_size
        self.extract_frames = extract_frames
        self.frames_per_video = frames_per_video
        
        # Get all image/video files and labels
        self.samples = []
        self._load_samples()
        
    def _load_samples(self):
        """Load all samples from the data directory."""
        # Check for video directories (SDFVD format)
        videos_real_dir = os.path.join(self.data_dir, 'videos_real')
        videos_fake_dir = os.path.join(self.data_dir, 'videos_fake')
        
        # Check for image directories (standard format or train/val split)
        real_dir = os.path.join(self.data_dir, 'real')
        fake_dir = os.path.join(self.data_dir, 'fake')
        
        frames_dir = os.path.join(self.data_dir, 'extracted_frames')
        
        # If videos exist in videos_real/videos_fake (original SDFVD format), extract frames
        if os.path.exists(videos_real_dir) and os.path.exists(videos_fake_dir):
            print(f"Found video directories. Extracting frames...")
            os.makedirs(frames_dir, exist_ok=True)
            
            real_frames_dir = os.path.join(frames_dir, 'real')
            fake_frames_dir = os.path.join(frames_dir, 'fake')
            os.makedirs(real_frames_dir, exist_ok=True)
            os.makedirs(fake_frames_dir, exist_ok=True)
            
            # Extract frames from real videos
            self._extract_video_frames(videos_real_dir, real_frames_dir, label=0)
            # Extract frames from fake videos
            self._extract_video_frames(videos_fake_dir, fake_frames_dir, label=1)
            
            # Load extracted frames
            if os.path.exists(real_frames_dir):
                for img_name in os.listdir(real_frames_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(real_frames_dir, img_name), 0))
            
            if os.path.exists(fake_frames_dir):
                for img_name in os.listdir(fake_frames_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(fake_frames_dir, img_name), 1))
        
        # If real/fake directories exist (train/val split), check for videos or images
        elif os.path.exists(real_dir) or os.path.exists(fake_dir):
            # Check if directories contain videos
            if os.path.exists(real_dir):
                real_videos = [f for f in os.listdir(real_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
                real_images = [f for f in os.listdir(real_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                if real_videos and not real_images:
                    # Extract frames from videos
                    os.makedirs(frames_dir, exist_ok=True)
                    real_frames_dir = os.path.join(frames_dir, 'real')
                    os.makedirs(real_frames_dir, exist_ok=True)
                    self._extract_video_frames(real_dir, real_frames_dir, label=0)
                    real_dir = real_frames_dir
                
                # Load images/frames
                for img_name in os.listdir(real_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(real_dir, img_name), 0))
            
            if os.path.exists(fake_dir):
                fake_videos = [f for f in os.listdir(fake_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
                fake_images = [f for f in os.listdir(fake_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                if fake_videos and not fake_images:
                    # Extract frames from videos
                    os.makedirs(frames_dir, exist_ok=True)
                    fake_frames_dir = os.path.join(frames_dir, 'fake')
                    os.makedirs(fake_frames_dir, exist_ok=True)
                    self._extract_video_frames(fake_dir, fake_frames_dir, label=1)
                    fake_dir = fake_frames_dir
                
                # Load images/frames
                for img_name in os.listdir(fake_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(fake_dir, img_name), 1))
        
        print(f"Loaded {len(self.samples)} samples (Real: {sum(1 for _, l in self.samples if l == 0)}, Fake: {sum(1 for _, l in self.samples if l == 1)})")
    
    def _extract_video_frames(self, video_dir, output_dir, label):
        """Extract frames from videos in a directory."""
        video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        for video_file in video_files:
            video_path = os.path.join(video_dir, video_file)
            self._extract_video_frames_single(video_path, output_dir, label)
    
    def _extract_video_frames_single(self, video_path, output_dir, label):
        """Extract frames from a single video."""
        if not os.path.exists(video_path):
            return
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Warning: Could not open video {video_path}")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        label_str = 'real' if label == 0 else 'fake'
        
        # Calculate frame interval to extract frames_per_video frames
        if total_frames > 0:
            frame_interval = max(1, total_frames // self.frames_per_video)
        else:
            frame_interval = 30  # Default: every 30 frames
        
        frame_count = 0
        saved_count = 0
        
        while saved_count < self.frames_per_video:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frame_filename = f"{label_str}_{video_name}_frame_{saved_count:05d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                saved_count += 1
            
            frame_count += 1
        
        cap.release()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load and preprocess image
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            else:
                # Default transform
                image = transforms.ToTensor()(image)
                image = transforms.Resize((self.img_size, self.img_size))(image)
                image = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                          std=[0.229, 0.224, 0.225])(image)
            
            return image, torch.tensor(label, dtype=torch.float32)
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image if loading fails
            dummy_image = torch.randn(3, self.img_size, self.img_size)
            return dummy_image, torch.tensor(label, dtype=torch.float32)

def get_transforms(img_size=224, is_training=True):
    """
    Get data transforms for training and validation.
    Enhanced with better augmentation for improved model performance.
    
    Args:
        img_size: Target image size
        is_training: Whether to apply training augmentations
        
    Returns:
        transform: Data transformation pipeline
    """
    if is_training:
        transform = transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),  # Reduced rotation
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Reduced jitter
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),  # Reduced transform
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.15),  # Reduced blur
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.05, scale=(0.02, 0.2))  # Reduced erasing
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform

def create_dataloaders(data_dir, batch_size=32, img_size=224, num_workers=4, train_split=0.8, frames_per_video=10):
    """
    Create training and validation dataloaders from SDFVD dataset.
    Automatically splits data into train/val sets.
    
    Args:
        data_dir: Directory containing the dataset (SDFVD format with videos_real and videos_fake)
        batch_size: Batch size for training
        img_size: Target image size
        num_workers: Number of worker processes
        train_split: Fraction of data to use for training (default: 0.8)
        frames_per_video: Number of frames to extract per video
        
    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader
    """
    # Check if train/val split already exists
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    # If train/val directories don't exist, create them and split the data
    if not os.path.exists(train_dir) or not os.path.exists(val_dir) or \
       (os.path.exists(train_dir) and len(os.listdir(train_dir)) == 0):
        print("Creating train/val split...")
        _create_train_val_split(data_dir, train_split)
    
    # Get transforms
    train_transform = get_transforms(img_size, is_training=True)
    val_transform = get_transforms(img_size, is_training=False)
    
    # Create datasets - pass frames_per_video to extract frames from videos
    train_dataset = DeepfakeDataset(
        train_dir, 
        transform=train_transform, 
        img_size=img_size, 
        extract_frames=True,
        frames_per_video=frames_per_video
    )
    val_dataset = DeepfakeDataset(
        val_dir, 
        transform=val_transform, 
        img_size=img_size, 
        extract_frames=True,
        frames_per_video=frames_per_video
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader

def _create_train_val_split(data_dir, train_split=0.8, seed=42):
    """Create train/val split from SDFVD dataset structure."""
    random.seed(seed)
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Create subdirectories
    for split_dir in [train_dir, val_dir]:
        os.makedirs(os.path.join(split_dir, 'real'), exist_ok=True)
        os.makedirs(os.path.join(split_dir, 'fake'), exist_ok=True)
    
    # Split videos_real
    videos_real_dir = os.path.join(data_dir, 'videos_real')
    if os.path.exists(videos_real_dir):
        real_videos = sorted([f for f in os.listdir(videos_real_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))])
        random.shuffle(real_videos)
        split_idx = int(len(real_videos) * train_split)
        
        for i, video in enumerate(real_videos):
            src = os.path.join(videos_real_dir, video)
            if i < split_idx:
                dst = os.path.join(train_dir, 'real', video)
            else:
                dst = os.path.join(val_dir, 'real', video)
            if not os.path.exists(dst):  # Only copy if doesn't exist
                shutil.copy2(src, dst)
    
    # Split videos_fake
    videos_fake_dir = os.path.join(data_dir, 'videos_fake')
    if os.path.exists(videos_fake_dir):
        fake_videos = sorted([f for f in os.listdir(videos_fake_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))])
        random.shuffle(fake_videos)
        split_idx = int(len(fake_videos) * train_split)
        
        for i, video in enumerate(fake_videos):
            src = os.path.join(videos_fake_dir, video)
            if i < split_idx:
                dst = os.path.join(train_dir, 'fake', video)
            else:
                dst = os.path.join(val_dir, 'fake', video)
            if not os.path.exists(dst):  # Only copy if doesn't exist
                shutil.copy2(src, dst)
    
    print(f"Created train/val split with {train_split*100:.0f}% training data")

def apply_texture_analysis(image_tensor):
    """
    Apply texture analysis using Gram matrices for detecting inconsistencies.
    
    Args:
        image_tensor: Input image tensor (B, C, H, W)
        
    Returns:
        texture_features: Texture analysis features
    """
    batch_size, channels, height, width = image_tensor.shape
    
    # Reshape for Gram matrix computation
    features = image_tensor.view(batch_size, channels, height * width)
    
    # Compute Gram matrix
    gram_matrix = torch.bmm(features, features.transpose(1, 2))
    
    # Normalize by spatial dimensions
    gram_matrix = gram_matrix / (height * width)
    
    return gram_matrix

def extract_patches(image_tensor, patch_size=16, stride=8):
    """
    Extract overlapping patches from images for detailed analysis.
    
    Args:
        image_tensor: Input image tensor (B, C, H, W)
        patch_size: Size of patches to extract
        stride: Stride for patch extraction
        
    Returns:
        patches: Extracted patches
        patch_positions: Positions of patches in original image
    """
    batch_size, channels, height, width = image_tensor.shape
    
    patches = []
    patch_positions = []
    
    for h in range(0, height - patch_size + 1, stride):
        for w in range(0, width - patch_size + 1, stride):
            patch = image_tensor[:, :, h:h+patch_size, w:w+patch_size]
            patches.append(patch)
            patch_positions.append((h, w))
    
    patches = torch.stack(patches, dim=1)  # (B, num_patches, C, patch_size, patch_size)
    
    return patches, patch_positions

def create_synthetic_data(num_samples=1000, img_size=224):
    """
    Create synthetic data for testing purposes.
    
    Args:
        num_samples: Number of synthetic samples to create
        img_size: Size of synthetic images
        
    Returns:
        images: Synthetic image tensor
        labels: Synthetic labels
    """
    # Create synthetic real images (random noise with some structure)
    real_images = torch.randn(num_samples // 2, 3, img_size, img_size)
    real_images = torch.clamp(real_images, 0, 1)
    
    # Create synthetic fake images (more structured, potentially with artifacts)
    fake_images = torch.randn(num_samples // 2, 3, img_size, img_size)
    # Add some artificial patterns that might indicate manipulation
    fake_images[:, :, :img_size//2, :img_size//2] += 0.3  # Add brightness variation
    fake_images = torch.clamp(fake_images, 0, 1)
    
    # Combine images and labels
    images = torch.cat([real_images, fake_images], dim=0)
    labels = torch.cat([
        torch.zeros(num_samples // 2),  # Real images
        torch.ones(num_samples // 2)    # Fake images
    ])
    
    # Shuffle
    indices = torch.randperm(num_samples)
    images = images[indices]
    labels = labels[indices]
    
    return images, labels 