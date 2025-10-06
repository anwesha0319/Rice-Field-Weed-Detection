# Rice Field Weed Detection Project by Anwesha Chatterjee

A machine learning project for detecting weeds in rice fields using YOLOv8 object detection.

## 🌾 Project Overview

This project uses computer vision and deep learning to automatically detect and classify weeds in rice field images. The model can identify:
- Rice 
- weeds

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone/download the project
cd 22BCE10918_ANWESHA_RiceWeedDetection

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

```bash
# Generate sample data (if no real data available)
python src/data_preprocessing.py
```

### 3. Training

```bash
# Train the model
python yolo task=detect mode=predict model=runs/detect/yolov8_weed_detection/weights/best.pt source=processed_data/images/test/ 

# Training will create:
# - models/best.pt (trained model)
# - output/output_images/ (detailed results)
```

## 📊 Model Performance

Expected performance metrics:
- **mAP@0.5:** >80% (target requirement)
- **mAP@0.5:0.95:** >60%
- **Precision:** >85%
- **Recall:** >80%

Results are saved in `output/metrics.json`

## 🔧 Configuration

### Training Configuration (`models/data.yaml`)

```yaml
model: yolov8n.pt      # Model architecture
epochs: 100            # Training epochs
imgsz: 640            # Input image size
batch: 16             # Batch size
lr0: 0.01             # Learning rate
patience: 10          # Early stopping patience
device: auto          # Device (auto/cpu/cuda)
```

### Dataset Configuration (`data/data.yaml`)

```yaml
path: /absolute/path/to/data
train: images/train
val: images/val  
test: images/test
nc: 2
names: ['rice', 'weed']
```

## 📈 Data Augmentation

The project includes automatic data augmentation:
- Random resized crop
- Horizontal flip
- Brightness/contrast adjustment

## 🎯 Class Information

| Class ID | Class Name | Description |
|----------|------------|-------------|
| 0        | rice       | Rice crops  |
| 1        | weed       | weeds       |


## 📝 Usage Examples

### Training with Custom Data

1. Place your images in appropriate folders:
   ```
   data/raw_data/images/
   ```

2. Create YOLO format labels in:
   ```
   data/raw_data/labels
   ```

## 🔍 Output Files

After training and prediction:

- `models/weights/best.pt` - Trained PyTorch model
- `output/metrics.json` - Performance metrics
- `output/output_images/` - Detailed prediction results

## ⚡ Performance Optimization

For better performance:

1. **GPU Training:** Ensure CUDA is available
2. **Batch Size:** Increase if you have more GPU memory
3. **Image Size:** Use 640x640 for balance of speed/accuracy
4. **Model Size:** Try yolov8s.pt or yolov8m.pt for higher accuracy

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory:**
   ```bash
   # Reduce batch size in config
   batch: 8  # instead of 16
   ```

2. **No GPU Detected:**
   ```bash
   # Force CPU training
   device: cpu
   ```

3. **Missing Dependencies:**
   ```bash
   pip install --upgrade ultralytics opencv-python torch torchvision
   ```

## 📚 Dependencies

- ultralytics>=8.0.0 (YOLOv8)
- opencv-python>=4.8.0 (Image processing)
- torch>=2.0.0 (Deep learning framework)
- pandas>=1.5.0 (Data manipulation)
- matplotlib>=3.5.0 (Visualization)
- albumentations>=1.3.0 (Data augmentation)

## 🏆 Submission Checklist

- [x] **Code runs without errors**
- [x] **Model achieves 76% mAP**
- [x] **ZIP contains all required folders**
- [x] **README explains how to train/detect**
- [x] **Requirements.txt provided**

## 👨‍💻 Author


**Name:** [ANWESHA CHATTERJEE]  
**Project:** Rice Field Weed Detection  
