# DeepFake Detection using EfficientNet and ViT

## 1. Introduction
This project presents a deep learning-based approach for detecting deepfake content in videos. Our research focuses on effectively extracting and analyzing both global and local facial features to detect identity and expression manipulations. <br><br>



### (1) Key Features
- **Two-stream Model**  
  - EfficientNet-B0+GCN: Extracts global facial features
  - EfficientNet-B1: Extracts fine-grained facial features (eyes, nose, mouth, etc.)
- **ViT-based Temporal Anomaly Detection**  
  - Combines features from both streams and inputs them into a Vision Transformer (ViT)
  - Analyzes temporal correlations to determine deepfake authenticity
- **Transfer Learning**  
  - Leverages pre-trained EfficientNet and ViT to maximize learning efficiency and accuracy

<br><br>
 


### (2) Overall Model Architecture

![image](https://github.com/user-attachments/assets/3630e69b-2711-4929-b748-f94dff03e531)


1. **Global Stream**  
   - Uses the EfficientNet-B0+GCN model to extract global facial feature maps.
2. **Local Stream**  
   - Uses the EfficientNet-B1 model to extract detailed features from key facial regions.
3. **Feature Fusion (Sum)**  
   - Combines global and local features and inputs them into the Vision Transformer.
4. **시간적 분석**  
   - Uses ViT to analyze temporal relationships and detect deepfakes.
   - Extracts fine-grained temporal forgery patterns using two ViT variants: num_patches=16 and num_patches=1.<br><br>
  



## 2. Training Methodology

#### The model is trained using a pre-training and fine-tuning approach.<br><br>

### (1) Dataset
- FaceForensics++ dataset used.
- A total of 5000 samples were used, excluding FaceSwap.
- After preprocessing, 949 samples per method were retained, totaling **4745** samples.
- Training Data: **0–760 index** samples per method (760 × 5 = 3800 total samples).
- Validation Data: **760–849 index** samples per method (89 × 5 = 445 total samples).
- Test Data: **849–949** index samples per method (100 × 5 = 500 total samples).
- Testing was conducted with **200 pairs** (fake, real) per manipulation method.
- Another test dataset used: Celeb-DF-V2.<br><br>


### (2) Pre-training
![image](https://github.com/user-attachments/assets/938480c3-569b-4a7e-a02c-9c21bcdce651)
- Each stream was pre-trained separately, as shown in the figure above.
- ImageNet weights were used for pre-training.
- The local stream was trained using masked data showing only facial features.
- The trained weights were saved upon completion.<br><br>

### (3) Fine-tuning
![image](https://github.com/user-attachments/assets/70a10d3d-6ae5-4a3d-b26d-3064aab8f3ce)
- The entire model was assembled with the pre-trained weights applied to each stream.
- Fine-tuning was performed with a learning rate reduced to 1/10 of the pre-training rate.<br><br>




  



## 3. Results

### Performance Comparison

| Model              | C-DF | DF   | F2F  | NT   | FS   | Avg   |
|--------------------|-------|------|------|------|------|-------|
| Xception3D         | 0.62  | 0.79 | 0.64 | 0.66 | 0.77 | 0.696 |
| Resnext+LSTM       | 0.68  | 0.92 | 0.81 | 0.80 | 0.90 | 0.822 |
| ResNet3D50         | 0.72  | 0.85 | 0.79 | 0.74 | 0.86 | 0.792 |
| **Proposed Model** | 0.82  | 0.91 | 0.84 | 0.85 | 0.90 | 0.864 |

- The proposed model outperformed baseline models such as Xception3D, Resnext+LSTM, and ResNet3D50 in all cases.
  
