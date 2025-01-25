# DeepFake-Detection using EfficientNet and ViT

## 1. 소개
본 프로젝트는 영상 속 딥페이크(Deepfake)를 탐지하기 위해 딥러닝 기법을 활용한 연구 결과를 소개합니다. 본 연구는 얼굴의 전역(Global) 및 국소(Local) 특징을 효과적으로 추출하고 분석하여, 신원 및 표정 변조를 모두 탐지할 수 있는 모델을 개발하였습니다. 



### 🛠️ 주요 특징
- **이중 스트림 모델**  
  - EfficientNet-B0+GCN: 얼굴의 전역적인 특징 추출
  - EfficientNet-B1: 얼굴의 세부적인 특징 추출(눈, 코, 입 등)
- **ViT 기반 시간적 이상 탐지**  
  - 두 스트림의 특징을 결합하여 Vision Transformer (ViT)에 입력. 
  - 시간적 상관성을 분석해 딥페이크 여부를 판별
- **전이 학습(Transfer Learning)**  
  - 사전 학습된 EfficientNet과 ViT를 활용하여 학습 효율성과 정확도를 극대화
 


### 🛖 전체 모델 구조

![image](https://github.com/user-attachments/assets/3630e69b-2711-4929-b748-f94dff03e531)


1. **전역 스트림(Global Stream)**  
   - EfficientNet-B0+GCN 모델을 활용해 얼굴 전체의 특징 맵을 생성
2. **국소 스트림(Local Stream)**  
   - EfficientNet-B1 모델을 통해 얼굴의 주요 세부 영역에서 세밀한 특징을 추출
3. **출력 결합(Sum)**  
   - 전역 및 국소 스트림의 특징을 합하여 Vision Transformer에 입력
4. **시간적 분석**  
   - ViT로 시간적 관계를 분석하여 딥페이크 여부를 최종 판별
   - num_patchs=16, num_patchs=1 두 개의 ViT로 시간적 위조 특성 세분화 추출
  

## 2. 학습 방법

**사전 학습 후 미세 조정을 하는 방식으로 학습 진행**

### 데이터 셋
- FaceForensics++ 데이터 사용
- 해당 데이터는 5개의 위조 방식(Deepfakes, Face2Face, FaceSwap, NeuralTextures 등)과 원본 데이터 각각 1000개씩 포함하여 총 **6000개**의 영상으로 구성
- 해당 실험에서는 FaceSwap을 제외한 총 5000개의 데이터 사용
- 


### 사전 학습
![image](https://github.com/user-attachments/assets/ce0a10ff-4fb3-443b-bff2-b7b3e5e706a5)


  



## 3. 결과

### 성능 비교

| Model              | C-D F | DF   | F2F  | NT   | FS   | Avg   |
|--------------------|-------|------|------|------|------|-------|
| Xception3D         | 0.62  | 0.79 | 0.64 | 0.66 | 0.77 | 0.696 |
| Resnext+LSTM       | 0.68  | 0.92 | 0.81 | 0.80 | 0.90 | 0.822 |
| ResNet3D50         | 0.72  | 0.85 | 0.79 | 0.74 | 0.86 | 0.792 |
| **제안 모델** | 0.82  | 0.91 | 0.84 | 0.85 | 0.90 | 0.864 |

- 비교 모델인 Xception3D, Resnext+LSTM, ResNet3D50에 비해 모두 높은 성능을 보임.
  
