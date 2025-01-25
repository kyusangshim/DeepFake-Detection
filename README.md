# DeepFake-Detection using EfficientNet and ViT

## 1. 소개
본 프로젝트는 영상 속 딥페이크(Deepfake)를 탐지하기 위해 딥러닝 기법을 활용한 연구 결과를 소개합니다. 본 연구는 얼굴의 전역(Global) 및 국소(Local) 특징을 효과적으로 추출하고 분석하여, 신원 및 표정 변조를 모두 탐지할 수 있는 모델을 개발하였습니다. 

### 🛠️ 주요 특징
- **이중 스트림 모델**  
  - EfficientNet-B0: 얼굴의 전역적인 특징 추출.  
  - EfficientNet-B1: 얼굴의 세부적인 특징 추출(눈, 코, 입 등).  
- **ViT 기반 시간적 이상 탐지**  
  - 두 스트림의 특징을 결합하여 Vision Transformer (ViT)에 입력.  
  - 시간적 상관성을 분석해 딥페이크 여부를 판별.  
- **전이 학습(Transfer Learning)**  
  - 사전 학습된 EfficientNet과 ViT를 활용하여 학습 효율성과 정확도를 극대화.
