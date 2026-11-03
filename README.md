# WEMP

Wavelet-Based Motion Deblurring via Event-Derived Edge Guidance and Motion Priors

이벤트 기반 영상 디블러링을 위한 WEMP의 모델, 손실 함수, HDF5 데이터 로더 및 학습·평가 코드를 제공합니다.

## 설치

Python 3.10 이상과 호환되는 PyTorch/torchvision 환경을 사용합니다.

python -m pip install -e .

# LPIPS 평가 시
python -m pip install -e '.[lpips]'

## 주요 구현 설정

| 항목 | 설정 |
|---|---|
| Feature width / depth | 128 / 4 |
| Event 입력 | 로더의 9채널 voxel에서 중앙 채널을 제거한 8채널 |
| Cost-volume 해상도 / 탐색 반경 | 입력 해상도의 1/4 / 3 |
| Cost-volume softmax temperature | 0.1 |
| Memory / prompt 개수 | 각 memory 모듈에서 256개 |
| Memory 초기화 | 표준정규분포 |
| Memory 갱신 | 학습 forward마다 EMA 적용, 계수 0.9 |
| Prompt 초기화·학습 | [0, 1) 균등분포 초기화, 역전파로 학습 |
| Uncertainty head | 최종 decoder feature → 3×3 Conv → ELU → 1×1 Conv, 3채널 출력 |
| Uncertainty loss | exp(-U)로 가중한 Laplacian 복원 오차의 평균 L1, 가중치 50 |
| Frequency loss | FFT의 실수·허수 성분에 대한 평균 L1, 가중치 0.1, norm="backward" |

Memory 갱신에는 gradient를 사용하지 않으며, softmax 기반 retrieval을 통해 motion feature와 prompt bank는 역전파로 학습됩니다. 실제 모델에서 사용하는 EMA 계수는 0.9입니다.

주요 코드:

- 모델 및 uncertainty head (basicsr/models/archs/ours_arch.py)
- Cost-volume, wavelet fusion 및 memory 모듈 (basicsr/models/archs/main_arch_ours.py)
- 공통 블록 (basicsr/models/archs/arch_util_ours.py)
- 학습 및 uncertainty loss (basicsr/models/image_event_restoration_ours_model.py)

## 데이터 형식

Sequence별 HDF5 파일에 다음 그룹과 CHW 배열을 준비합니다.

images/image000000000        uint8, (3, H, W)
sharp_images/image000000000  uint8, (3, H, W)
voxels/voxel000000000        (9, H, W)
masks/mask000000000          uint8, (1, H, W), return_mask=true인 경우

동일한 번호의 영상·정답·voxel·mask는 같은 관측 구간에 대응해야 합니다. YAML의 dataroot에는 HDF5 파일이 있는 폴더 또는 파일 경로 목록 CSV를 지정합니다. 데이터와 raw event의 HDF5 변환 코드는 포함하지 않습니다.

## 학습 및 평가

YAML의 데이터 경로와 checkpoint 경로를 수정한 뒤 실행합니다.

# 실행 환경 점검
python -m tools.smoke_test

# 2-GPU 학습
torchrun --nproc_per_node=2 --module basicsr.train -opt options/train.yml --launcher pytorch

# 평가
python -m basicsr.test -opt options/test.yml

평가할 가중치는 options/test.yml의 path.pretrain_network_g에 지정합니다. 데이터와 pretrained weights는 현재 저장소에 포함되어 있지 않습니다.

## 논문 학습 프로토콜

GoPro 주실험은 총 batch size 32로 400,000 iterations 동안 학습합니다. AdamW를 사용하며, 초기 learning rate는 2e-4, betas는 (0.9, 0.99), weight decay는 1e-4입니다. Cosine annealing으로 learning rate를 1e-7까지 감소시키며, 256×256 random crop과 flip·rotation augmentation을 적용합니다.

실제 데이터셋은 GoPro 사전학습 모델에서 시작하여 데이터셋별 training split으로 독립적으로 fine-tuning합니다.

| 데이터셋 | Fine-tuning iterations | 평가 checkpoint |
|---|---:|---|
| REBlur | 400 | 마지막 iteration |
| EVRB | 1,000 | 마지막 iteration |
| REVD | 1,000 | 마지막 iteration |

GoPro와 실제 데이터셋 모두 사전에 정한 마지막 iteration의 checkpoint를 평가에 사용합니다. Validation 또는 test 점수를 기준으로 checkpoint를 선택하지 않습니다.

제공된 options/train.yml은 300,000 iterations, 총 batch size 4, seed 1의 학습 예제입니다. 논문 GoPro 주실험을 재현할 때는 위 프로토콜에 맞게 설정을 변경해야 합니다.

## Acknowledgments

이 코드는 EFNet 및 BasicSR을 기반으로 합니다. 관련 저작권 고지와 LICENSE를 유지합니다.
