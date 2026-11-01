# WEMP 메인 모델 코드

산학124에서 내려받은 `Deblur-Unet` 사본에서 메인 모델의 학습·평가 경로를 추린 코드입니다.
아키텍처, 고주파 uncertainty loss, 공간·주파수 복원 loss, HDF5 로더와 실행 코드를 포함합니다.
모델과 loss의 계산식은 보존했습니다. 후속 CYCL·HFCDB·G022 등의 실험과 ablation 코드는 포함하지 않습니다.

## 설치

Python 3.10 이상과 서로 호환되는 PyTorch/torchvision을 사용합니다.

```bash
python -m pip install -e .
# LPIPS 평가가 필요한 경우
python -m pip install -e '.[lpips]'
```

LPIPS 가중치는 pyiqa의 다운로드 경로를 사용하며 최초 실행에 네트워크 연결이 필요할 수 있습니다.

## 구조와 손실

`ours_arch.py`: EFNetOurs, `main_arch_ours.py`: matching·wavelet·memory,
`arch_util_ours.py`: 공통 블록입니다. 폭 128, 깊이 4, event 8채널,
matching 해상도 1/4, radius 3입니다.

로더에서 전달한 voxel은 9채널이며 forward가 가운데 채널을 제거해 8채널로 만듭니다.
이미 8채널인 voxel을 직접 넣으면 현재 forward와 맞지 않습니다.
`last_var_head`는 3채널 보조 출력을 만들며,
`image_event_restoration_ours_model.py::_compute_hf_udl_loss`가 이를 학습에 사용합니다.
현재 구현은 예측·정답에 Laplacian을 적용하고 exp(-var)로 가중한 L1을 계산합니다.
이 코드를 임의의 Gaussian NLL 또는 보정된 확률적 분산으로 설명하지 않습니다.

## HDF5 입력

각 sequence 파일에 아래 그룹과 CHW 배열을 준비합니다.

```text
images/image000000000       uint8, (3,H,W)
sharp_images/image000000000 uint8, (3,H,W)
voxels/voxel000000000       (9,H,W)
masks/mask000000000         uint8, (1,H,W), return_mask=true일 때
```

같은 번호의 RGB·정답·voxel·mask는 같은 관측 구간에 대응해야 합니다.
폴더에는 sequence HDF5 파일만 두거나, dataroot에 파일 경로 목록을 담은 CSV를 지정합니다.
데이터와 raw event→HDF5 변환 파이프라인은 이 묶음에 포함되지 않습니다.
기존 로더의 voxel 정규화를 그대로 사용하므로 전체가 0인 voxel의 처리는 별도 확인이 필요합니다.

## 실행

```bash
python -m tools.smoke_test
torchrun --nproc_per_node=2 --module basicsr.train -opt options/train.yml --launcher pytorch
python -m basicsr.test -opt options/test.yml
```

먼저 YAML의 데이터 경로를 수정하고, 테스트에는 본인의 가중치를 `weights/wemp.pth`에 둡니다.
가중치와 데이터는 배포물에 포함하지 않았습니다. 데이터·가중치 없이 논문 점수가 재현됐다고 주장하지 않습니다.
`options/train.yml`은 124 사본의 30만 iteration·seed 1 학습 예제입니다.
논문 최종 실행의 전체 하이퍼파라미터와 동일하다고 검증된 설정은 아닙니다.
학습 중 모델 선택에는 별도로 준비한 validation split을 지정해야 합니다.
추론용 YAML의 network_g는 이 학습 예제와 동일한 메인 구조로 맞췄습니다.

현재 검증 상태: Python 구문과 내부 모듈 참조 검사를 통과했고, 아키텍처 3개 파일의
원본 해시 일치와 uncertainty loss·optimizer 갱신 함수의 AST 일치를 확인했습니다.
KT_CPU 접속 시간 초과로 실제 순전파·역전파·HDF5 실행 점검은 아직 수행하지 못했습니다.
배포 환경에서 `python -m tools.smoke_test`를 통과한 뒤 실행 검증 완료로 표시해야 합니다.

## 공개용 실행 경로 정리

메인 모델·HDF5 로더만 등록하도록 registry를 제한했습니다.
테스트가 datasets.val을 고정 참조하던 부분은 datasets.test도 읽도록 수정했습니다.
미사용 NIQE/LMDB 변환 도구의 자동 import를 제외하고 LPIPS import는 필요할 때만 수행합니다.
구조·손실 코드의 출처와 해시는 `source_manifest.json`에 있습니다.

## 출처

이 코드는 EFNet/BasicSR 기반입니다. 원본 저작권 주석과 LICENSE를 유지했습니다.
WEMP 논문 서지, 저자, 최종 가중치 다운로드 위치는 공개 전에 저자가 확정해야 합니다.
