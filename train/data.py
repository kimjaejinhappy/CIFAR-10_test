# train/data.py

# --- 1. 필요한 도구들 가져오기 ---
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# --- 2. 데이터 로더 생성 함수 정의 ---
def get_cifar10_loaders(batch_size=128, val_ratio =0.1):
    """
    CIFAR-10 데이터셋을 위한 (train, validatio, test)용
    데이터 로더(DataLoader)를 생성하여 반환합니다.

    Args:
        batch_size (int): 한 번에 모델에 주입할 데이터의 양 (배치 크기)
        val_ratio (float): 전체 훈련 데이터 중 검증 데이터로 사용할 비율 (예: 0.1은 10%)
    Returns:
        tuple: (train_loader, val_loader, test_loader) 튜플
    """

    # --- 3. 이미지 변환 규칙(Transforms) 정의 ---
    # 훈련용 데이터에 적용할 변환 규칙
    # 모델이 더 강인해지도록 데이터를 살짝씩 변형(증강)시킵니다.
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 이미지를 32x32 크기로 무작위로 자릅니다. 패딩을 줘서 가장자리 정보 손실을 막습니다.
        transforms.RandomHorizontalFlip(),     # 50% 확률로 이미지를 좌우로 뒤집습니다.
        transforms.ToTensor(),                 # PIL Image나 Numpy 배열을 PyTorch 텐서(Tensor)로 변환합니다. (값의 범위: 0.0 ~ 1.0)
        transforms.Normalize(                  # 텐서의 값을 정규화합니다. (평균 0.5, 표준편차 0.5) -> (값의 범위: -1.0 ~ 1.0)
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ),
    ])


 # 검증 및 테스트용 데이터에 적용할 변환 규칙 (Data Augmentation 제외)
 # 검증/테스트 데이터는 훈련 데이터와 동일한 정규화만 적용하며, 데이터 증강은 하지 않습니다.
    transform_test_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ),
    ])

    # --- 4. CIFAR-10 데이터셋 다운로드 및 변환 규칙 적용 ---
    # train 데이터셋
    full_train_dataset = datasets.CIFAR10(
        root='./data',      # 데이터가 저장될 경로
        train=True,         # 훈련용 데이터셋임을 명시
        download=True,      # 해당 경로에 데이터가 없으면 자동으로 다운로드
        transform=transform_train # 위에서 정의한 훈련용 변환 규칙 적용
    )


    # Train/Vlidaiton 데이터셋 분리
    total_train_size = len(full_train_dataset)
    val_size = int(val_ratio* total_train_size)
    train_size= total_train_size - val_size

    # 데이터셋을 Train과 Validation으로 무작위 분리합니다. random_split함수(총데이터, [분리하고 싶은 size1, size2])
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator= torch.Generator().manual_seed(42)  # 처음만 랜덤으로 분리되고, 분리된 순간, trian, vali 데이터셋 고정

    )

    # test 데이터셋
    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,        # 테스트용 데이터셋임을 명시
        download=True,
        transform=transform_test_val  # 위에서 정의한 테스트용 변환 규칙 적용
    )


    # --- 5. 데이터 로더(DataLoader) 생성 --- 한번에 많은 양 학습하면 효율적이지 않으니, 데이터를 나눠서 공급하는 거임
    # trian용 데이터 로더
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,       # 훈련 시 데이터 순서를 무작위로 섞음 (매우 중요!)
        num_workers=4,     # 데이터를 불러올 때 사용할 CPU 프로세스 수 (데이터 로딩 속도 향상)
        pin_memory=True # GPU 사용 시 데이터 전송 속도 향상
    )

    #validation용 데이터 로더
    val_loader = DataLoader(
        dataset= val_dataset,
        batch_size= batch_size,
        shuffle= False,  # val시에도 순서 섞을 필요없음. 기억을 못한다고 하네. 이유는 모름
        num_workers=4,
        pin_memory= True
    )

    # 테스트용 데이터 로더
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,      # test 시에는 순서를 섞을 필요가 없음
        num_workers=4,
        pin_memory= True
    )

    # --- 6. 생성된 데이터 로더 반환 ---
    return train_loader, val_loader, test_loader

# --- 7. 다른 파일에서 이 파일을 import 할 때, 어떤 함수를 공개할지 명시 (선택사항) ---
__all__ = ['get_cifar10_loaders']