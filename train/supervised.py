# train/supervised.py

# --- 1. 필요한 모든 도구와 부품 가져오기 ---
import torch
import torch.nn as nn
from tqdm import tqdm  # 학습 진행 상황을 보여주는 TQDM 라이브러리

# 우리가 직접 만든 '만능 공구함'에서 도구들을 가져옵니다.
from .utils import AverageMeter, accuracy


# --- 2. 훈련 작전 계획서: train_one_epoch 함수 ---
def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """
    모델을 1 에포크(epoch) 동안 훈련시키는 함수입니다.

    Args:
        model (nn.Module): 훈련시킬 모델
        train_loader (DataLoader): 훈련용 데이터 로더
        optimizer: 옵티마이저 (예: Adam, SGD)
        criterion: 손실 함수 (예: CrossEntropyLoss)
        device: 학습에 사용할 장치 (예: 'cuda' or 'cpu')

    Returns:
        tuple: (평균 손실, 평균 정확도)
    """
    # --- 2-1. 훈련 준비 ---
    model.train()  # 모델을 '훈련 모드'로 전환합니다. (Dropout, BatchNorm 등에 영향을 줌)

    # 손실과 정확도를 기록할 '평균 계량기'를 준비합니다.
    losses = AverageMeter()
    accs = AverageMeter()

    # --- 2-2. 데이터 로더를 이용해 배치 단위로 훈련 시작 ---
    # tqdm으로 train_loader를 감싸면 멋진 진행 바(progress bar)가 생깁니다.
    for images, labels in tqdm(train_loader, desc="Training"):
        # 데이터를 GPU나 CPU로 보냅니다.
        images = images.to(device)
        labels = labels.to(device)

        # --- 2-3. 모델 훈련의 핵심 5단계 ---
        # 1. 순전파(Forward pass): 이미지를 모델에 넣어 예측값을 얻습니다.
        outputs = model(images)
        loss = criterion(outputs, labels) # 예측값과 실제 정답으로 손실을 계산합니다.

        # 2. 기울기 초기화: 이전 배치의 기울기가 남아있지 않도록 깨끗하게 지웁니다.
        optimizer.zero_grad()

        # 3. 역전파(Backward pass): 손실을 기반으로 기울기를 계산합니다.
        loss.backward()

        # 4. 가중치 업데이트: 계산된 기울기를 이용해 모델의 파라미터를 업데이트합니다.
        optimizer.step()
        # ---------------------------------------------

        # --- 2-4. 결과 기록 ---
        # 이번 배치의 정확도를 계산합니다.
        acc = accuracy(outputs, labels)

        # 현재 배치의 손실과 정확도를 계량기에 기록합니다.
        # images.size(0)는 현재 배치의 실제 데이터 개수 (보통 128)입니다.
        losses.update(loss.item(), images.size(0))
        accs.update(acc, images.size(0))

    # --- 2-5. 1 에포크 훈련 결과 반환 ---
    # 계량기에 기록된 전체 평균 손실과 정확도를 반환합니다.
    return losses.avg, accs.avg


# --- 3. 평가 작전 계획서: evaluate 함수 ---
def evaluate(model, test_loader, criterion, device):
    """
    모델의 성능을 평가하는 함수입니다.

    Args:
        model (nn.Module): 평가할 모델
        test_loader (DataLoader): 테스트(검증)용 데이터 로더
        criterion: 손실 함수
        device: 평가에 사용할 장치

    Returns:
        tuple: (평균 손실, 평균 정확도)
    """
    # --- 3-1. 평가 준비 ---
    model.eval()  # 모델을 '평가 모드'로 전환합니다.

    losses = AverageMeter()
    accs = AverageMeter()

    # --- 3-2. 기울기 계산 비활성화 ---
    # 평가는 훈련이 아니므로, 기울기를 계산할 필요가 없습니다.
    # torch.no_grad()를 사용하면 계산 속도가 빨라지고 메모리를 아낄 수 있습니다.
    with torch.no_grad():
        # --- 3-3. 배치 단위로 평가 시작 ---
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            # 순전파(Forward pass)로 예측값을 얻고 손실을 계산합니다.
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 정확도를 계산하고 계량기에 기록합니다.
            acc = accuracy(outputs, labels)
            losses.update(loss.item(), images.size(0))
            accs.update(acc, images.size(0))

    # --- 3-4. 평가 결과 반환 ---
    return losses.avg, accs.avg