# train/supervised.py

# --- 1. 필요한 모든 도구와 부품 가져오기 ---
import torch
import torch.nn as nn
import os
import time


# from tqdm import tqdm  # 학습 진행 상황을 보여주는 TQDM 라이브러리  -> 이거 쓰지 말고 한번 printf로 해보라고 함. 그래야 내가 보고 싶은 것들을 볼 수 있다는 게 재빈이형의 코멘트

# 우리가 직접 만든 '만능 공구함'에서 도구들을 가져옵니다.
from .utils import AverageMeter, accuracy, EarlyStopping
from models.resnet import get_resnet


# --- 2. 훈련 작전 계획서: train_one_epoch 함수 ---
def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, num_epochs):
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


    #총 배치수
    total_batches = len(train_loader)

    # --- 2-2. 데이터 로더를 이용해 배치 단위로 훈련 시작 ---
    # for A in enumerate(B): B에 있는 index, 값들을 꺼내서 변수 A에 넣는다.
    for batch_idx, (images, labels) in enumerate(train_loader): 
        
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

        # --- 2-4. 결과 기록 --- 배치의 손실과 정확도를 계량기에 기록합니다.-> 밑에 3개 코드 뭔지 모르겠음 
        acc = accuracy(outputs, labels)
        losses.update(loss.item(), images.size(0))
        accs.update(acc, images.size(0))

        if (batch_idx+1)% 100 == 0:
            print(f"Epoch{epoch+1}/{num_epochs}|Batch [{batch_idx+1}/{total_batches}]"
                  f"| Loss: {losses.val:.4f} (Avg: {losses.avg: .4f})"
                  f"| Acc: {accs.val:.2f}% (Avg: {accs.avg: .2f}%)")

    # --- 2-5. 1 에포크 훈련 결과 반환 ---
    # 계량기에 기록된 전체 평균 손실과 정확도를 반환합니다.
    return losses.avg, accs.avg


# --- 3. 평가 작전 계획서: evaluate 함수 ---
def evaluate(model, data_loader, criterion, device):
    """
    모델의 성능을 평가하는 함수입니다.

    Args:
        model (nn.Module): 평가할 모델
        data_loader (DataLoader): 테스트(검증)용 데이터 로더
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
        for images, labels in data_loader:
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

#-----------------------------------------------------------------------------------------

# train 함수: 지정된 에포크마다 훈련하는 함수 
def train(model, train_loader, val_loader, optimizer, criterion, device, epochs, patience, writer):
    # 지정된 에포크만큼 후녈나혹, validation loss를 기반으로 early stopping함

    #1. early stooping 객체 및 저장 경로 준비
    early_stopper = EarlyStopping(patience=patience, verbose = True, mode = "min")

    #모델 가중치 저장할 경로 설정 ( 현재 폴더에 checkpoint 폴더 생성 후 저장)
    os.makedirs('checkpoint', exist_ok= True)
    best_model_path = os.path.join('checkpoint', 'best_model.pth')

    start_time = time.time()

    #2. 에포크 루프 시작
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 50)

        #train
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, epochs)

        #validation
        val_loss, val_acc = evaluate(model, val_loader, criterion, device )

        # 결과
       # writer.add_scalar('Loss/Train', train_loss, epoch)
       #writer.add_scalar('Loss/Validation', val_loss, epoch)
        #writer.add_scalar('Accuracy/Train', train_acc, epoch)
        #writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        writer.add_scalars('Accuracy',{'Train': train_acc, 'Validation': val_acc}, epoch)
        writer.add_scalars('Loss', {'Train': train_loss, 'Validation': val_loss}, epoch)
        early_stopper(val_loss, model, best_model_path)

        if early_stopper.early_stop:
            print("\n-----------------")
            print(f"Early stopping: Validation loss did not improve for {patience} epochs.")
            print("훈련을 중단합니다")
            print("---------------------------")
            break

    end_time = time.time()
    elapesed_time = end_time - start_time
    print(f"Training finished!Total time: {elapesed_time:.2f} seconds.")

    return best_model_path

# Final test
def final_test(best_model_path, model_name, test_loader, criterion, device):
    """
    저장된 최적 모델을 로드하여 Test 데이터셋에 대한 최종 성능을 평가하고 출력합니다.

    Args:
     best_model_path (str): 저장된 최적 모델 가중치 경로
     model_name (str): 사용할 모델 이름 (예: 'resnet18')
     test_loader (DataLoader): Test 데이터 로더
     criterion: 손실 함수
     device: 사용할 장치
    """
    if best_model_path and os.path.exists(best_model_path):
       print("\n" + "=" * 50)
       print("Starting Final Test on the Best Saved Model...")
        # 1. 최적 모델 로드 (구조 재정의 후 저장된 가중치 불러오기)
        # NOTE: model_name을 사용하여 모델 구조를 다시 만듭니다.
       best_model = get_resnet(name=model_name, num_classes=10).to(device)
       best_model.load_state_dict(torch.load(best_model_path))

      # 2. Test 데이터셋으로 평가

       test_loss, test_acc = evaluate(best_model, test_loader, criterion, device)
       print("\n--- Final Test Results ---")
       print(f"Model: {model_name} (Best Validation Model)")
       print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
       print("=" * 50)
    else:
        print("\n[Warning] Best model path not found. Skipping final test.")


# 외부 공개함수 명시
__all__ = ['train', 'evaluate', 'final_test']