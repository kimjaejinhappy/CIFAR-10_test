#import torch

#print(torch.__version__)
##print(torch.cuda.is_available())
#

# main.py

# --- 1. 필요한 라이브러리 및 모듈 가져오기 ---
import torch
import torch.nn as nn
import argparse  # 커맨드 라인 인자(argument)를 파싱하는 라이브러리

# 우리가 만든 모듈들을 가져옵니다.
from models.resnet import get_resnet
from train.data import get_cifar10_loaders
from train.supervised import train_one_epoch, evaluate

# --- 2. 메인 실행 함수 정의 ---
def main():
    # --- 2-1. Argparse를 사용한 하이퍼파라미터 설정 ---
    # 사용자가 터미널에서 직접 훈련 옵션을 지정할 수 있게 합니다.
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
    parser.add_argument('--model', default='resnet18', type=str,
                        help='model name (e.g., resnet18, resnet34)')
    parser.add_argument('--epochs', default=10, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--batch-size', default=128, type=int,
                        help='mini-batch size')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        help='initial learning rate')
    args = parser.parse_args()

    # --- 2-2. 기본 환경 설정 ---
    # GPU 사용 가능 여부를 확인하고, 사용할 장치를 설정합니다.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("=" * 50)
    print(f"Start training {args.model} for {args.epochs} epochs.")

    # --- 2-3. 부품 조립 (데이터, 모델, 손실함수, 옵티마이저) ---
    # 1. 데이터 로더 준비
    train_loader, test_loader = get_cifar10_loaders(batch_size=args.batch_size)

    # 2. 모델(엔진) 생성
    # get_resnet 함수로 모델을 가져와서 지정된 장치(GPU 또는 CPU)로 보냅니다.
    model = get_resnet(name=args.model, num_classes=10).to(device)

    # 3. 손실 함수(Criterion)와 옵티마이저(Optimizer) 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- 2-4. 훈련 루프(Training Loop) 시작 ---
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 50)

        # 1. 훈련 지시 (supervised.py의 함수 호출)
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # 2. 평가 지시 (supervised.py의 함수 호출)
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )

        # 3. 결과 출력
        print(f"\n[Epoch {epoch+1} Results]")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Test Loss : {test_loss:.4f} | Test Acc : {test_acc:.2f}%")
        print("=" * 50)

    print("Training finished!")

# --- 3. 스크립트 실행 시작점 ---
if __name__ == '__main__':
    main()