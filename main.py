#import torch

#print(torch.__version__)
##print(torch.cuda.is_available())
#

# main.py

# --- 1. 필요한 라이브러리 및 모듈 가져오기 ---
import torch
import torch.nn as nn
import argparse  # 커맨드 라인 인자(argument)를 파싱하는 라이브러리
import os

#TensorBoard 를 위한 묘듈
from torch.utils.tensorboard import SummaryWriter

# 우리가 만든 모듈들을 가져옵니다.
from models.resnet import get_resnet
from train.data import get_cifar10_loaders
from train.supervised import train, final_test 
#위에 코드 지울 예정 for train.supervised import train

# --- 2. 메인 실행 함수 정의 ---
def main():
    # --- 2-1. Argparse를 사용한 하이퍼파라미터 설정 --- : 사용자가 터미널에서 직접 훈련 옵션을 지정할 수 있게 합니다.
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
    parser.add_argument('--model', default='resnet18', type=str,
                        help='model name (e.g., resnet18, resnet34)')
    parser.add_argument('--epochs', default=50, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--batch-size', default=128, type=int,
                        help='mini-batch size')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('--patience', default=10, type=int,
                        help='Early Stopping patience(epochs to wait for improvement)')
    args = parser.parse_args()

    # --- 2-2. 기본 환경 설정 ---
    # GPU 사용 가능 여부를 확인하고, 사용할 장치를 설정합니다.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("=" * 50)
    print(f"Start training {args.model} for {args.epochs} epochs.")

    # --- 2-3. TensorBoard Writer 초기화
    writer = SummaryWriter(log_dir= f'runs/{args.model}_cifar10')  # utils.tensorboar 묘듈에 포함된 클래스, 파일이 저장될 폴더 경로를 설정하는거임
    print(f"Tensorboar logs saved to: runs/{args.model}_cifar10")
    print("="*50)

    # --- 2-4. 부품 조립 (데이터, 모델, 손실함수, 옵티마이저) ---
    # 1. 데이터 로더 준비: train,val, test
    train_loader, val_loader, test_loader = get_cifar10_loaders(batch_size=args.batch_size)

    # 2. 모델(엔진) 생성
    # get_resnet 함수로 모델을 가져와서 지정된 장치(GPU 또는 CPU)로 보냅니다.
    model = get_resnet(name=args.model, num_classes=10).to(device)

    # 3. 손실 함수(Criterion)와 옵티마이저(Optimizer) 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- 2-5. 훈련 시작 및 최적 모델 가중치 경로 받기
    best_model_path = train(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader,
        optimizer=optimizer, 
        criterion=criterion, 
        device= device, 
        epochs=args.epochs,
        patience= args.patience,
        writer=writer)

    # 최종 테스트: final_test 호출
    final_test(
        best_model_path= best_model_path,
        model_name= args.model,
        test_loader=test_loader,
        criterion= criterion,
        device=device
        )

    #tensorboard닫기
    writer.close()

# --- 3. 스크립트 실행 시작점 ---
if __name__ == '__main__':
    main()