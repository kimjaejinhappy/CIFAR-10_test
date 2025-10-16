# train/utils.py

# --- 1. 필요한 도구 가져오기 ---
import torch

# --- 2. 평균 계량기 (AverageMeter) 클래스 ---
class AverageMeter:
    """
    Epoch 내에서 손실(loss)이나 정확도(accuracy) 같은 값들의
    평균을 계산하고 저장하는 클래스.
    마치 평균을 자동으로 계산해주는 디지털 스톱워치와 같습니다.
    """
    def __init__(self):
        """스톱워치를 처음 켰을 때 모든 값을 0으로 초기화합니다."""
        self.val = 0    # 현재 값 (Current value)
        self.avg = 0    # 평균 값 (Average value)
        self.sum = 0    # 모든 값의 합 (Sum of values)
        self.count = 0  # 값의 개수 (Count of values)
        self.reset()    # 모든 값을 0으로 초기화

    def reset(self):
        """스톱워치의 리셋 버튼을 누르는 것과 같습니다."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1): #여기서 n=1은 그냥 기본값이고 추후 superviesd에서 batch size 넣을 거임.
        """
        새로운 값을 받아서 평균을 업데이트합니다.
        val: 새로운 값 (예: 이번 배치의 평균 손실)
        n: 새로운 값이 몇 개의 데이터로부터 계산되었는지 (예: 배치 사이즈)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# --- 3. 정확도 계산기 (accuracy) 함수 ---
def accuracy(output, target):
    """
    모델의 출력(output)과 실제 정답(target)을 받아
    Top-1 정확도를 계산합니다.

    Args:
        output (torch.Tensor): 모델의 예측 결과 (보통 로짓 형태, shape: [배치 사이즈, 클래스 수])
        target (torch.Tensor): 실제 정답 레이블 (shape: [배치 사이즈])

    Returns:
        float: 정확도 (%)
    """
    # 이 블록 안에서는 기울기 계산을 하지 않도록 설정하여 메모리를 아끼고 속도를 높입니다.
    with torch.no_grad():
        # --- 1. 모델의 예측 중 가장 높은 점수를 받은 클래스를 찾습니다. ---
        # output 텐서의 1번 차원(dim=1)을 따라 가장 큰 값의 인덱스를 찾습니다.
        # 이 인덱스가 바로 모델이 예측한 클래스입니다.
        pred = torch.argmax(output, dim=1) #dim=1은 각 문제(각 줄)마다 최고 점수를 찾으라는 방향을 알려주는 중요한 옵션

        # --- 2. 예측과 정답이 일치하는지 확인합니다. ---
        # pred와 target이 같은지 비교합니다. 결과는 [True, False, True, ...] 형태의 텐서가 됩니다.
        # .sum()으로 True(1로 취급됨)의 개수를 세고, .item()으로 텐서에서 숫자만 꺼냅니다.
        correct = (pred == target).sum().item()

        # --- 3. 정확도를 퍼센트로 계산합니다. ---
        # (맞은 개수 / 전체 개수) * 100
        acc = correct / len(target) * 100.0
        return acc

# --- 4. 다른 파일에서 이 파일을 import 할 때, 어떤 것들을 공개할지 명시 (선택사항) ---
__all__ = ['AverageMeter', 'accuracy']