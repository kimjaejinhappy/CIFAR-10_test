#목차 및 주석 설계
#Basic block: resnet에서 가장 기본이 되는 블록
# “입력”을 두 번의 Conv + BatchNorm + ReLU를 거쳐 출력으로 만들고, 입력(identity)과 출력을 더해주는 skip connection(잔차 연결) 구조입니다.
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """
    ResNet의 가장 기본이 되는 Residual Block
    - 입력(x)을 두 번의 conv + bn + relu로 처리
    - 입력과 출력을 더함 (skip connection)
    - 필요시 downsample로 차원을 맞춤
    """
    expansion = 1  # 출력 채널 확장 계수 (Bottleneck 블록에서는 4) ->? 이게 뭐지?

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        in_channels: 입력 채널 수
        out_channels: 출력 채널 수
        stride: 첫번째 conv에만 적용. 주로 블록의 첫번째에만 사용
        downsample: 입력 x와 출력(out)의 크기가 다를 때, x를 변환해주는 모듈 (보통 1x1 conv)
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample  # None 또는 nn.Sequential(1x1 conv + BN)

    def forward(self, x):
        identity = x  # skip connection을 위한 원본 입력 저장

        # 첫 번째 conv + BN + ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 두 번째 conv + BN (ReLU는 더하기 후에)
        out = self.conv2(out)
        out = self.bn2(out)

        # 차원이 다르면 downsample로 변환
        if self.downsample is not None:
            identity = self.downsample(x)

        # skip connection(입력과 합침)
        out += identity
        out = self.relu(out)

        return out
    

class ResNet(nn.Module):
    """
    전체 ResNet 네트워크 클래스
    - BasicBlock을 쌓아 원하는 깊이의 모델 생성
    - CIFAR-10 (3x32x32 이미지)에 맞춰 초반 conv를 조정
    """

    def __init__(self, block, layers, num_classes=10):
        """
        block: 사용할 블록 클래스 (예: BasicBlock)
        layers: 각 스테이지마다 block 몇 개 쌓을지 리스트(예: [2,2,2,2] for ResNet18)
        num_classes: 분류 클래스 수
        """
        super().__init__()
        self.in_channels = 64  # 첫 블록의 입력 채널

        # 입력 레이어: CIFAR-10은 이미지가 작아서 kernel=3, stride=1, padding=1 사용
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 네 개의 스테이지 (채널 수 증가, 다운샘플링)
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # AdaptiveAvgPool2d: 입력 크기와 상관없이 1x1로 평균
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride):
        """
        블록들을 쌓아서 하나의 stage(layer)로 만듦
        - block: 사용할 블록 클래스
        - out_channels: 이 layer의 출력 채널 수
        - blocks: 몇 개 쌓을지
        - stride: 첫 블록에서만 적용 (downsampling)
        """
        downsample = None
        layers = []

        # 입력과 출력 크기가 다르면 downsample로 projection
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        # 첫 번째 블록 (stride와 downsample 적용)
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion

        # 나머지 블록 (stride=1, downsample 없음)
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 입력층
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # 네 개의 stage
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 분류를 위한 평균 풀링, flatten, FC
        x = self.avgpool(x)      # (B, 512, 1, 1)
        x = torch.flatten(x, 1)  # (B, 512)
        x = self.fc(x)           # (B, num_classes)

        return x
    
def get_resnet(name='resnet18', num_classes=10): #이 부분이 resnet 18, resnet34일때 basicblock을 어떻게 사용할건지 알려주는 코드
    """
    ResNet 모델을 이름으로 쉽게 생성하는 함수
    - name: 'resnet18' 또는 'resnet34'
    - num_classes: 분류 클래스 수 (CIFAR-10은 10)
    """
    if name == 'resnet18':
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
    elif name == 'resnet34':
        return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)
    else:
        raise ValueError(f"지원하지 않는 모델 이름입니다: {name}")
    
    
__all__ = ['BasicBlock', 'ResNet', 'get_resnet']
