import torch
import sys
sys.path.insert(0, 'src')
from models.components.protonet import ResNet, BasicBlock
from omegaconf import OmegaConf

# Load config
conf = OmegaConf.load('configs/train.yaml')

# Create model
model = ResNet(conf.features)
model.eval()

# Test input (17, 128) - standard audio input
x = torch.randn(1, 17, 128)

print('='*60)
print('ResNet Architecture Analysis')
print('='*60)
print(f'Input shape: {x.shape}')
print()

# Track shapes through layers
x_reshaped = x.view(-1, 1, 17, 128)
print(f'After reshape: {x_reshaped.shape}')

with torch.no_grad():
    # Layer 1
    x1 = model.layer1(x_reshaped)
    print(f'After Layer1: {x1.shape}')
    
    # Layer 2
    x2 = model.layer2(x1)
    print(f'After Layer2 (CIFE source): {x2.shape}')
    
    # Layer 3
    x3 = model.layer3(x2)
    print(f'After Layer3 (CIFE target): {x3.shape}')
    
    # Layer 4 (if enabled)
    if conf.features.layer_4:
        x4 = model.layer4(x3)
        print(f'After Layer4: {x4.shape}')
    else:
        print('Layer4: Disabled')
        x4 = x3
    
    # Pooling
    x_pooled = model.pool_avg(x4)
    print(f'After AdaptiveAvgPool2d: {x_pooled.shape}')
    
    # Final flatten
    x_final = x_pooled.view(x_pooled.size(0), -1)
    print(f'Final output (flattened): {x_final.shape}')

print()
print('='*60)
print('Layer Details')
print('='*60)
print(f'Layer1 channels: 1 -> 64, stride=2')
print(f'Layer2 channels: 64 -> 128, stride=2')
print(f'Layer3 channels: 128 -> 64, stride=2')
print(f'Layer4 channels: 64 -> 64, stride=2')
print(f'AdaptiveAvgPool2d target: ({conf.features.time_max_pool_dim}, {int(conf.features.embedding_dim / (conf.features.time_max_pool_dim * 64))})')
print(f'Embedding dimension: {conf.features.embedding_dim}')
print()
print('='*60)
print('BasicBlock Structure')
print('='*60)
print('Each BasicBlock contains:')
print('  - conv1 (3x3, stride=1)')
print('  - bn1 + relu')
print('  - conv2 (3x3, stride=1)')
print('  - bn2 + relu')
print('  - conv3 (3x3, stride=1)')
print('  - bn3')
print('  - residual connection')
print('  - MaxPool2d(stride) at the end')
print()
print('Spatial dimension calculation:')
print('  Input: (17, 128)')
print('  After Layer1 (stride=2): (17/2, 128/2) = (8, 64)')
print('  After Layer2 (stride=2): (8/2, 64/2) = (4, 32)')
print('  After Layer3 (stride=2): (4/2, 32/2) = (2, 16)')
print('  After Layer4 (stride=2): (2/2, 16/2) = (1, 8)')
