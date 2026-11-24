"""
Comprehensive Test Suite for CIFE Integration

This test suite verifies that the CIFE (Cross-scale Interactive Feature Enhancement)
integration maintains correct shapes, gradient flow, and functionality in both
enabled and disabled modes.

Tests cover:
1. Shape preservation across CIFE enabled/disabled modes
2. Gradient flow through CIFE module
3. CIFE impact verification (non-zero modification)
4. Variable input size handling
5. Configuration parameter application
6. Error handling and fallback mechanisms

Date: 2026-04-22
"""

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    # Create dummy pytest.fixture decorator for standalone execution
    class pytest:
        @staticmethod
        def fixture(func):
            return func

import torch
import torch.nn as nn
from omegaconf import OmegaConf
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.components.protonet import ResNet
from models.components.audio_cife import AudioCIFE


# ==================== FIXTURES ====================

@pytest.fixture
def base_config():
    """
    Base configuration for testing without CIFE.
    
    Returns:
        OmegaConf configuration object with CIFE disabled
    """
    return OmegaConf.create({
        'embedding_dim': 2048,
        'drop_rate': 0.0,  # Disable dropout for deterministic tests
        'with_bias': False,
        'non_linearity': 'leaky_relu',
        'time_max_pool_dim': 4,
        'layer_4': False,
        'use_cife': False,  # CIFE disabled
    })


@pytest.fixture
def cife_config(base_config):
    """
    Configuration with CIFE enabled.
    
    Returns:
        OmegaConf configuration object with CIFE enabled and configured
    """
    config = base_config.copy()
    config.use_cife = True
    config.cife_channels = 128
    config.cife_d_model = 128
    config.cife_nhead = 8
    config.cife_num_encoder_layers = 2
    config.cife_num_decoder_layers = 2
    config.cife_dim_feedforward = 256
    config.cife_dropout = 0.0  # Disable dropout for deterministic tests
    config.cife_alpha = 0.3
    return config


@pytest.fixture
def sample_input():
    """
    Create a sample input tensor for testing.
    
    Returns:
        Tensor of shape (batch_size, seq_len, mel_bins)
    """
    batch_size = 4
    seq_len = 17
    mel_bins = 128
    return torch.randn(batch_size, seq_len, mel_bins)


@pytest.fixture
def device():
    """
    Get the appropriate device for testing.
    
    Returns:
        torch.device: CUDA if available, else CPU
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==================== TEST FUNCTIONS ====================

def test_shape_verification(base_config, cife_config, sample_input):
    """
    Verify that CIFE maintains exact output shapes.
    
    This test ensures that enabling CIFE does not change the output shape
    of the ResNet model, maintaining backward compatibility.
    
    Test Steps:
    1. Create baseline model (CIFE disabled)
    2. Create CIFE-enabled model
    3. Forward pass with same input
    4. Verify output shapes are identical
    5. Verify expected output shape
    """
    print("\n" + "="*60)
    print("TEST: Shape Verification")
    print("="*60)
    
    # Create models
    model_baseline = ResNet(base_config)
    model_cife = ResNet(cife_config)
    
    # Set to eval mode for deterministic behavior
    model_baseline.eval()
    model_cife.eval()
    
    batch_size = sample_input.size(0)
    
    # Forward pass
    with torch.no_grad():
        output_baseline = model_baseline(sample_input)
        output_cife = model_cife(sample_input)
    
    print(f"Input shape: {sample_input.shape}")
    print(f"Baseline output shape: {output_baseline.shape}")
    print(f"CIFE output shape: {output_cife.shape}")
    
    # Verify shapes match
    assert output_baseline.shape == output_cife.shape, \
        f"Shape mismatch: baseline={output_baseline.shape}, cife={output_cife.shape}"
    
    # Verify expected shape
    expected_shape = (batch_size, 2048)
    assert output_baseline.shape == expected_shape, \
        f"Expected {expected_shape}, got {output_baseline.shape}"
    assert output_cife.shape == expected_shape, \
        f"Expected {expected_shape}, got {output_cife.shape}"
    
    print("✓ Shape verification passed")
    print(f"  - Both models output shape: {expected_shape}")


def test_gradient_flow(cife_config, sample_input):
    """
    Verify that gradients flow through the CIFE module correctly.
    
    This test ensures that the CIFE module is properly integrated into
    the computational graph and receives gradients during backpropagation.
    
    Test Steps:
    1. Create CIFE-enabled model
    2. Forward pass
    3. Compute loss and backward pass
    4. Check CIFE module has gradients
    5. Check backbone layers have gradients
    """
    print("\n" + "="*60)
    print("TEST: Gradient Flow")
    print("="*60)
    
    # Create model with CIFE
    model = ResNet(cife_config)
    model.train()  # Enable training mode
    
    # Forward pass
    output = model(sample_input)
    
    # Create dummy target and compute loss
    target = torch.randn_like(output)
    loss = nn.MSELoss()(output, target)
    
    print(f"Loss: {loss.item():.6f}")
    
    # Backward pass
    loss.backward()
    
    # Check CIFE module has gradients
    cife_has_gradients = False
    cife_grad_count = 0
    for name, param in model.cife_module.named_parameters():
        if param.grad is not None:
            cife_has_gradients = True
            cife_grad_count += 1
            grad_norm = param.grad.norm().item()
            print(f"  CIFE {name}: grad_norm={grad_norm:.6f}")
    
    assert cife_has_gradients, "CIFE module has no gradients!"
    print(f"✓ CIFE module has gradients ({cife_grad_count} parameters)")
    
    # Check backbone has gradients
    backbone_has_gradients = False
    backbone_grad_count = 0
    for name, param in model.named_parameters():
        if 'cife_module' not in name and param.grad is not None:
            backbone_has_gradients = True
            backbone_grad_count += 1
    
    assert backbone_has_gradients, "Backbone has no gradients!"
    print(f"✓ Backbone has gradients ({backbone_grad_count} parameters)")
    
    print("✓ Gradient flow verification passed")


def test_cife_impact(base_config, cife_config, sample_input):
    """
    Verify that CIFE actually modifies features (not a no-op).
    
    This test ensures that CIFE is actively processing features and
    producing different outputs compared to the baseline model.
    
    Test Steps:
    1. Create baseline and CIFE models
    2. Forward pass with same input
    3. Measure output difference
    4. Verify difference is non-zero but controlled
    """
    print("\n" + "="*60)
    print("TEST: CIFE Impact Verification")
    print("="*60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create models
    model_baseline = ResNet(base_config)
    
    torch.manual_seed(42)  # Reset seed for same initialization
    model_cife = ResNet(cife_config)
    
    # Copy baseline weights to CIFE model (except CIFE module)
    # This ensures differences are due to CIFE, not random initialization
    baseline_state = model_baseline.state_dict()
    cife_state = model_cife.state_dict()
    
    for key in baseline_state.keys():
        if key in cife_state and 'cife_module' not in key:
            cife_state[key] = baseline_state[key].clone()
    
    model_cife.load_state_dict(cife_state, strict=False)
    
    # Set to eval mode
    model_baseline.eval()
    model_cife.eval()
    
    # Forward pass
    with torch.no_grad():
        output_baseline = model_baseline(sample_input)
        output_cife = model_cife(sample_input)
    
    # Measure difference
    abs_diff = torch.abs(output_cife - output_baseline)
    mean_abs_diff = abs_diff.mean().item()
    max_abs_diff = abs_diff.max().item()
    
    # Relative difference
    rel_diff = abs_diff / (torch.abs(output_baseline) + 1e-8)
    mean_rel_diff = rel_diff.mean().item()
    
    print(f"Mean absolute difference: {mean_abs_diff:.6f}")
    print(f"Max absolute difference: {max_abs_diff:.6f}")
    print(f"Mean relative difference: {mean_rel_diff:.4%}")
    
    # Verify CIFE has an impact (non-zero difference)
    assert mean_abs_diff > 1e-6, \
        f"CIFE appears to be a no-op! Mean difference: {mean_abs_diff}"
    
    # Verify impact is controlled (not too large)
    # With alpha=0.3, we expect moderate changes
    assert mean_rel_diff < 1.0, \
        f"CIFE impact too large! Mean relative difference: {mean_rel_diff:.4%}"
    
    print("✓ CIFE impact verification passed")
    print(f"  - CIFE modifies features with controlled impact")


def test_variable_input_sizes(cife_config):
    """
    Test CIFE with different input dimensions.
    
    This test verifies that the model handles various input sizes correctly,
    which is important for real-world audio data with varying lengths.
    
    Test Steps:
    1. Create CIFE-enabled model
    2. Test with multiple input sizes
    3. Verify all produce correct output shapes
    """
    print("\n" + "="*60)
    print("TEST: Variable Input Sizes")
    print("="*60)
    
    # Create model
    model = ResNet(cife_config)
    model.eval()
    
    # Test different input sizes
    test_sizes = [
        (4, 17, 128),   # Standard size
        (2, 15, 128),   # Smaller batch, shorter sequence
        (8, 20, 128),   # Larger batch, longer sequence
        (1, 10, 128),   # Single sample, short sequence
        (6, 25, 128),   # Medium batch, long sequence
    ]
    
    expected_output_dim = 2048
    
    for batch_size, seq_len, mel_bins in test_sizes:
        # Create input
        x = torch.randn(batch_size, seq_len, mel_bins)
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        # Verify output shape
        expected_shape = (batch_size, expected_output_dim)
        assert output.shape == expected_shape, \
            f"Input {x.shape} -> Expected output {expected_shape}, got {output.shape}"
        
        print(f"✓ Input {x.shape} -> Output {output.shape}")
    
    print("✓ Variable input size test passed")


def test_configuration_parameters(base_config):
    """
    Test different CIFE configurations.
    
    This test verifies that different CIFE configuration parameters
    are properly applied and the model works correctly with various settings.
    
    Test Steps:
    1. Test different nhead values
    2. Test different dropout values
    3. Test different alpha values
    4. Verify all configurations work correctly
    """
    print("\n" + "="*60)
    print("TEST: Configuration Parameters")
    print("="*60)
    
    batch_size = 4
    x = torch.randn(batch_size, 17, 128)
    
    # Test configurations
    test_configs = [
        {'nhead': 4, 'dropout': 0.0, 'alpha': 0.1, 'desc': 'Low alpha, 4 heads'},
        {'nhead': 8, 'dropout': 0.1, 'alpha': 0.3, 'desc': 'Standard config'},
        {'nhead': 16, 'dropout': 0.2, 'alpha': 0.5, 'desc': 'High alpha, 16 heads'},
        {'nhead': 8, 'dropout': 0.0, 'alpha': 0.7, 'desc': 'Very high alpha'},
    ]
    
    for config_params in test_configs:
        # Create config
        config = base_config.copy()
        config.use_cife = True
        config.cife_channels = 128
        config.cife_d_model = 128
        config.cife_nhead = config_params['nhead']
        config.cife_num_encoder_layers = 2
        config.cife_num_decoder_layers = 2
        config.cife_dim_feedforward = 256
        config.cife_dropout = config_params['dropout']
        config.cife_alpha = config_params['alpha']
        
        # Create model
        model = ResNet(config)
        model.eval()
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        # Verify output shape
        expected_shape = (batch_size, 2048)
        assert output.shape == expected_shape, \
            f"Config {config_params['desc']}: Expected {expected_shape}, got {output.shape}"
        
        # Verify alpha is applied
        assert abs(model.cife_alpha - config_params['alpha']) < 1e-6, \
            f"Alpha not applied correctly: expected {config_params['alpha']}, got {model.cife_alpha}"
        
        # Verify nhead is applied
        assert model.cife_module.nhead == config_params['nhead'], \
            f"nhead not applied correctly: expected {config_params['nhead']}, got {model.cife_module.nhead}"
        
        print(f"✓ Config: {config_params['desc']}")
        print(f"    nhead={config_params['nhead']}, dropout={config_params['dropout']}, alpha={config_params['alpha']}")
    
    print("✓ Configuration parameter test passed")


def test_error_handling(cife_config, sample_input):
    """
    Test error handling and fallback mechanisms.
    
    This test verifies that the model gracefully handles failures in the
    CIFE module and falls back to the original processing path.
    
    Test Steps:
    1. Create CIFE-enabled model
    2. Simulate CIFE failure by corrupting module
    3. Verify fallback to original path works
    4. Verify output is still produced
    """
    print("\n" + "="*60)
    print("TEST: Error Handling and Fallback")
    print("="*60)
    
    # Create model
    model = ResNet(cife_config)
    model.eval()
    
    # Test 1: Normal operation
    print("\n1. Testing normal operation...")
    with torch.no_grad():
        output_normal = model(sample_input)
    
    assert output_normal.shape == (sample_input.size(0), 2048), \
        "Normal operation failed"
    print("✓ Normal operation successful")
    
    # Test 2: Simulate CIFE failure by breaking the module
    print("\n2. Testing fallback mechanism...")
    
    # Save original forward method
    original_forward = model.cife_module.forward
    
    # Replace with a method that raises an exception
    def failing_forward(*args, **kwargs):
        raise RuntimeError("Simulated CIFE failure")
    
    model.cife_module.forward = failing_forward
    
    # Forward pass should still work due to try-except in ResNet
    with torch.no_grad():
        output_fallback = model(sample_input)
    
    assert output_fallback.shape == (sample_input.size(0), 2048), \
        "Fallback mechanism failed"
    print("✓ Fallback mechanism works correctly")
    
    # Restore original forward method
    model.cife_module.forward = original_forward
    
    # Test 3: Verify model still works after restoration
    print("\n3. Testing recovery after failure...")
    with torch.no_grad():
        output_recovered = model(sample_input)
    
    assert output_recovered.shape == (sample_input.size(0), 2048), \
        "Recovery after failure failed"
    print("✓ Model recovered successfully")
    
    print("\n✓ Error handling test passed")


def test_cife_module_standalone():
    """
    Test AudioCIFE module in isolation.
    
    This test verifies that the AudioCIFE module works correctly
    when used standalone, independent of the ResNet integration.
    
    Test Steps:
    1. Create AudioCIFE module
    2. Test with sample inputs
    3. Verify output shapes
    4. Verify gradient flow
    """
    print("\n" + "="*60)
    print("TEST: AudioCIFE Module Standalone")
    print("="*60)
    
    # Create module
    cife = AudioCIFE(
        channels=128,
        d_model=128,
        nhead=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=256,
        dropout=0.0
    )
    
    # Create sample inputs (Layer2 output shape)
    batch_size = 4
    source = torch.randn(batch_size, 128, 4, 32)
    target = torch.randn(batch_size, 128, 4, 32)
    
    print(f"Source shape: {source.shape}")
    print(f"Target shape: {target.shape}")
    
    # Test 1: Forward pass
    cife.eval()
    with torch.no_grad():
        output = cife(source, target)
    
    assert output.shape == target.shape, \
        f"Output shape mismatch: expected {target.shape}, got {output.shape}"
    print(f"✓ Output shape correct: {output.shape}")
    
    # Test 2: Gradient flow
    cife.train()
    source.requires_grad = True
    target.requires_grad = True
    
    output = cife(source, target)
    loss = output.sum()
    loss.backward()
    
    assert source.grad is not None, "Source gradients not computed"
    assert target.grad is not None, "Target gradients not computed"
    print("✓ Gradients flow through module")
    
    # Test 3: Parameter count
    total_params = sum(p.numel() for p in cife.parameters())
    trainable_params = sum(p.numel() for p in cife.parameters() if p.requires_grad)
    
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    print("\n✓ AudioCIFE standalone test passed")


def test_memory_efficiency(cife_config, sample_input):
    """
    Test memory efficiency of CIFE integration.
    
    This test verifies that CIFE doesn't cause excessive memory usage
    and that memory is properly released after forward/backward passes.
    
    Test Steps:
    1. Measure baseline memory usage
    2. Measure CIFE memory usage
    3. Verify memory is released after operations
    """
    print("\n" + "="*60)
    print("TEST: Memory Efficiency")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("⚠ CUDA not available, skipping memory test")
        return
    
    device = torch.device('cuda')
    
    # Create model and move to GPU
    model = ResNet(cife_config).to(device)
    sample_input_gpu = sample_input.to(device)
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Measure memory before forward pass
    mem_before = torch.cuda.memory_allocated() / 1024**2  # MB
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(sample_input_gpu)
    
    # Measure memory after forward pass
    mem_after = torch.cuda.memory_allocated() / 1024**2  # MB
    mem_peak = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    print(f"Memory before: {mem_before:.2f} MB")
    print(f"Memory after: {mem_after:.2f} MB")
    print(f"Peak memory: {mem_peak:.2f} MB")
    print(f"Memory increase: {mem_after - mem_before:.2f} MB")
    
    # Verify memory is reasonable (less than 500MB for this model)
    assert mem_peak < 500, f"Memory usage too high: {mem_peak:.2f} MB"
    
    print("✓ Memory efficiency test passed")


# ==================== MAIN TEST RUNNER ====================

if __name__ == "__main__":
    """
    Run all tests when script is executed directly.
    
    This allows running tests without pytest for quick verification.
    """
    print("\n" + "="*70)
    print(" "*15 + "CIFE INTEGRATION TEST SUITE")
    print("="*70)
    
    # Create fixtures
    base_cfg = OmegaConf.create({
        'embedding_dim': 2048,
        'drop_rate': 0.0,
        'with_bias': False,
        'non_linearity': 'leaky_relu',
        'time_max_pool_dim': 4,
        'layer_4': False,
        'use_cife': False,
    })
    
    cife_cfg = base_cfg.copy()
    cife_cfg.use_cife = True
    cife_cfg.cife_channels = 128
    cife_cfg.cife_d_model = 128
    cife_cfg.cife_nhead = 8
    cife_cfg.cife_num_encoder_layers = 2
    cife_cfg.cife_num_decoder_layers = 2
    cife_cfg.cife_dim_feedforward = 256
    cife_cfg.cife_dropout = 0.0
    cife_cfg.cife_alpha = 0.3
    
    sample_inp = torch.randn(4, 17, 128)
    
    # Run tests
    try:
        test_shape_verification(base_cfg, cife_cfg, sample_inp)
        test_gradient_flow(cife_cfg, sample_inp)
        test_cife_impact(base_cfg, cife_cfg, sample_inp)
        test_variable_input_sizes(cife_cfg)
        test_configuration_parameters(base_cfg)
        test_error_handling(cife_cfg, sample_inp)
        test_cife_module_standalone()
        test_memory_efficiency(cife_cfg, sample_inp)
        
        print("\n" + "="*70)
        print(" "*20 + "ALL TESTS PASSED! ✓")
        print("="*70)
        
    except AssertionError as e:
        print("\n" + "="*70)
        print(" "*20 + "TEST FAILED! ✗")
        print("="*70)
        print(f"\nError: {e}")
        raise
