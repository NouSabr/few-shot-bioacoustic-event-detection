#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate YAML configuration files."""

import sys
import io
from pathlib import Path

# Set UTF-8 encoding for stdout on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

try:
    import yaml
except ImportError:
    print("PyYAML not installed. Attempting basic validation...")
    yaml = None

def validate_yaml_basic(filepath):
    """Basic YAML validation without PyYAML."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Basic checks
        if not content.strip():
            return False, "File is empty"
        
        # Check for common YAML syntax issues
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            stripped = line.lstrip()
            if stripped and not stripped.startswith('#'):
                # Check for tabs (YAML doesn't allow tabs for indentation)
                if '\t' in line:
                    return False, f"Line {i}: Contains tab character (use spaces)"
                
                # Check for basic key-value format
                if ':' in stripped and not stripped.startswith('-'):
                    parts = stripped.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        if not key:
                            return False, f"Line {i}: Empty key before colon"
        
        return True, "Basic validation passed"
    except Exception as e:
        return False, str(e)

def validate_yaml_full(filepath):
    """Full YAML validation with PyYAML."""
    try:
        with open(filepath, 'r') as f:
            yaml.safe_load(f)
        return True, "Valid YAML"
    except yaml.YAMLError as e:
        return False, f"YAML Error: {e}"
    except Exception as e:
        return False, str(e)

def check_cife_parameters(filepath):
    """Check if CIFE parameters are present and valid."""
    try:
        with open(filepath, 'r') as f:
            if yaml:
                config = yaml.safe_load(f)
            else:
                # Basic check without parsing
                content = f.read()
                required_params = [
                    'use_cife',
                    'cife_channels',
                    'cife_d_model',
                    'cife_nhead',
                    'cife_num_encoder_layers',
                    'cife_num_decoder_layers',
                    'cife_dim_feedforward',
                    'cife_dropout',
                    'cife_alpha'
                ]
                missing = [p for p in required_params if p not in content]
                if missing:
                    return False, f"Missing CIFE parameters: {', '.join(missing)}"
                return True, "All CIFE parameters present"
        
        # If we have yaml, do deeper validation
        if 'features' not in config:
            return False, "No 'features' section found"
        
        features = config['features']
        required_params = {
            'use_cife': bool,
            'cife_channels': int,
            'cife_d_model': int,
            'cife_nhead': int,
            'cife_num_encoder_layers': int,
            'cife_num_decoder_layers': int,
            'cife_dim_feedforward': int,
            'cife_dropout': float,
            'cife_alpha': float
        }
        
        missing = []
        type_errors = []
        
        for param, expected_type in required_params.items():
            if param not in features:
                missing.append(param)
            elif not isinstance(features[param], expected_type):
                type_errors.append(f"{param} (expected {expected_type.__name__}, got {type(features[param]).__name__})")
        
        if missing:
            return False, f"Missing CIFE parameters: {', '.join(missing)}"
        if type_errors:
            return False, f"Type errors: {', '.join(type_errors)}"
        
        # Validate parameter values
        if features['cife_channels'] != 128:
            return False, f"cife_channels must be 128 (got {features['cife_channels']})"
        if features['cife_d_model'] != 128:
            return False, f"cife_d_model must be 128 (got {features['cife_d_model']})"
        if features['cife_d_model'] % features['cife_nhead'] != 0:
            return False, f"cife_nhead ({features['cife_nhead']}) must divide cife_d_model ({features['cife_d_model']})"
        if not 0.0 <= features['cife_alpha'] <= 1.0:
            return False, f"cife_alpha must be between 0.0 and 1.0 (got {features['cife_alpha']})"
        if not 0.0 <= features['cife_dropout'] <= 1.0:
            return False, f"cife_dropout must be between 0.0 and 1.0 (got {features['cife_dropout']})"
        
        return True, "All CIFE parameters valid"
    except Exception as e:
        return False, str(e)

def main():
    """Main validation function."""
    config_dir = Path(__file__).parent
    files_to_validate = ['train.yaml', 'test.yaml']
    
    print("=" * 60)
    print("YAML Configuration Validation")
    print("=" * 60)
    print()
    
    all_valid = True
    
    for filename in files_to_validate:
        filepath = config_dir / filename
        print(f"Validating: {filename}")
        print("-" * 60)
        
        if not filepath.exists():
            print(f"  ✗ File not found: {filepath}")
            all_valid = False
            print()
            continue
        
        # YAML syntax validation
        if yaml:
            valid, message = validate_yaml_full(filepath)
        else:
            valid, message = validate_yaml_basic(filepath)
        
        if valid:
            print(f"  ✓ YAML Syntax: {message}")
        else:
            print(f"  ✗ YAML Syntax: {message}")
            all_valid = False
            print()
            continue
        
        # CIFE parameters validation
        valid, message = check_cife_parameters(filepath)
        if valid:
            print(f"  ✓ CIFE Parameters: {message}")
        else:
            print(f"  ✗ CIFE Parameters: {message}")
            all_valid = False
        
        print()
    
    print("=" * 60)
    if all_valid:
        print("✓ All configuration files are valid!")
        return 0
    else:
        print("✗ Some configuration files have errors")
        return 1

if __name__ == '__main__':
    sys.exit(main())
