#!/usr/bin/env python3
"""
Split multi-class CSV files into single-class CSV files.

This script enables the test dataloader to work with multi-class datasets by:
1. Splitting CSV files with multiple Q_* columns into separate single-class CSVs
2. Preserving the original audio filename in the Filename column
3. No audio file duplication needed - all CSVs reference the same audio files

Usage:
    python split_multiclass_csv.py --input_dir path/to/multiclass_data --output_dir path/to/output

Example:
    Input CSV (audio1.csv):
        Filename,Starttime,Endtime,Q_species1,Q_species2
        audio1.wav,0.5,1.2,POS,NEG
        audio1.wav,2.0,3.5,NEG,POS
    
    Output CSVs:
        audio1_species1.csv:
            Filename,Starttime,Endtime,Q_species1
            audio1.wav,0.5,1.2,POS
        
        audio1_species2.csv:
            Filename,Starttime,Endtime,Q_species2
            audio1.wav,2.0,3.5,POS
"""

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd

CLASS_NAME_SEPARATOR = "__"

def split_multiclass_csv(input_csv_path, output_dir, verbose=True):
    """
    Split a multi-class CSV into separate single-class CSV files.
    
    Args:
        input_csv_path: Path to input CSV file
        output_dir: Directory to save output CSV files
        verbose: Print progress messages
    
    Returns:
        List of created output filenames
    """
    df = pd.read_csv(input_csv_path)
    
    # Find all Q columns (class columns)
    q_columns = [col for col in df.columns if col.startswith('Q_')]
    
    if not q_columns:
        if verbose:
            print(f"  ⚠️  No Q_* columns found in {Path(input_csv_path).name}")
        return []
    
    # Base columns (Filename, Starttime, Endtime, etc.)
    base_columns = [col for col in df.columns if not col.startswith('Q_')]
    
    # Get base filename for output
    csv_basename = Path(input_csv_path).stem
    
    created_files = []
    
    # Split into separate CSV files
    for q_col in q_columns:
        # Extract class name (e.g., "Q_species1" -> "species1")
        class_name = q_col.replace('Q_', '')
        
        # Filter rows where this class has POS labels
        class_df = df[df[q_col] == 'POS'].copy()
        
        if len(class_df) > 0:
            # Create new dataframe with base columns + this Q column
            output_df = class_df[base_columns + [q_col]].copy()
            
            # Output filename: original_classname.csv
            output_filename = f"{csv_basename}_{class_name}.csv"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save
            output_df.to_csv(output_path, index=False)
            created_files.append(output_filename)
            
            if verbose:
                audio_file = output_df['Filename'].iloc[0]
                print(f"  ✓ {output_filename}: {len(output_df)} rows, audio: {audio_file}")
        else:
            if verbose:
                print(f"  ⚠️  No POS labels for {q_col} in {csv_basename}")
    
    return created_files

def split_multiclass_csv_auto(input_csv_path, output_dir, base_columns=None, verbose=True):
    """
    Split a multi-class CSV into separate single-class CSV files (auto-detect mode).
    
    This function automatically detects class columns by treating all columns
    that are not in the base_columns list as class columns. Useful for CSVs
    that don't follow the Q_* naming convention.
    
    Args:
        input_csv_path: Path to input CSV file
        output_dir: Directory to save output CSV files
        base_columns: List of base column names (e.g., ['Filename', 'Starttime', 'Endtime'])
                     If None, defaults to ['Filename', 'Starttime', 'Endtime']
        verbose: Print progress messages
    
    Returns:
        List of created output filenames
    
    Example:
        Input CSV (audio1.csv):
            Filename,Starttime,Endtime,species1,species2
            audio1.wav,0.5,1.2,POS,NEG
            audio1.wav,2.0,3.5,NEG,POS
        
        Output CSVs:
            audio1_species1.csv:
                Filename,Starttime,Endtime,species1
                audio1.wav,0.5,1.2,POS
            
            audio1_species2.csv:
                Filename,Starttime,Endtime,species2
                audio1.wav,2.0,3.5,POS
    """    
    df = pd.read_csv(input_csv_path)
    
    # Default base columns if not provided
    if base_columns is None:
        base_columns = ['Audiofilename', 'Starttime', 'Endtime']
    
    # Validate that base columns exist in the CSV
    missing_base = [col for col in base_columns if col not in df.columns]
    if missing_base:
        if verbose:
            print(f"  ⚠️  Missing base columns in {Path(input_csv_path).name}: {missing_base}")
        return []
    
    # Auto-detect class columns (all columns not in base_columns)
    class_columns = [col for col in df.columns if col not in base_columns]
    
    if not class_columns:
        if verbose:
            print(f"  ⚠️  No class columns found in {Path(input_csv_path).name}")
            print(f"      All columns are base columns: {list(df.columns)}")
        return []
    
    if verbose:
        print(f"  📋 Detected {len(class_columns)} class column(s): {class_columns}")
    
    # Get base filename for output
    csv_basename = Path(input_csv_path).stem
    
    created_files = []
    
    # Split into separate CSV files
    for class_col in class_columns:
        # Filter rows where this class has POS labels
        class_df = df[df[class_col] == 'POS'].copy()
        
        if len(class_df) > 0:
            # Create new dataframe with base columns + this class column
            output_df = class_df[base_columns + [class_col]].copy()
            
            # Output filename: original_classname.csv
            output_filename = f"{csv_basename}{CLASS_NAME_SEPARATOR}{class_col}.csv"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save
            output_df.to_csv(output_path, index=False)
            created_files.append(output_filename)
            
            if verbose:
                audio_file = output_df['Filename'].iloc[0] if 'Filename' in output_df.columns else 'N/A'
                print(f"  ✓ {output_filename}: {len(output_df)} rows, audio: {audio_file}")
        else:
            if verbose:
                print(f"  ⚠️  No POS labels for {class_col} in {csv_basename}")
    
    return created_files

def process_multiclass_dataset_auto(input_dir, output_dir, base_columns=None, 
                                    copy_audio=True, copy_features=True, 
                                    preserve_structure=True, create_symlinks=True, verbose=True):
    """Same as process_multiclass_dataset but uses auto-detect mode"""
    # Split CSVs
    csv_output_dir=output_dir
    for csv_file in Path(input_dir).rglob('*.csv'):
        if preserve_structure:
            rel_path = csv_file.parent.relative_to(input_dir)
            csv_output_dir = Path(output_dir) / rel_path
            csv_output_dir.mkdir(parents=True, exist_ok=True)
        
        split_multiclass_csv_auto(csv_file, csv_output_dir, base_columns, verbose)
    
    # Copy/symlink operations
    if copy_audio:
        copy_audio_files(input_dir, output_dir, preserve_structure, verbose)
    if copy_features:
        copy_feature_files(input_dir, output_dir, preserve_structure, verbose)
    if create_symlinks:
        create_symlinks_for_multiclass(input_dir, output_dir, preserve_structure, verbose)

def copy_audio_files(input_dir, output_dir, preserve_structure=True, verbose=True):
    """
    Copy audio files from input to output directory, preserving directory structure.
    
    Args:
        input_dir: Source directory containing audio files
        output_dir: Destination directory
        preserve_structure: Whether to preserve subdirectory structure
        verbose: Print progress messages
    
    Returns:
        Number of files copied
    """
    copied = 0
    input_path = Path(input_dir)
    
    # Find all .wav files recursively
    for wav_file in input_path.rglob('*.wav'):
        rel_path = wav_file.relative_to(input_path)
        
        if preserve_structure:
            # Preserve directory structure
            output_path = Path(output_dir) / rel_path
        else:
            # Flat structure
            output_path = Path(output_dir) / wav_file.name
        
        # Create parent directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not output_path.exists():
            shutil.copy2(wav_file, output_path)
            if verbose:
                display_path = rel_path if preserve_structure else wav_file.name
                print(f"  ✓ Copied: {display_path}")
            copied += 1
    
    return copied


def copy_feature_files(input_dir, output_dir, preserve_structure=True, verbose=True):
    """
    Copy pre-computed feature files (.h5, .npy) from input to output directory.
    
    Args:
        input_dir: Source directory containing feature files
        output_dir: Destination directory
        preserve_structure: Whether to preserve subdirectory structure
        verbose: Print progress messages
    
    Returns:
        Number of files copied
    """
    copied = 0
    input_path = Path(input_dir)
    
    for ext in ['*.h5', '*.npy']:
        for feature_file in input_path.rglob(ext):
            rel_path = feature_file.relative_to(input_path)
            
            if preserve_structure:
                # Preserve directory structure
                output_path = Path(output_dir) / rel_path
            else:
                # Flat structure
                output_path = Path(output_dir) / feature_file.name
            
            # Create parent directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if not output_path.exists():
                shutil.copy2(feature_file, output_path)
                if verbose:
                    display_path = rel_path if preserve_structure else feature_file.name
                    print(f"  ✓ Copied: {display_path}")
                copied += 1
    
    return copied


def create_symlinks_for_multiclass(original_audio_dir=None, output_dir=None, preserve_structure=True, verbose=True):
    """
    Create symlinks for audio and feature files to match split CSV filenames.
    
    This solves the evaluation system's expectation that CSV filename matches audio filename.
    For example: audio1_species1.csv expects audio1_species1.wav to exist.
    
    Args:
        output_dir: Directory containing split CSV files
        original_audio_dir: Original directory with audio files (if None, assumes audio already in output_dir)
        preserve_structure: Whether directory structure was preserved
        verbose: Print progress messages
    
    Returns:
        Number of symlinks created
    """
    feature_types = ["logmel"]

    output_path = Path(output_dir)
    print(f"Output Path: {output_path}")
    audio_path = Path(original_audio_dir) if original_audio_dir else output_path
    print(f"Audio Path: {audio_path}")

    symlinks_created = 0
    
    # Find all split CSV files (those with underscore pattern like audio1_species1.csv)
    for csv_file in output_path.rglob('*.csv'):
        csv_stem = csv_file.stem  # e.g., 'audio1_species1'
        
        # Check if this is a split CSV (contains underscore after base name)
        if CLASS_NAME_SEPARATOR not in csv_stem:
            continue
        
        # Extract base audio name (everything before last underscore)
        parts = csv_stem.rsplit(CLASS_NAME_SEPARATOR, 1)
        if len(parts) != 2:
            continue
            
        base_name = parts[0]  # e.g., 'audio1'
        
        # Determine source audio location
        if original_audio_dir:
            # Audio in original location
            if preserve_structure:
                rel_path = csv_file.parent.relative_to(output_path)
                source_parent = audio_path / rel_path
            else:
                source_parent = audio_path
        else:
            # Audio already copied to output_dir
            source_parent = csv_file.parent
        
        # Create symlinks for audio files
        for ext in ['.wav']:
            source_audio = source_parent / f"{base_name}{ext}"
            target_audio = csv_file.parent / f"{csv_stem}{ext}"

            if source_audio.exists() and not target_audio.exists():
                try:
                    if original_audio_dir:
                        os.symlink(source_audio, target_audio)
                    else:
                        os.symlink(source_audio.name, target_audio)
                    symlinks_created += 1
                    if verbose:
                        print(f"  ✓ Symlink: {target_audio.name} → {source_audio}")
                except Exception as e:
                    if verbose:
                        print(f"  ⚠️  Failed to create link for {target_audio.name}: {e}")
        
        # Create symlinks for feature files
        for ext in ['.h5']:
            for feature in feature_types:
                source_feature = source_parent / f"{base_name}_{feature}{ext}"
                target_feature = csv_file.parent / f"{csv_stem}_{feature}{ext}"
                print(f"Source_Feature exists: {source_feature}, {source_feature.exists()}")
                print(f"Target_Feature exists: {target_feature}, {target_feature.exists()}")

                # print(source_feature.exists() and not target_feature.exists())
                if source_feature.exists() and not target_feature.exists():
                    try:
                        if original_audio_dir:
                            os.symlink(source_feature, target_feature)
                        else:
                            os.symlink(source_feature.name, target_feature)
                        symlinks_created += 1
                        if verbose:
                            print(f"  ✓ Symlink: {target_feature.name} → {source_feature}")
                    except Exception as e:
                        if verbose:
                            print(f"  ⚠️  Failed to create link for {target_feature.name}: {e}")
        
    return symlinks_created


def process_multiclass_dataset(input_dir, output_dir, copy_audio=True, copy_features=True, preserve_structure=True, create_symlinks=True, verbose=True):
    """
    Complete preprocessing pipeline for multi-class datasets with nested directory supposrt.
    
    Args:
        input_dir: Directory containing multi-class CSV files and audio (supports nested structure)
        output_dir: Directory to save split CSV files
        copy_audio: Whether to copy audio files
        copy_features: Whether to copy feature files (.h5, .npy)
        preserve_structure: Whether to preserve subdirectory structure
        create_symlinks: Whether to create symlinks for split CSV files (recommended for evaluation)
        verbose: Print progress messages
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print("=" * 70)
        print("Multi-Class CSV Splitting Tool (Recursive)")
        print("=" * 70)
        print(f"Input directory:  {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Preserve structure: {preserve_structure}")
        print(f"Create symlinks: {create_symlinks}")
        print("=" * 70)
    
    # Step 1: Split CSV files (recursively)
    if verbose:
        print("\nStep 1: Splitting CSV files (recursive search)...")
        print("-" * 70)
    
    input_path = Path(input_dir)
    csv_files = list(input_path.rglob('*.csv'))  # Recursive search
    total_created = 0
    
    for csv_file in csv_files:
        # Determine output directory based on structure preservation
        if preserve_structure:
            rel_path = csv_file.parent.relative_to(input_path)
            csv_output_dir = Path(output_dir) / rel_path
            csv_output_dir.mkdir(parents=True, exist_ok=True)
        else:
            csv_output_dir = output_dir
        
        if verbose:
            rel_file = csv_file.relative_to(input_path)
            print(f"\nProcessing: {rel_file}")
        
        created = split_multiclass_csv(csv_file, csv_output_dir, verbose=verbose)
        total_created += len(created)
    
    if verbose:
        print(f"\n✓ Created {total_created} single-class CSV files from {len(csv_files)} input files")
    
    # Step 2: Copy audio files
    if copy_audio:
        if verbose:
            print("\n" + "=" * 70)
            print("Step 2: Copying audio files...")
            print("-" * 70)
        
        copied_audio = copy_audio_files(input_dir, output_dir, preserve_structure=preserve_structure, verbose=verbose)
        
        if verbose:
            print(f"\n✓ Copied {copied_audio} audio files")
    
    # Step 3: Copy feature files
    if copy_features:
        if verbose:
            print("\n" + "=" * 70)
            print("Step 3: Copying feature files...")
            print("-" * 70)
        
        copied_features = copy_feature_files(input_dir, output_dir, preserve_structure=preserve_structure, verbose=verbose)
        
        if verbose:
            print(f"\n✓ Copied {copied_features} feature files")
    
    # Step 4: Create symlinks for evaluation compatibility
    if create_symlinks:
        if verbose:
            print("\n" + "=" * 70)
            print("Step 4: Creating symlinks for evaluation compatibility...")
            print("-" * 70)
            print("(This ensures CSV filenames match expected audio filenames)")
        
        symlinks_created = create_symlinks_for_multiclass(original_audio_dir=None, output_dir=output_dir, preserve_structure=preserve_structure, verbose=verbose)
        
        if verbose:
            print(f"\n✓ Created {symlinks_created} symlinks/copies")
    
    if verbose:
        print("\n" + "=" * 70)
        print("✓ Processing complete!")
        print("=" * 70)
        print(f"\nYou can now use '{output_dir}' as your eval_dir in the config.")
        if create_symlinks:
            print("Symlinks have been created to ensure evaluation compatibility.")


def main():
    parser = argparse.ArgumentParser(
        description="Split multi-class CSV files for few-shot learning evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory containing multi-class CSV files and audio'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save split CSV files'
    )
    
    parser.add_argument(
        '--no-copy-audio',
        action='store_true',
        help='Skip copying audio files (if already in output_dir)'
    )
    
    parser.add_argument(
        '--no-copy-features',
        action='store_true',
        help='Skip copying feature files (if already in output_dir)'
    )
    
    parser.add_argument(
        '--no-preserve-structure',
        action='store_true',
        help='Flatten directory structure (put all files in output_dir root)'
    )
    
    parser.add_argument(
        '--no-symlinks',
        action='store_true',
        help='Skip creating symlinks for evaluation compatibility'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress messages'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return 1
    
    # Process the dataset
    process_multiclass_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        copy_audio=not args.no_copy_audio,
        copy_features=not args.no_copy_features,
        preserve_structure=not args.no_preserve_structure,
        create_symlinks=not args.no_symlinks,
        verbose=not args.quiet
    )
    
    return 0


if __name__ == '__main__':
    exit(main())
