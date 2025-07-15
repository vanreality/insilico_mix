#!/usr/bin/env python3
"""
BAM to Parquet Converter with ML Model Probability Simulation

This script processes target and background BAM files, extracts read information,
simulates machine learning model probabilities, and outputs the data as parquet files.
"""

import click
import pandas as pd
import pysam
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Any
import sys
import os
from rich.console import Console
from rich.progress import Progress, TaskID, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich import print as rprint
from rich.panel import Panel

console = Console()


def validate_bam_file(bam_path: str) -> bool:
    """
    Validate that a BAM file exists and is readable.
    
    Args:
        bam_path: Path to the BAM file
        
    Returns:
        True if the BAM file is valid, False otherwise
        
    Raises:
        FileNotFoundError: If the BAM file doesn't exist
        OSError: If the BAM file cannot be read
    """
    try:
        if not os.path.exists(bam_path):
            raise FileNotFoundError(f"BAM file not found: {bam_path}")
        
        # Try to open and read the header to validate the file
        with pysam.AlignmentFile(bam_path, "rb") as bam_file:
            _ = bam_file.header
        return True
    except Exception as e:
        console.print(f"[red]Error validating BAM file {bam_path}: {e}[/red]")
        return False


def extract_reads_from_bam(bam_path: str, progress: Progress, task_id: TaskID) -> List[Dict[str, Any]]:
    """
    Extract read information from a BAM file.
    
    Args:
        bam_path: Path to the BAM file
        progress: Rich progress bar instance
        task_id: Task ID for progress tracking
        
    Returns:
        List of dictionaries containing read information
        
    Raises:
        IOError: If the BAM file cannot be read
        ValueError: If the BAM file is corrupted or invalid
    """
    reads_data = []
    
    try:
        with pysam.AlignmentFile(bam_path, "rb") as bam_file:
            # Count total reads first for progress tracking
            total_reads = bam_file.count()
            progress.update(task_id, total=total_reads)
            
            # Reset to beginning of file
            bam_file.seek(0)
            
            for read_count, read in enumerate(bam_file.fetch()):
                # Skip unmapped reads
                if read.is_unmapped:
                    continue
                
                read_data = {
                    'chr': read.reference_name,
                    'start': read.reference_start,
                    'end': read.reference_end,
                    'text': read.query_sequence if read.query_sequence else "",
                    'name': read.query_name,
                }
                reads_data.append(read_data)
                
                # Update progress every 1000 reads
                if read_count % 1000 == 0:
                    progress.update(task_id, advance=1000)
            
            # Final progress update
            progress.update(task_id, completed=total_reads)
            
    except Exception as e:
        console.print(f"[red]Error reading BAM file {bam_path}: {e}[/red]")
        raise
    
    return reads_data


def generate_probabilities(n_target: int, n_background: int, model_acc: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate simulated ML model probabilities for target and background reads.
    
    The probabilities are generated with bimodal distributions and flat regions:
    - Target reads: peaks around 0.95 (correct) and 0.05 (misclassified), flat elsewhere
    - Background reads: peaks around 0.05 (correct) and 0.95 (misclassified), flat elsewhere
    - Classification threshold: 0.5
    - Overall accuracy matches model_acc parameter
    
    Args:
        n_target: Number of target reads
        n_background: Number of background reads
        model_acc: Model accuracy (between 0 and 1)
        
    Returns:
        Tuple of (target_probabilities, background_probabilities)
        
    Raises:
        ValueError: If model_acc is not between 0 and 1
    """
    if not 0 <= model_acc <= 1:
        raise ValueError(f"Model accuracy must be between 0 and 1, got {model_acc}")
    
    # Calculate how many reads should be correctly classified
    total_reads = n_target + n_background
    correct_predictions = int(model_acc * total_reads)
    
    # Distribute correct predictions proportionally between target and background
    # This ensures a more balanced approach to meeting accuracy requirements
    if total_reads == 0:
        return np.array([]), np.array([])
    
    target_ratio = n_target / total_reads
    background_ratio = n_background / total_reads
    
    # Calculate correct classifications for each type
    target_correct = min(n_target, int(correct_predictions * target_ratio + 0.5))
    background_correct = min(n_background, correct_predictions - target_correct)
    
    # Adjust if we haven't allocated enough correct predictions
    remaining_correct = correct_predictions - target_correct - background_correct
    if remaining_correct > 0:
        if target_correct < n_target:
            additional_target = min(remaining_correct, n_target - target_correct)
            target_correct += additional_target
            remaining_correct -= additional_target
        if remaining_correct > 0 and background_correct < n_background:
            background_correct += min(remaining_correct, n_background - background_correct)
    
    target_misclassified = n_target - target_correct
    background_misclassified = n_background - background_correct
    
    def generate_bimodal_probs(n_samples: int, peak_value: float, is_correct: bool) -> np.ndarray:
        """
        Generate bimodal distribution with one peak and flat background.
        
        Args:
            n_samples: Number of samples to generate
            peak_value: Center of the peak (0.05 or 0.95)
            is_correct: True if this represents correct classification
            
        Returns:
            Array of probabilities
        """
        if n_samples == 0:
            return np.array([])
        
        # Proportion of samples that form the peak vs flat distribution
        peak_proportion = 0.7  # 70% form the peak, 30% are flat
        n_peak = int(n_samples * peak_proportion)
        n_flat = n_samples - n_peak
        
        probs = np.zeros(n_samples)
        
        # Generate peaked distribution using normal distribution with small std
        if n_peak > 0:
            if peak_value <= 0.5:
                # Lower peak (around 0.05)
                peak_probs = np.random.normal(peak_value, 0.02, n_peak)
                if is_correct:
                    # For correct classification, must be ≤ 0.5
                    probs[:n_peak] = np.clip(peak_probs, 0.0, 0.5)
                else:
                    # For misclassification, must be ≤ 0.5 but peaked around 0.05
                    probs[:n_peak] = np.clip(peak_probs, 0.0, 0.5)
            else:
                # Higher peak (around 0.95)
                peak_probs = np.random.normal(peak_value, 0.02, n_peak)
                if is_correct:
                    # For correct classification, must be > 0.5
                    probs[:n_peak] = np.clip(peak_probs, 0.501, 1.0)
                else:
                    # For misclassification, must be > 0.5 but peaked around 0.95
                    probs[:n_peak] = np.clip(peak_probs, 0.501, 1.0)
        
        # Generate flat distribution in the appropriate range
        if n_flat > 0:
            if is_correct:
                if peak_value <= 0.5:
                    # Correct classification for low peak: flat in [0.0, 0.5]
                    probs[n_peak:] = np.random.uniform(0.0, 0.5, n_flat)
                else:
                    # Correct classification for high peak: flat in (0.5, 1.0]
                    probs[n_peak:] = np.random.uniform(0.501, 1.0, n_flat)
            else:
                if peak_value <= 0.5:
                    # Misclassification for low peak: flat in [0.0, 0.5]
                    probs[n_peak:] = np.random.uniform(0.0, 0.5, n_flat)
                else:
                    # Misclassification for high peak: flat in (0.5, 1.0]
                    probs[n_peak:] = np.random.uniform(0.501, 1.0, n_flat)
        
        return probs
    
    # Generate probabilities for target reads
    target_probs = np.zeros(n_target)
    
    # Correctly classified targets (prob > 0.5) - peak around 0.95
    if target_correct > 0:
        target_probs[:target_correct] = generate_bimodal_probs(target_correct, 0.95, True)
    
    # Misclassified targets (prob ≤ 0.5) - peak around 0.05
    if target_misclassified > 0:
        target_probs[target_correct:] = generate_bimodal_probs(target_misclassified, 0.05, False)
    
    # Generate probabilities for background reads
    background_probs = np.zeros(n_background)
    
    # Correctly classified background (prob ≤ 0.5) - peak around 0.05
    if background_correct > 0:
        background_probs[:background_correct] = generate_bimodal_probs(background_correct, 0.05, True)
    
    # Misclassified background (prob > 0.5) - peak around 0.95
    if background_misclassified > 0:
        background_probs[background_correct:] = generate_bimodal_probs(background_misclassified, 0.95, False)
    
    # Shuffle to randomize order
    np.random.shuffle(target_probs)
    np.random.shuffle(background_probs)
    
    return target_probs, background_probs


def create_dataframe(reads_data: List[Dict[str, Any]], probabilities: np.ndarray) -> pd.DataFrame:
    """
    Create a pandas DataFrame from reads data and probabilities.
    
    Args:
        reads_data: List of dictionaries containing read information
        probabilities: Array of probabilities for each read
        
    Returns:
        DataFrame with columns ['chr', 'start', 'end', 'text', 'name', 'prob_class_1']
        
    Raises:
        ValueError: If the lengths of reads_data and probabilities don't match
    """
    if len(reads_data) != len(probabilities):
        raise ValueError(f"Mismatch between reads data ({len(reads_data)}) and probabilities ({len(probabilities)})")
    
    # Create DataFrame from reads data
    df = pd.DataFrame(reads_data)
    
    # Add probabilities
    df['prob_class_1'] = probabilities
    
    # Ensure correct column order
    df = df[['chr', 'start', 'end', 'text', 'name', 'prob_class_1']]
    
    return df


def save_to_parquet(df: pd.DataFrame, output_path: str) -> None:
    """
    Save DataFrame to parquet file.
    
    Args:
        df: DataFrame to save
        output_path: Path for the output parquet file
        
    Raises:
        IOError: If the file cannot be written
    """
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Save to parquet with compression
        df.to_parquet(output_path, compression='snappy', index=False)
        console.print(f"[green]Successfully saved {len(df)} reads to {output_path}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error saving to parquet file {output_path}: {e}[/red]")
        raise


def display_summary(target_df: pd.DataFrame, background_df: pd.DataFrame, model_acc: float) -> None:
    """
    Display a summary table of the processing results.
    
    Args:
        target_df: DataFrame containing target reads
        background_df: DataFrame containing background reads
        model_acc: Model accuracy used for simulation
    """
    # Calculate actual accuracy
    target_correct = (target_df['prob_class_1'] > 0.5).sum()
    background_correct = (background_df['prob_class_1'] <= 0.5).sum()
    total_reads = len(target_df) + len(background_df)
    actual_acc = (target_correct + background_correct) / total_reads if total_reads > 0 else 0
    
    # Create summary table
    table = Table(title="Processing Summary")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    
    table.add_row("Target reads", f"{len(target_df):,}")
    table.add_row("Background reads", f"{len(background_df):,}")
    table.add_row("Total reads", f"{total_reads:,}")
    table.add_row("Expected accuracy", f"{model_acc:.3f}")
    table.add_row("Actual accuracy", f"{actual_acc:.3f}")
    table.add_row("Target prob range", f"{target_df['prob_class_1'].min():.3f} - {target_df['prob_class_1'].max():.3f}")
    table.add_row("Background prob range", f"{background_df['prob_class_1'].min():.3f} - {background_df['prob_class_1'].max():.3f}")
    
    console.print(table)


@click.command()
@click.option('--target_bam', required=True, type=click.Path(exists=True), 
              help='Path to target BAM file')
@click.option('--background_bam', required=True, type=click.Path(exists=True), 
              help='Path to background BAM file')
@click.option('--model_acc', required=True, type=float, 
              help='Model accuracy (between 0 and 1)')
@click.option('--output_dir', default='.', type=click.Path(), 
              help='Output directory for parquet files (default: current directory)')
def main(target_bam: str, background_bam: str, model_acc: float, output_dir: str):
    """
    Convert BAM files to parquet format with simulated ML model probabilities.
    
    This script processes target and background BAM files, extracts read information,
    simulates machine learning model probabilities based on the specified accuracy,
    and outputs separate parquet files for target and background reads.
    
    Args:
        target_bam: Path to the target BAM file
        background_bam: Path to the background BAM file  
        model_acc: Model accuracy for probability simulation (0-1)
        output_dir: Output directory for parquet files
    """
    console.print(Panel.fit("[bold blue]BAM to Parquet Converter[/bold blue]"))
    
    try:
        # Validate model accuracy
        if not 0 <= model_acc <= 1:
            console.print("[red]Error: Model accuracy must be between 0 and 1[/red]")
            sys.exit(1)
        
        # Validate BAM files
        console.print("[yellow]Validating BAM files...[/yellow]")
        if not validate_bam_file(target_bam):
            sys.exit(1)
        if not validate_bam_file(background_bam):
            sys.exit(1)
        
        # Extract prefixes for output files
        target_prefix = Path(target_bam).stem.replace('.bam', '')
        background_prefix = Path(background_bam).stem.replace('.bam', '')
        
        target_output = os.path.join(output_dir, f"{target_prefix}.parquet")
        background_output = os.path.join(output_dir, f"{background_prefix}.parquet")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            # Extract reads from target BAM
            target_task = progress.add_task("Processing target BAM...", total=None)
            target_reads = extract_reads_from_bam(target_bam, progress, target_task)
            
            # Extract reads from background BAM
            background_task = progress.add_task("Processing background BAM...", total=None)
            background_reads = extract_reads_from_bam(background_bam, progress, background_task)
            
            # Generate probabilities
            prob_task = progress.add_task("Generating probabilities...", total=100)
            target_probs, background_probs = generate_probabilities(
                len(target_reads), len(background_reads), model_acc
            )
            progress.update(prob_task, completed=100)
            
            # Create DataFrames
            df_task = progress.add_task("Creating DataFrames...", total=2)
            target_df = create_dataframe(target_reads, target_probs)
            progress.update(df_task, advance=1)
            
            background_df = create_dataframe(background_reads, background_probs)
            progress.update(df_task, advance=1)
            
            # Save to parquet files
            save_task = progress.add_task("Saving parquet files...", total=2)
            save_to_parquet(target_df, target_output)
            progress.update(save_task, advance=1)
            
            save_to_parquet(background_df, background_output)
            progress.update(save_task, advance=1)
        
        # Display summary
        console.print("\n")
        display_summary(target_df, background_df, model_acc)
        
        console.print(f"\n[green]✓ Processing completed successfully![/green]")
        console.print(f"[blue]Output files:[/blue]")
        console.print(f"  • Target: {target_output}")
        console.print(f"  • Background: {background_output}")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Processing interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
