#!/usr/bin/env python3
"""
BAM to Parquet Converter with ML Model Probability Simulation (Optimized)

This script processes target and background BAM files, extracts read information,
simulates machine learning model probabilities, and outputs the data as parquet files.
Optimized for large files with streaming, chunked processing, and multiprocessing.
"""

import click
import pandas as pd
import pysam
import numpy as np
import gc
from pathlib import Path
from typing import Tuple, List, Dict, Any, Iterator, Generator
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from functools import partial
import sys
import os
import tempfile
import shutil
from rich.console import Console
from rich.progress import Progress, TaskID, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.table import Table
from rich import print as rprint
from rich.panel import Panel
from rich.text import Text

console = Console()

# Configuration constants for optimization
DEFAULT_CHUNK_SIZE = 10000  # Number of reads per chunk
DEFAULT_BATCH_SIZE = 50000  # Number of reads per batch for parquet writing
MAX_WORKERS = min(cpu_count(), 8)  # Maximum number of worker processes


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


def count_reads_in_bam(bam_path: str) -> int:
    """
    Count total number of mapped reads in BAM file efficiently.
    
    Args:
        bam_path: Path to the BAM file
        
    Returns:
        Total number of mapped reads
        
    Raises:
        IOError: If the BAM file cannot be read
    """
    try:
        with pysam.AlignmentFile(bam_path, "rb") as bam_file:
            # Try to get count from index if available, otherwise count manually
            try:
                return bam_file.mapped
            except:
                # Fallback to manual counting
                count = 0
                for read in bam_file.fetch():
                    if not read.is_unmapped:
                        count += 1
                return count
    except Exception as e:
        console.print(f"[red]Error counting reads in {bam_path}: {e}[/red]")
        raise


def extract_reads_chunk(bam_path: str, start_pos: int, chunk_size: int) -> List[Dict[str, Any]]:
    """
    Extract a chunk of reads from BAM file starting at a specific position.
    
    Args:
        bam_path: Path to the BAM file
        start_pos: Starting position (read number) in the file
        chunk_size: Number of reads to extract
        
    Returns:
        List of dictionaries containing read information for the chunk
        
    Raises:
        IOError: If the BAM file cannot be read
        ValueError: If the BAM file is corrupted or invalid
    """
    reads_data = []
    
    try:
        with pysam.AlignmentFile(bam_path, "rb") as bam_file:
            current_pos = 0
            reads_processed = 0
            
            for read in bam_file.fetch():
                # Skip to starting position
                if current_pos < start_pos:
                    current_pos += 1
                    continue
                
                # Skip unmapped reads
                if read.is_unmapped:
                    current_pos += 1
                    continue
                
                # Extract read data
                read_data = {
                    'chr': read.reference_name,
                    'start': read.reference_start,
                    'end': read.reference_end,
                    'text': read.query_sequence if read.query_sequence else "",
                    'name': read.query_name,
                    'insert_size': read.template_length,
                }
                reads_data.append(read_data)
                
                reads_processed += 1
                current_pos += 1
                
                # Stop when chunk is complete
                if reads_processed >= chunk_size:
                    break
                    
    except Exception as e:
        console.print(f"[red]Error reading chunk from BAM file {bam_path}: {e}[/red]")
        raise
    
    return reads_data


def process_bam_chunk(args: Tuple[str, int, int, int]) -> Tuple[List[Dict[str, Any]], int]:
    """
    Process a single chunk of BAM file (for multiprocessing).
    
    Args:
        args: Tuple containing (bam_path, start_pos, chunk_size, chunk_id)
        
    Returns:
        Tuple of (reads_data, chunk_id)
    """
    bam_path, start_pos, chunk_size, chunk_id = args
    reads_data = extract_reads_chunk(bam_path, start_pos, chunk_size)
    return reads_data, chunk_id


def extract_reads_from_bam_parallel(bam_path: str, progress: Progress, task_id: TaskID, 
                                   chunk_size: int = DEFAULT_CHUNK_SIZE, 
                                   max_workers: int = MAX_WORKERS) -> Iterator[List[Dict[str, Any]]]:
    """
    Extract read information from a BAM file using parallel processing and streaming.
    
    Args:
        bam_path: Path to the BAM file
        progress: Rich progress bar instance
        task_id: Task ID for progress tracking
        chunk_size: Number of reads per chunk
        max_workers: Maximum number of worker processes
        
    Yields:
        Chunks of read data as lists of dictionaries
        
    Raises:
        IOError: If the BAM file cannot be read
        ValueError: If the BAM file is corrupted or invalid
    """
    try:
        # Count total reads for progress tracking
        total_reads = count_reads_in_bam(bam_path)
        progress.update(task_id, total=total_reads)
        
        # Calculate number of chunks needed
        num_chunks = (total_reads + chunk_size - 1) // chunk_size
        
        # Create chunk arguments for parallel processing
        chunk_args = [
            (bam_path, i * chunk_size, chunk_size, i)
            for i in range(num_chunks)
        ]
        
        processed_reads = 0
        
        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all chunks for processing
            future_to_chunk = {
                executor.submit(process_bam_chunk, args): args[3] 
                for args in chunk_args
            }
            
            # Process completed chunks in order
            completed_chunks = {}
            next_chunk_id = 0
            
            for future in as_completed(future_to_chunk):
                try:
                    reads_data, chunk_id = future.result()
                    completed_chunks[chunk_id] = reads_data
                    
                    # Yield chunks in order
                    while next_chunk_id in completed_chunks:
                        chunk_data = completed_chunks.pop(next_chunk_id)
                        if chunk_data:  # Only yield non-empty chunks
                            yield chunk_data
                            processed_reads += len(chunk_data)
                            progress.update(task_id, completed=processed_reads)
                        next_chunk_id += 1
                        
                        # Trigger periodic garbage collection
                        if next_chunk_id % 10 == 0:  # Every 10 chunks
                            gc.collect()
                            
                except Exception as e:
                    console.print(f"[red]Error processing chunk: {e}[/red]")
                    raise
            
        # Final progress update
        progress.update(task_id, completed=total_reads)
        
    except Exception as e:
        console.print(f"[red]Error reading BAM file {bam_path}: {e}[/red]")
        raise


def generate_probabilities_streaming(n_target: int, n_background: int, model_acc: float) -> Tuple[Iterator[float], Iterator[float]]:
    """
    Generate simulated ML model probabilities for target and background reads using streaming.
    
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
        Tuple of (target_probabilities_iterator, background_probabilities_iterator)
        
    Raises:
        ValueError: If model_acc is not between 0 and 1
    """
    if not 0 <= model_acc <= 1:
        raise ValueError(f"Model accuracy must be between 0 and 1, got {model_acc}")
    
    # Calculate how many reads should be correctly classified
    total_reads = n_target + n_background
    correct_predictions = int(model_acc * total_reads)
    
    # Distribute correct predictions proportionally between target and background
    if total_reads == 0:
        return iter([]), iter([])
    
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
    
    def generate_bimodal_probs_stream(n_samples: int, peak_value: float, is_correct: bool) -> Iterator[float]:
        """
        Generate bimodal distribution with one peak and flat background (streaming version).
        
        Args:
            n_samples: Number of samples to generate
            peak_value: Center of the peak (0.05 or 0.95)
            is_correct: True if this represents correct classification
            
        Yields:
            Individual probability values
        """
        if n_samples == 0:
            return
        
        # Generate indices for peak vs flat distribution
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        # Proportion of samples that form the peak vs flat distribution
        peak_proportion = 0.7
        n_peak = int(n_samples * peak_proportion)
        
        peak_indices = set(indices[:n_peak])
        
        for i in range(n_samples):
            if i in peak_indices:
                # Generate peaked distribution
                if peak_value <= 0.5:
                    prob = np.random.normal(peak_value, 0.02)
                    if is_correct:
                        yield np.clip(prob, 0.0, 0.5)
                    else:
                        yield np.clip(prob, 0.0, 0.5)
                else:
                    prob = np.random.normal(peak_value, 0.02)
                    if is_correct:
                        yield np.clip(prob, 0.501, 1.0)
                    else:
                        yield np.clip(prob, 0.501, 1.0)
            else:
                # Generate flat distribution
                if is_correct:
                    if peak_value <= 0.5:
                        yield np.random.uniform(0.0, 0.5)
                    else:
                        yield np.random.uniform(0.501, 1.0)
                else:
                    if peak_value <= 0.5:
                        yield np.random.uniform(0.0, 0.5)
                    else:
                        yield np.random.uniform(0.501, 1.0)
    
    def target_prob_generator():
        """Generate probabilities for target reads."""
        # Correctly classified targets (prob > 0.5) - peak around 0.95
        for prob in generate_bimodal_probs_stream(target_correct, 0.95, True):
            yield prob
        
        # Misclassified targets (prob ≤ 0.5) - peak around 0.05
        for prob in generate_bimodal_probs_stream(target_misclassified, 0.05, False):
            yield prob
    
    def background_prob_generator():
        """Generate probabilities for background reads."""
        # Correctly classified background (prob ≤ 0.5) - peak around 0.05
        for prob in generate_bimodal_probs_stream(background_correct, 0.05, True):
            yield prob
        
        # Misclassified background (prob > 0.5) - peak around 0.95
        for prob in generate_bimodal_probs_stream(background_misclassified, 0.95, False):
            yield prob
    
    # Shuffle the generators by creating lists and shuffling them
    target_probs = list(target_prob_generator())
    background_probs = list(background_prob_generator())
    
    np.random.shuffle(target_probs)
    np.random.shuffle(background_probs)
    
    return iter(target_probs), iter(background_probs)


def create_dataframe_batch(reads_batch: List[Dict[str, Any]], probabilities_batch: List[float]) -> pd.DataFrame:
    """
    Create a pandas DataFrame from a batch of reads data and probabilities.
    
    Args:
        reads_batch: List of dictionaries containing read information for the batch
        probabilities_batch: List of probabilities for each read in the batch
        
    Returns:
        DataFrame with columns ['chr', 'start', 'end', 'text', 'name', 'prob_class_1', 'insert_size']
        
    Raises:
        ValueError: If the lengths of reads_batch and probabilities_batch don't match
    """
    if len(reads_batch) != len(probabilities_batch):
        raise ValueError(f"Mismatch between reads batch ({len(reads_batch)}) and probabilities batch ({len(probabilities_batch)})")
    
    # Create DataFrame from reads data
    df = pd.DataFrame(reads_batch)
    
    # Add probabilities
    df['prob_class_1'] = probabilities_batch
    
    # Ensure correct column order
    df = df[['chr', 'start', 'end', 'text', 'name', 'prob_class_1', 'insert_size']]
    
    return df


def save_to_parquet_streaming(reads_chunks: Iterator[List[Dict[str, Any]]], 
                            probabilities: Iterator[float], 
                            output_path: str,
                            progress: Progress,
                            task_id: TaskID,
                            batch_size: int = DEFAULT_BATCH_SIZE) -> int:
    """
    Save streaming data to parquet file in batches for memory efficiency.
    
    Args:
        reads_chunks: Iterator of read data chunks
        probabilities: Iterator of probabilities
        output_path: Path for the output parquet file
        progress: Rich progress bar instance
        task_id: Task ID for progress tracking
        batch_size: Number of reads per batch for writing
        
    Returns:
        Total number of reads processed
        
    Raises:
        IOError: If the file cannot be written
    """
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Create temporary directory for batch files
        temp_dir = tempfile.mkdtemp()
        batch_files = []
        total_reads = 0
        
        try:
            current_batch = []
            current_probs = []
            batch_num = 0
            
            # Process chunks and create batches
            for chunk in reads_chunks:
                for read_data in chunk:
                    try:
                        prob = next(probabilities)
                        current_batch.append(read_data)
                        current_probs.append(prob)
                        total_reads += 1
                        
                        # Write batch when it reaches target size
                        if len(current_batch) >= batch_size:
                            batch_df = create_dataframe_batch(current_batch, current_probs)
                            batch_file = os.path.join(temp_dir, f"batch_{batch_num}.parquet")
                            batch_df.to_parquet(batch_file, compression='snappy', index=False)
                            batch_files.append(batch_file)
                            
                            # Clear batch data and trigger garbage collection
                            current_batch.clear()
                            current_probs.clear()
                            del batch_df
                            batch_num += 1
                            
                            # Trigger periodic garbage collection
                            if batch_num % 5 == 0:  # Every 5 batches
                                gc.collect()
                            
                            # Update progress
                            progress.update(task_id, advance=batch_size)
                            
                    except StopIteration:
                        break
            
            # Handle remaining reads in the last batch
            if current_batch:
                batch_df = create_dataframe_batch(current_batch, current_probs)
                batch_file = os.path.join(temp_dir, f"batch_{batch_num}.parquet")
                batch_df.to_parquet(batch_file, compression='snappy', index=False)
                batch_files.append(batch_file)
                progress.update(task_id, advance=len(current_batch))
            
            # Combine all batch files into final parquet file
            if batch_files:
                dfs = []
                for batch_file in batch_files:
                    df = pd.read_parquet(batch_file)
                    dfs.append(df)
                
                # Concatenate all DataFrames
                final_df = pd.concat(dfs, ignore_index=True)
                final_df.to_parquet(output_path, compression='snappy', index=False)
                
                console.print(f"[green]Successfully saved {total_reads} reads to {output_path}[/green]")
        
        finally:
            # Clean up temporary files
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        return total_reads
        
    except Exception as e:
        console.print(f"[red]Error saving to parquet file {output_path}: {e}[/red]")
        raise


def display_summary(total_target: int, total_background: int, model_acc: float, 
                   target_file: str, background_file: str) -> None:
    """
    Display a summary table of the processing results.
    
    Args:
        total_target: Total number of target reads processed
        total_background: Total number of background reads processed
        model_acc: Model accuracy used for simulation
        target_file: Path to target output file
        background_file: Path to background output file
    """
    total_reads = total_target + total_background
    
    # Create summary table
    table = Table(title="Processing Summary")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    
    table.add_row("Target reads", f"{total_target:,}")
    table.add_row("Background reads", f"{total_background:,}")
    table.add_row("Total reads", f"{total_reads:,}")
    table.add_row("Expected accuracy", f"{model_acc:.3f}")
    
    # Add file size information
    if os.path.exists(target_file):
        target_size = os.path.getsize(target_file) / (1024 ** 2)  # MB
        table.add_row("Target file size", f"{target_size:.2f} MB")
    
    if os.path.exists(background_file):
        background_size = os.path.getsize(background_file) / (1024 ** 2)  # MB
        table.add_row("Background file size", f"{background_size:.2f} MB")
    
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
@click.option('--chunk_size', default=DEFAULT_CHUNK_SIZE, type=int,
              help=f'Number of reads per processing chunk (default: {DEFAULT_CHUNK_SIZE})')
@click.option('--batch_size', default=DEFAULT_BATCH_SIZE, type=int,
              help=f'Number of reads per batch for writing (default: {DEFAULT_BATCH_SIZE})')
@click.option('--max_workers', default=MAX_WORKERS, type=int,
              help=f'Maximum number of worker processes (default: {MAX_WORKERS})')
def main(target_bam: str, background_bam: str, model_acc: float, output_dir: str,
         chunk_size: int, batch_size: int, max_workers: int):
    """
    Convert BAM files to parquet format with simulated ML model probabilities (Optimized).
    
    This script processes target and background BAM files using streaming, chunked processing,
    and multiprocessing for optimal performance with large files. It extracts read information,
    simulates machine learning model probabilities based on the specified accuracy,
    and outputs separate parquet files for target and background reads.
    
    Args:
        target_bam: Path to the target BAM file
        background_bam: Path to the background BAM file  
        model_acc: Model accuracy for probability simulation (0-1)
        output_dir: Output directory for parquet files
        chunk_size: Number of reads per processing chunk
        batch_size: Number of reads per batch for writing
        max_workers: Maximum number of worker processes
    """
    console.print(Panel.fit("[bold blue]BAM to Parquet Converter (Optimized)[/bold blue]"))
    
    # Display system information
    system_info = Table(title="System Information")
    system_info.add_column("Resource", style="cyan")
    system_info.add_column("Value", style="magenta")
    system_info.add_row("CPU cores", str(cpu_count()))
    system_info.add_row("Max workers", str(max_workers))
    system_info.add_row("Chunk size", f"{chunk_size:,}")
    system_info.add_row("Batch size", f"{batch_size:,}")
    console.print(system_info)
    console.print()
    
    try:
        # Validate model accuracy
        if not 0 <= model_acc <= 1:
            console.print("[red]Error: Model accuracy must be between 0 and 1[/red]")
            sys.exit(1)
        
        # Validate chunk and batch sizes
        if chunk_size <= 0 or batch_size <= 0:
            console.print("[red]Error: Chunk size and batch size must be positive[/red]")
            sys.exit(1)
        
        if max_workers <= 0:
            console.print("[red]Error: Max workers must be positive[/red]")
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
        
        # Count reads for probability generation
        console.print("[yellow]Counting reads for optimization...[/yellow]")
        target_count = count_reads_in_bam(target_bam)
        background_count = count_reads_in_bam(background_bam)
        
        console.print(f"[blue]Target reads: {target_count:,}[/blue]")
        console.print(f"[blue]Background reads: {background_count:,}[/blue]")
        console.print()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            # Generate probabilities
            prob_task = progress.add_task("Generating probabilities...", total=100)
            target_probs, background_probs = generate_probabilities_streaming(
                target_count, background_count, model_acc
            )
            progress.update(prob_task, completed=100)
            
            # Process target BAM
            target_task = progress.add_task("Processing target BAM...", total=target_count)
            target_chunks = extract_reads_from_bam_parallel(
                target_bam, progress, target_task, chunk_size, max_workers
            )
            
            # Save target data
            save_target_task = progress.add_task("Saving target data...", total=target_count)
            total_target_saved = save_to_parquet_streaming(
                target_chunks, target_probs, target_output, progress, save_target_task, batch_size
            )
            
            # Process background BAM
            background_task = progress.add_task("Processing background BAM...", total=background_count)
            background_chunks = extract_reads_from_bam_parallel(
                background_bam, progress, background_task, chunk_size, max_workers
            )
            
            # Save background data
            save_background_task = progress.add_task("Saving background data...", total=background_count)
            total_background_saved = save_to_parquet_streaming(
                background_chunks, background_probs, background_output, progress, save_background_task, batch_size
            )
        
        # Display summary
        console.print("\n")
        display_summary(total_target_saved, total_background_saved, model_acc, 
                       target_output, background_output)
        
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
