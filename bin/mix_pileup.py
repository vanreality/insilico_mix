#!/usr/bin/env python3
"""
In silico mixing of pileup files.

This script performs in silico mixing of two pileup files (target and background)
to achieve a desired fetal fraction in the final mixture.
"""

import gzip
import sys
from pathlib import Path
from typing import Optional, Tuple

import click
import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

# Initialize rich console for formatted output
console = Console()


def detect_gzip(filepath: str) -> bool:
    """
    Detect if a file is gzipped by checking the file extension.
    
    Args:
        filepath: Path to the file to check.
        
    Returns:
        True if the file appears to be gzipped (has .gz extension), False otherwise.
    """
    return str(filepath).endswith('.gz')


def read_pileup(filepath: str) -> pd.DataFrame:
    """
    Read a pileup file (plain or gzipped) into a pandas DataFrame.
    
    The function automatically detects if the file is gzipped based on the
    file extension and reads it accordingly.
    
    Args:
        filepath: Path to the pileup file (plain or .gz).
        
    Returns:
        DataFrame containing the pileup data with columns:
        chr, pos, ref, alt, af, cfDNA_ref_reads, cfDNA_alt_reads, current_depth.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or missing required columns.
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Determine if file is gzipped and open accordingly
    if detect_gzip(filepath):
        open_func = gzip.open
        mode = 'rt'  # text mode for gzip
    else:
        open_func = open
        mode = 'r'
    
    try:
        with open_func(filepath, mode) as f:
            df = pd.read_csv(f, sep='\t', dtype={
                'chr': str,
                'pos': int,
                'ref': str,
                'alt': str,
                'af': float,
                'cfDNA_ref_reads': int,
                'cfDNA_alt_reads': int,
                'current_depth': int
            })
    except Exception as e:
        raise ValueError(f"Error reading pileup file {filepath}: {e}")
    
    # Validate required columns are present
    required_columns = [
        'chr', 'pos', 'ref', 'alt', 'af',
        'cfDNA_ref_reads', 'cfDNA_alt_reads', 'current_depth'
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {filepath}: {missing_columns}"
        )
    
    if df.empty:
        raise ValueError(f"Empty pileup file: {filepath}")
    
    return df


def merge_pileups(
    target_df: pd.DataFrame,
    background_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge target and background pileup DataFrames on key SNP columns.
    
    Performs an inner join on the five key columns (chr, pos, ref, alt, af)
    to keep only SNPs present in both files. Renames columns to distinguish
    between target and background data.
    
    Args:
        target_df: DataFrame containing target pileup data.
        background_df: DataFrame containing background pileup data.
        
    Returns:
        Merged DataFrame with renamed columns:
        - Target columns: target_ref, target_alt, target_depth
        - Background columns: background_ref, background_alt, background_depth
        - Shared columns: chr, pos, ref, alt, af (from target)
    """
    # Define the key columns for merging
    merge_keys = ['chr', 'pos', 'ref', 'alt', 'af']
    
    # Rename columns before merging to distinguish target vs background
    target_rename = {
        'cfDNA_ref_reads': 'target_ref',
        'cfDNA_alt_reads': 'target_alt',
        'current_depth': 'target_depth'
    }
    background_rename = {
        'cfDNA_ref_reads': 'background_ref',
        'cfDNA_alt_reads': 'background_alt',
        'current_depth': 'background_depth'
    }
    
    target_renamed = target_df[merge_keys + list(target_rename.keys())].rename(
        columns=target_rename
    )
    background_renamed = background_df[merge_keys + list(background_rename.keys())].rename(
        columns=background_rename
    )
    
    # Perform inner join on the key columns
    merged = pd.merge(
        target_renamed,
        background_renamed,
        on=merge_keys,
        how='inner',
        suffixes=('', '_bg')
    )
    
    return merged


def filter_valid_snps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to keep only SNPs where both target and background depths are > 0.
    
    Args:
        df: Merged DataFrame with target_depth and background_depth columns.
        
    Returns:
        Filtered DataFrame containing only SNPs with positive depths in both samples.
    """
    filtered = df[
        (df['target_depth'] > 0) & (df['background_depth'] > 0)
    ].copy()
    
    return filtered


def calculate_target_reads_to_sample(
    background_total: int,
    target_ff: float,
    background_ff: float,
    mix_ff: float
) -> int:
    """
    Calculate the total number of target reads (T_mix) needed to achieve mix_ff.
    
    Given:
    - B_total = total background reads (kept as-is)
    - T_mix = total target reads to sample
    - target_ff = fetal fraction of target sample
    - background_ff = fetal fraction of background sample
    - mix_ff = desired fetal fraction of mixture
    
    The mixture fetal fraction is:
    mix_ff = (T_mix * target_ff + B_total * background_ff) / (T_mix + B_total)
    
    Solving for T_mix:
    T_mix = B_total * (mix_ff - background_ff) / (target_ff - mix_ff)
    
    Args:
        background_total: Total number of background reads.
        target_ff: Fetal fraction of the target sample.
        background_ff: Fetal fraction of the background sample.
        mix_ff: Desired fetal fraction of the mixture.
        
    Returns:
        Total number of target reads to sample (T_mix).
        
    Raises:
        ValueError: If the fetal fraction constraints are not met.
    """
    # Validate fetal fraction constraints
    if not (background_ff < mix_ff < target_ff):
        raise ValueError(
            f"Fetal fraction constraints not met: "
            f"background_ff ({background_ff}) < mix_ff ({mix_ff}) < target_ff ({target_ff})"
        )
    
    # Calculate T_mix
    numerator = background_total * (mix_ff - background_ff)
    denominator = target_ff - mix_ff
    
    if denominator <= 0:
        raise ValueError(
            f"Invalid fetal fraction values: target_ff ({target_ff}) must be > mix_ff ({mix_ff})"
        )
    
    T_mix = int(np.round(numerator / denominator))
    
    if T_mix <= 0:
        raise ValueError(
            f"Calculated T_mix ({T_mix}) must be positive. "
            f"Check fetal fraction values."
        )
    
    return T_mix


def allocate_reads_across_snps(
    df: pd.DataFrame,
    total_reads: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Allocate total reads across SNPs using depth-based weights.
    
    Uses a multinomial distribution to allocate reads proportionally to
    each SNP's target depth. SNPs with higher depth receive more reads.
    
    Args:
        df: DataFrame with target_depth column.
        total_reads: Total number of reads to allocate.
        rng: NumPy random number generator for reproducibility.
        
    Returns:
        Array of allocated read counts per SNP (one per row in df).
    """
    # Calculate weights based on target depth
    weights = df['target_depth'].values.astype(float)
    
    # Normalize weights to probabilities
    weights = weights / weights.sum()
    
    # Use multinomial to allocate reads
    # This gives us the number of reads to sample from each SNP
    allocated = rng.multinomial(total_reads, weights)
    
    return allocated


def sample_ref_alt_counts(
    ref_count: int,
    alt_count: int,
    n_samples: int,
    rng: np.random.Generator
) -> Tuple[int, int]:
    """
    Sample ref and alt counts from a SNP using hypergeometric distribution.
    
    Given a SNP with ref_count reference reads and alt_count alternate reads,
    sample n_samples reads without replacement. This is equivalent to a
    hypergeometric distribution.
    
    Args:
        ref_count: Number of reference reads in the target SNP.
        alt_count: Number of alternate reads in the target SNP.
        n_samples: Number of reads to sample from this SNP.
        rng: NumPy random number generator for reproducibility.
        
    Returns:
        Tuple of (sampled_ref_count, sampled_alt_count).
    """
    total = ref_count + alt_count
    
    # Edge cases
    if n_samples == 0:
        return 0, 0
    
    if n_samples >= total:
        # Sample all reads
        return ref_count, alt_count
    
    if ref_count == 0:
        # Only alt reads available
        return 0, min(n_samples, alt_count)
    
    if alt_count == 0:
        # Only ref reads available
        return min(n_samples, ref_count), 0
    
    # Use hypergeometric distribution to sample without replacement
    # Sample n_samples from total, where ref_count are "successes"
    sampled_ref = rng.hypergeometric(
        ngood=ref_count,      # number of ref reads (successes)
        nbad=alt_count,       # number of alt reads (failures)
        nsample=n_samples     # number of samples to draw
    )
    sampled_alt = n_samples - sampled_ref
    
    return int(sampled_ref), int(sampled_alt)


def mix_pileups(
    target_df: pd.DataFrame,
    background_df: pd.DataFrame,
    target_ff: float,
    background_ff: float,
    mix_ff: float,
    random_seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Perform in silico mixing of target and background pileups.
    
    This is the main mixing function that:
    1. Merges target and background pileups
    2. Filters to valid SNPs
    3. Calculates how many target reads to sample
    4. Allocates reads across SNPs
    5. Samples ref/alt counts from target
    6. Combines with background counts
    
    Args:
        target_df: DataFrame containing target pileup data.
        background_df: DataFrame containing background pileup data.
        target_ff: Fetal fraction of the target sample.
        background_ff: Fetal fraction of the background sample.
        mix_ff: Desired fetal fraction of the mixture.
        random_seed: Random seed for reproducibility (optional).
        
    Returns:
        DataFrame with mixed pileup data in the original format.
    """
    # Initialize random number generator
    rng = np.random.default_rng(random_seed)
    
    # Step 1: Merge pileups
    merged = merge_pileups(target_df, background_df)
    
    if merged.empty:
        raise ValueError("No overlapping SNPs found between target and background files")
    
    # Step 2: Filter to valid SNPs (both depths > 0)
    filtered = filter_valid_snps(merged)
    
    if filtered.empty:
        raise ValueError("No valid SNPs found after filtering (both depths > 0)")
    
    # Step 3: Calculate total background reads and target reads to sample
    background_total = filtered['background_depth'].sum()
    T_mix = calculate_target_reads_to_sample(
        background_total, target_ff, background_ff, mix_ff
    )
    
    # Step 4: Allocate T_mix reads across SNPs using depth-based weights
    allocated_reads = allocate_reads_across_snps(
        filtered, T_mix, rng=rng
    )
    
    # Step 5: Sample ref/alt counts from target for each SNP
    sampled_ref = []
    sampled_alt = []
    
    for idx, row in filtered.iterrows():
        n_samples = allocated_reads[filtered.index.get_loc(idx)]
        ref, alt = sample_ref_alt_counts(
            row['target_ref'],
            row['target_alt'],
            n_samples,
            rng=rng
        )
        sampled_ref.append(ref)
        sampled_alt.append(alt)
    
    # Step 6: Combine sampled target counts with background counts
    result = filtered[['chr', 'pos', 'ref', 'alt', 'af']].copy()
    result['cfDNA_ref_reads'] = (
        filtered['background_ref'].values + np.array(sampled_ref)
    )
    result['cfDNA_alt_reads'] = (
        filtered['background_alt'].values + np.array(sampled_alt)
    )
    result['current_depth'] = (
        result['cfDNA_ref_reads'] + result['cfDNA_alt_reads']
    )
    
    return result


def write_pileup(df: pd.DataFrame, output_path: str) -> None:
    """
    Write pileup DataFrame to a gzipped TSV file.
    
    Args:
        df: DataFrame containing pileup data.
        output_path: Path to the output file (will be gzipped).
        
    Raises:
        IOError: If the file cannot be written.
    """
    try:
        with gzip.open(output_path, 'wt') as f:
            df.to_csv(f, sep='\t', index=False)
    except Exception as e:
        raise IOError(f"Error writing output file {output_path}: {e}")


@click.command()
@click.option(
    '--target',
    required=True,
    type=click.Path(exists=True),
    help='Path to the target pileup file (e.g., PL sample).'
)
@click.option(
    '--background',
    required=True,
    type=click.Path(exists=True),
    help='Path to the background pileup file (e.g., cfDNA sample).'
)
@click.option(
    '--target_ff',
    required=True,
    type=float,
    help='Fetal fraction of the target sample (float, e.g., ~0.9).'
)
@click.option(
    '--background_ff',
    required=True,
    type=float,
    help='Fetal fraction of the background sample (float, e.g., ~0.03).'
)
@click.option(
    '--mix_ff',
    required=True,
    type=float,
    help='Desired fetal fraction for the final mixture (float).'
)
@click.option(
    '--output_prefix',
    required=True,
    type=str,
    help='Prefix for the output file name.'
)
@click.option(
    '--seed',
    type=int,
    default=None,
    help='Random seed for reproducibility (optional).'
)
def main(
    target: str,
    background: str,
    target_ff: float,
    background_ff: float,
    mix_ff: float,
    output_prefix: str,
    seed: Optional[int]
) -> None:
    """
    Perform in silico mixing of pileup files.
    
    This script mixes two pileup files (target and background) to achieve
    a desired fetal fraction in the final mixture. The mixing is performed
    by keeping all background reads and sampling an appropriate number of
    target reads based on the desired mixture fetal fraction.
    
    Example:
        python mix_pileup.py \\
            --target target.pileup.tsv.gz \\
            --background background.pileup.tsv.gz \\
            --target_ff 0.9 \\
            --background_ff 0.03 \\
            --mix_ff 0.15 \\
            --output_prefix mixed_sample \\
            --seed 42
    """
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            # Read target pileup
            task1 = progress.add_task("Reading target pileup...", total=None)
            target_df = read_pileup(target)
            progress.update(task1, completed=True)
            console.print(f"[green]✓[/green] Read {len(target_df)} SNPs from target file")
            
            # Read background pileup
            task2 = progress.add_task("Reading background pileup...", total=None)
            background_df = read_pileup(background)
            progress.update(task2, completed=True)
            console.print(f"[green]✓[/green] Read {len(background_df)} SNPs from background file")
            
            # Perform mixing
            task3 = progress.add_task("Mixing pileups...", total=None)
            mixed_df = mix_pileups(
                target_df,
                background_df,
                target_ff,
                background_ff,
                mix_ff,
                random_seed=seed
            )
            progress.update(task3, completed=True)
            console.print(f"[green]✓[/green] Mixed {len(mixed_df)} SNPs")
            
            # Write output
            output_path = f"{output_prefix}_pileup.tsv.gz"
            task4 = progress.add_task("Writing output...", total=None)
            write_pileup(mixed_df, output_path)
            progress.update(task4, completed=True)
            console.print(f"[green]✓[/green] Output written to {output_path}")
        
        # Print summary statistics
        console.print("\n[bold]Summary Statistics:[/bold]")
        console.print(f"  Total SNPs in mixture: {len(mixed_df):,}")
        console.print(f"  Total depth: {mixed_df['current_depth'].sum():,}")
        console.print(f"  Mean depth per SNP: {mixed_df['current_depth'].mean():.2f}")
        console.print(f"  Median depth per SNP: {mixed_df['current_depth'].median():.2f}")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

