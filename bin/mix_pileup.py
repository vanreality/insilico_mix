#!/usr/bin/env python3
"""
SNP Pileup Mixing Tool

This module implements mixing of SNP pileup data from target and background samples
to simulate various fetal fraction (ff) and sequencing depth combinations.

The tool performs statistical sampling using hypergeometric distribution for target
reads and beta-binomial distribution for background reads to create realistic
mixed samples for downstream analysis.
"""

import gzip
import logging
import multiprocessing as mp
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import click
import numpy as np
import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, TaskID
from scipy.stats import hypergeom

# Initialize rich console for formatted output
console = Console()

# Configure logging with rich handler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console)]
)
logger = logging.getLogger(__name__)


class PileupMixer:
    """
    A class to handle SNP pileup mixing operations.
    
    This class manages the loading, filtering, and mixing of pileup data
    from target and background samples to simulate various experimental
    conditions.
    """
    
    def __init__(self, target_file: str, background_file: str, snp_file: str):
        """
        Initialize the PileupMixer with input files.
        
        Args:
            target_file: Path to target pileup file (tsv.gz format)
            background_file: Path to background pileup file (tsv.gz format)
            snp_file: Path to SNP sites file (tsv format)
        
        Raises:
            FileNotFoundError: If any input file doesn't exist
            ValueError: If file formats are invalid
        """
        self.target_file = Path(target_file)
        self.background_file = Path(background_file)
        self.snp_file = Path(snp_file)
        
        # Validate input files exist
        for file_path in [self.target_file, self.background_file, self.snp_file]:
            if not file_path.exists():
                raise FileNotFoundError(f"Input file not found: {file_path}")
        
        self.target_df: Optional[pd.DataFrame] = None
        self.background_df: Optional[pd.DataFrame] = None
        self.snp_sites: Optional[pd.DataFrame] = None
        self.filtered_target: Optional[pd.DataFrame] = None
        self.filtered_background: Optional[pd.DataFrame] = None
        self.background_genotypes: Optional[Dict] = None
    
    def load_data(self) -> None:
        """
        Load and validate all input data files.
        
        Raises:
            ValueError: If data format is invalid or columns are missing
        """
        console.print("[blue]Loading input data...[/blue]")
        
        try:
            # Load target pileup
            self.target_df = pd.read_csv(
                self.target_file,
                sep='\t',
                compression='gzip',
                dtype={
                    'ref': str,
                    'alt': str,
                    'af': float,
                    'cfDNA_ref_reads': int,
                    'cfDNA_alt_reads': int,
                    'current_depth': int
                }
            )
            
            # Load background pileup
            self.background_df = pd.read_csv(
                self.background_file,
                sep='\t',
                compression='gzip',
                dtype={
                    'ref': str,
                    'alt': str,
                    'af': float,
                    'cfDNA_ref_reads': int,
                    'cfDNA_alt_reads': int,
                    'current_depth': int
                }
            )
            
            # Load SNP sites
            self.snp_sites = pd.read_csv(
                self.snp_file,
                sep='\t',
                usecols=[0, 1],
                names=['chr', 'pos'],
                dtype={'chr': str, 'pos': int}
            )
            
        except Exception as e:
            raise ValueError(f"Error loading data files: {e}")
        
        # Validate required columns
        required_pileup_cols = ['ref', 'alt', 'af', 'cfDNA_ref_reads', 'cfDNA_alt_reads', 'current_depth']
        for df_name, df in [('target', self.target_df), ('background', self.background_df)]:
            missing_cols = set(required_pileup_cols) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing columns in {df_name} file: {missing_cols}")
        
        console.print(f"[green]✓[/green] Loaded {len(self.target_df):,} target records")
        console.print(f"[green]✓[/green] Loaded {len(self.background_df):,} background records")
        console.print(f"[green]✓[/green] Loaded {len(self.snp_sites):,} SNP sites")
    
    def filter_data(self) -> None:
        """
        Filter pileup data according to specified criteria.
        
        Filtering steps:
        1. Keep only SNP sites of interest
        2. Apply depth thresholds (>10 for target, >20 for background)
        3. Keep only common SNP sites in both datasets
        """
        console.print("[blue]Filtering pileup data...[/blue]")
        
        # Add chr_pos identifier for filtering
        self.target_df['chr_pos'] = self.target_df['chr'].astype(str) + '_' + self.target_df['pos'].astype(str)
        self.background_df['chr_pos'] = self.background_df['chr'].astype(str) + '_' + self.background_df['pos'].astype(str)
        self.snp_sites['chr_pos'] = self.snp_sites['chr'].astype(str) + '_' + self.snp_sites['pos'].astype(str)
        
        snp_set = set(self.snp_sites['chr_pos'])
        
        # Filter by SNP sites of interest
        target_filtered = self.target_df[self.target_df['chr_pos'].isin(snp_set)].copy()
        background_filtered = self.background_df[self.background_df['chr_pos'].isin(snp_set)].copy()
        
        console.print(f"[green]✓[/green] After SNP filtering: {len(target_filtered):,} target, {len(background_filtered):,} background")
        
        # Apply depth thresholds
        target_filtered = target_filtered[target_filtered['current_depth'] > 10].copy()
        background_filtered = background_filtered[background_filtered['current_depth'] > 20].copy()
        
        console.print(f"[green]✓[/green] After depth filtering: {len(target_filtered):,} target, {len(background_filtered):,} background")
        
        # Keep only common SNP sites
        target_sites = set(target_filtered['chr_pos'])
        background_sites = set(background_filtered['chr_pos'])
        common_sites = target_sites & background_sites
        
        self.filtered_target = target_filtered[target_filtered['chr_pos'].isin(common_sites)].copy()
        self.filtered_background = background_filtered[background_filtered['chr_pos'].isin(common_sites)].copy()
        
        console.print(f"[green]✓[/green] Final filtered data: {len(common_sites):,} common SNP sites")
        
        if len(common_sites) == 0:
            raise ValueError("No common SNP sites found after filtering")
    
    def genotype_background(self) -> None:
        """
        Pre-genotype background data using VAF thresholds.
        
        This method classifies each SNP site into genotypes based on VAF:
        - 0/0 (homozygous reference): VAF < 0.2
        - 0/1 (heterozygous): 0.2 ≤ VAF ≤ 0.8
        - 1/1 (homozygous alternate): VAF > 0.8
        """
        console.print("[blue]Genotyping background data using VAF thresholds...[/blue]")
        
        self.background_genotypes = {}
        genotype_counts = {'0/0': 0, '0/1': 0, '1/1': 0}
        
        for _, row in self.filtered_background.iterrows():
            chr_pos = row['chr_pos']
            vaf = row['cfDNA_alt_reads'] / row['current_depth']
            
            # Genotype based on VAF thresholds
            if vaf < 0.3:
                genotype = '0/0'
                alt_prob = 0.001  # No alternate alleles
            elif vaf <= 0.7:
                genotype = '0/1'
                alt_prob = 0.5  # 50% alternate alleles
            else:
                genotype = '1/1'
                alt_prob = 0.999  # All alternate alleles
            
            genotype_counts[genotype] += 1
            
            self.background_genotypes[chr_pos] = {
                'vaf': vaf,
                'genotype': genotype,
                'alt_prob': alt_prob,
                'ref': row['ref'],
                'alt': row['alt']
            }
        
        console.print(f"[green]✓[/green] Generated genotypes for {len(self.background_genotypes):,} background sites")
        console.print(f"[green]✓[/green] Genotype distribution: 0/0={genotype_counts['0/0']:,}, 0/1={genotype_counts['0/1']:,}, 1/1={genotype_counts['1/1']:,}")


def sample_target_reads(row: pd.Series, target_reads: int) -> Tuple[int, int]:
    """
    Sample target reads using hypergeometric distribution.
    
    Args:
        row: Row from filtered target dataframe
        target_reads: Number of target reads to sample
    
    Returns:
        Tuple of (ref_reads, alt_reads)
    """
    total_reads = row['current_depth_target']
    ref_reads_orig = row['cfDNA_ref_reads_target']
    
    # Use hypergeometric distribution to sample
    # Population: total_reads, successes: ref_reads_orig, sample size: target_reads
    try:
        # Validate hypergeometric parameters before sampling
        if target_reads > total_reads or ref_reads_orig > total_reads or ref_reads_orig < 0:
            raise ValueError("Invalid hypergeometric parameters")
        
        ref_sampled = hypergeom.rvs(total_reads, ref_reads_orig, target_reads)
        alt_sampled = target_reads - ref_sampled
        return max(0, ref_sampled), max(0, alt_sampled)
    except (ValueError, OverflowError) as e:
        # Fallback to proportional sampling
        ref_prop = ref_reads_orig / total_reads if total_reads > 0 else 0.5
        ref_sampled = int(target_reads * ref_prop)
        alt_sampled = target_reads - ref_sampled
        return max(0, ref_sampled), max(0, alt_sampled)


def sample_background_reads(genotype_info: Dict, background_reads: int) -> Tuple[int, int]:
    """
    Sample background reads using binomial distribution based on genotype.
    For 0/1 genotypes, adds variance by sampling probability from beta distribution.
    
    Args:
        genotype_info: Dictionary containing genotype information with alt_prob
        background_reads: Number of background reads to sample
    
    Returns:
        Tuple of (ref_reads, alt_reads)
    """
    if background_reads <= 0:
        return 0, 0
    
    genotype = genotype_info['genotype']
    alt_prob = genotype_info['alt_prob']
    
    try:
        # For heterozygous sites, add variance by sampling from beta distribution
        if genotype == '0/1':
            # Beta distribution parameters for variance around 0.5
            # alpha = beta = 20 gives reasonable variance while keeping mean at 0.5
            alpha = 20
            beta = 20
            # Sample probability from beta distribution
            sampled_prob = np.random.beta(alpha, beta)
            # Ensure probability stays within reasonable bounds for heterozygous sites
            sampled_prob = max(0.1, min(0.9, sampled_prob))
            alt_reads = np.random.binomial(background_reads, sampled_prob)
        else:
            # For homozygous sites, use fixed probabilities
            alt_reads = np.random.binomial(background_reads, alt_prob)

        alt_reads = np.random.binomial(background_reads, alt_prob)
        
        ref_reads = background_reads - alt_reads
        return max(0, ref_reads), max(0, alt_reads)
    except (ValueError, OverflowError) as e:
        # Fallback: deterministic assignment based on genotype
        if genotype == '0/0':
            return background_reads, 0
        elif genotype == '0/1':
            alt_reads = background_reads // 2
            ref_reads = background_reads - alt_reads
            return ref_reads, alt_reads
        else:  # genotype == '1/1'
            return 0, background_reads


def mix_single_combination(args: Tuple) -> str:
    """
    Perform mixing for a single ff-depth combination.
    
    Args:
        args: Tuple containing (ff, depth, repeat_idx, mixer_data, factor, output_prefix)
    
    Returns:
        Output filename of the generated mixed pileup
    """
    try:
        ff, depth, repeat_idx, mixer_data, factor, output_prefix, model_acc = args
        
        filtered_target, filtered_background, background_genotypes = mixer_data
        
        # Validate inputs
        if len(filtered_target) == 0:
            raise ValueError("No target data available for mixing")
        if depth <= 0:
            raise ValueError(f"Invalid depth: {depth}")
        if not (0 <= ff <= 1):
            raise ValueError(f"Invalid fetal fraction: {ff}")
        
        # Calculate reads distribution
        n_snps = len(filtered_target)
        n_total_reads = n_snps * depth
        n_target_reads = int(n_total_reads * ff)
        n_background_reads = n_total_reads - n_target_reads
        
        # Distribute reads proportionally to original depths
        total_target_depth = filtered_target['current_depth'].sum()
        total_background_depth = filtered_background['current_depth'].sum()

        filtered_df = pd.merge(filtered_target, 
                               filtered_background, 
                               on=['chr_pos', 'chr', 'pos', 'ref', 'alt', 'af'], 
                               how='inner', 
                               suffixes=('_target', '_background'))
        
        if total_target_depth <= 0:
            raise ValueError("Total target depth is zero or negative")
        
        results = []
        
        for _, row in filtered_df.iterrows():
            chr_pos = row['chr_pos']
            
            # Calculate target reads for this SNP
            target_proportion = row['current_depth_target'] / total_target_depth
            snp_target_reads = int(n_target_reads * target_proportion)

            background_proportion = row['current_depth_background'] / total_background_depth
            snp_background_reads = int(n_background_reads * background_proportion)
            
            # Sample target reads
            target_ref, target_alt = sample_target_reads(row, snp_target_reads)
            
            # Sample background reads
            background_genotype = background_genotypes[chr_pos]
            background_ref, background_alt = sample_background_reads(background_genotype, snp_background_reads)
            
            # Combine reads
            final_ref = target_ref + background_ref
            final_alt = target_alt + background_alt
            final_depth = final_ref + final_alt

            # Use model accuracy to classify target reads
            fetal_ref = int(target_ref * model_acc + background_ref * (1 - model_acc))
            fetal_alt = int(target_alt * model_acc + background_alt * (1 - model_acc))
            
            results.append({
                'chr': row['chr'],
                'pos': row['pos'],
                'ref': row['ref'],
                'alt': row['alt'],
                'af': row['af'],
                'cfDNA_ref_reads': final_ref,
                'cfDNA_alt_reads': final_alt,
                'current_depth': final_depth,
                'fetal_ref_reads_from_model': fetal_ref,
                'fetal_alt_reads_from_model': fetal_alt
            })
        
        # Create output dataframe and save
        output_df = pd.DataFrame(results)
        output_filename = f"{output_prefix}/Mix_{ff:.3f}_{depth}_{factor}_{repeat_idx}_pileup.tsv.gz"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        
        # Save to compressed TSV
        output_df.to_csv(output_filename, sep='\t', compression='gzip', index=False)
        
        return output_filename
        
    except Exception as e:
        # Log detailed error information for debugging
        error_msg = f"Error in mixing ff={ff}, depth={depth}, repeat={repeat_idx}: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)


@click.command()
@click.option('--target', required=True, help='Target pileup file (tsv.gz format)')
@click.option('--background', required=True, help='Background pileup file (tsv.gz format)')
@click.option('--tsv', required=True, help='SNP sites file (tsv format)')
@click.option('--factor', required=True, help='Factor string for output files')
@click.option('--ff-min', type=float, required=True, help='Minimum fetal fraction')
@click.option('--ff-max', type=float, required=True, help='Maximum fetal fraction')
@click.option('--ff-number', type=int, required=True, help='Number of fetal fraction values')
@click.option('--depth-min', type=int, required=True, help='Minimum sequencing depth')
@click.option('--depth-max', type=int, required=True, help='Maximum sequencing depth')
@click.option('--depth-number', type=int, required=True, help='Number of depth values')
@click.option('--repeat', type=int, default=1, help='Number of repeats for each combination')
@click.option('--output-prefix', required=True, help='Output directory prefix')
@click.option('--threads', type=int, default=None, help='Number of threads (default: CPU count)')
@click.option('--seed', type=int, default=42, help='Random seed for reproducibility')
@click.option('--model-acc', type=float, default=0.81, help='Model accuracy for target reads classification')
def main(target: str, background: str, tsv: str, factor: str,
         ff_min: float, ff_max: float, ff_number: int,
         depth_min: int, depth_max: int, depth_number: int,
         repeat: int, output_prefix: str, threads: Optional[int], seed: int, 
         model_acc: float) -> None:
    """
    SNP Pileup Mixing Tool
    
    Mix target and background pileup data to simulate various fetal fraction
    and sequencing depth combinations for downstream analysis.
    
    Example:
        python mix_pileup.py --target target.tsv.gz --background bg.tsv.gz 
        --tsv snps.tsv --factor test --ff-min 0.05 --ff-max 0.15 --ff-number 3
        --depth-min 100 --depth-max 300 --depth-number 3 --repeat 2 
        --output-prefix ./output
    """
    
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Determine number of threads
    if threads is None:
        threads = mp.cpu_count()
    
    console.print(f"[bold blue]SNP Pileup Mixing Tool[/bold blue]")
    console.print(f"Using {threads} threads with random seed {seed}")
    
    try:
        # Initialize mixer and load data
        mixer = PileupMixer(target, background, tsv)
        mixer.load_data()
        mixer.filter_data()
        mixer.genotype_background()
        
        # Generate parameter combinations
        ff_values = np.linspace(ff_min, ff_max, ff_number)
        depth_values = np.linspace(depth_min, depth_max, depth_number, dtype=int)
        
        # Prepare all combinations
        combinations = []
        for ff in ff_values:
            for depth in depth_values:
                for repeat_idx in range(repeat):
                    mixer_data = (mixer.filtered_target, mixer.filtered_background, mixer.background_genotypes)
                    combinations.append((ff, depth, repeat_idx, mixer_data, factor, output_prefix, model_acc))
        
        total_combinations = len(combinations)
        console.print(f"[blue]Generating {total_combinations:,} mixed pileup files...[/blue]")
        
        # Execute mixing with progress tracking
        with Progress() as progress:
            task = progress.add_task("[cyan]Mixing pileups...", total=total_combinations)
            
            with ProcessPoolExecutor(max_workers=threads) as executor:
                # Submit all jobs
                futures = [executor.submit(mix_single_combination, combo) for combo in combinations]
                
                # Process completed jobs
                completed_files = []
                for future in as_completed(futures):
                    try:
                        output_file = future.result()
                        completed_files.append(output_file)
                        progress.update(task, advance=1)
                    except Exception as e:
                        logger.error(f"Error in mixing process: {e}")
                        progress.update(task, advance=1)
        
        console.print(f"[bold green]✓ Successfully generated {len(completed_files):,} mixed pileup files[/bold green]")
        console.print(f"Output files saved to: {output_prefix}")
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise click.ClickException(str(e))


if __name__ == '__main__':
    main()
