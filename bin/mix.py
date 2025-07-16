#!/usr/bin/env python3
"""
Read Mixture Simulation Tool

This script simulates read mixtures from fetal and maternal reads across different
fetal fractions and depths, with support for triploid chromosomes and genotyping.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import logging

import click
import pandas as pd
import numpy as np
from scipy import stats
from intervaltree import IntervalTree
from rich.console import Console
from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.logging import RichHandler
from rich.panel import Panel
from rich import print as rprint

# Setup rich console and logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)


class ReadMixtureSimulator:
    """
    A class for simulating read mixtures from fetal and maternal genomic data.
    
    This simulator handles loading genomic reads, building interval trees from VCF files,
    intersecting reads with variant sites, and generating mixed read datasets with
    specified fetal fractions and depths.
    """
    
    def __init__(
        self,
        target_file: str,
        background_file: str,
        vcf_file: str,
        ff_min: float,
        ff_max: float,
        ff_step: float,
        depth_min: int,
        depth_max: int,
        depth_step: int,
        triploid_chr: Optional[str] = None
    ):
        """
        Initialize the ReadMixtureSimulator.
        
        Args:
            target_file: Path to parquet file with fetal reads
            background_file: Path to parquet file with maternal reads
            vcf_file: Path to VCF file with variant sites
            ff_min: Minimum fetal fraction
            ff_max: Maximum fetal fraction
            ff_step: Fetal fraction step size
            depth_min: Minimum depth
            depth_max: Maximum depth
            depth_step: Depth step size
            triploid_chr: Optional chromosome with triploid condition (1.5x fetal fraction)
        """
        self.target_file = target_file
        self.background_file = background_file
        self.vcf_file = vcf_file
        self.ff_min = ff_min
        self.ff_max = ff_max
        self.ff_step = ff_step
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.depth_step = depth_step
        self.triploid_chr = triploid_chr
        
        # Data containers
        self.target_reads: Optional[pd.DataFrame] = None
        self.background_reads: Optional[pd.DataFrame] = None
        self.variant_sites: Optional[pd.DataFrame] = None
        self.interval_trees: Dict[str, IntervalTree] = {}
        self.target_pools: Dict[str, Dict[int, pd.DataFrame]] = defaultdict(dict)
        self.background_pools: Dict[str, Dict[int, pd.DataFrame]] = defaultdict(dict)
        self.background_genotypes: Dict[str, Dict[int, str]] = defaultdict(dict)
        
    def load_data(self) -> None:
        """Load target reads, background reads, and variant sites from files."""
        console.print("\n[bold blue]Loading input data...[/bold blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            # Load target reads
            task1 = progress.add_task("Loading target reads...", total=None)
            try:
                self.target_reads = pd.read_parquet(self.target_file)
                logger.info(f"Loaded {len(self.target_reads):,} target reads")
                progress.update(task1, completed=True)
            except Exception as e:
                logger.error(f"Failed to load target reads: {e}")
                sys.exit(1)
            
            # Load background reads
            task2 = progress.add_task("Loading background reads...", total=None)
            try:
                self.background_reads = pd.read_parquet(self.background_file)
                logger.info(f"Loaded {len(self.background_reads):,} background reads")
                progress.update(task2, completed=True)
            except Exception as e:
                logger.error(f"Failed to load background reads: {e}")
                sys.exit(1)
            
            # Load VCF file
            task3 = progress.add_task("Loading VCF file...", total=None)
            try:
                self.variant_sites = pd.read_csv(
                    self.vcf_file,
                    sep='\t',
                    usecols=[0, 1, 3, 4],
                    names=['chr', 'pos', 'ref', 'alt'],
                    comment='#'
                )
                logger.info(f"Loaded {len(self.variant_sites):,} variant sites")
                progress.update(task3, completed=True)
            except Exception as e:
                logger.error(f"Failed to load VCF file: {e}")
                sys.exit(1)
        
        # Validate required columns
        self._validate_data()
        
    def _validate_data(self) -> None:
        """Validate that required columns exist in the loaded data."""
        required_cols = ['chr', 'start', 'end', 'text']
        
        for col in required_cols:
            if col not in self.target_reads.columns:
                logger.error(f"Missing required column '{col}' in target reads")
                sys.exit(1)
            if col not in self.background_reads.columns:
                logger.error(f"Missing required column '{col}' in background reads")
                sys.exit(1)
                
        logger.info("Data validation passed")
        
    def build_interval_trees(self) -> None:
        """Build interval trees for each chromosome from variant sites."""
        console.print("\n[bold blue]Building interval trees...[/bold blue]")
        
        chromosomes = self.variant_sites['chr'].unique()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Building interval trees...", total=len(chromosomes))
            
            for chrom in chromosomes:
                chrom_variants = self.variant_sites[self.variant_sites['chr'] == chrom]
                tree = IntervalTree()
                
                for _, variant in chrom_variants.iterrows():
                    # Use position as both start and end for point intervals
                    tree[variant['pos']:variant['pos']+1] = {
                        'pos': variant['pos'],
                        'ref': variant['ref'],
                        'alt': variant['alt']
                    }
                
                self.interval_trees[chrom] = tree
                progress.advance(task)
                
        logger.info(f"Built interval trees for {len(chromosomes)} chromosomes")
        
    def intersect_reads_with_sites(self) -> None:
        """Intersect target and background reads with variant sites to build read pools."""
        console.print("\n[bold blue]Intersecting reads with variant sites...[/bold blue]")
        
        chromosomes = list(self.interval_trees.keys())
        total_sites = sum(len(tree) for tree in self.interval_trees.values())
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Processing chromosomes...", total=len(chromosomes))
            
            for chrom in chromosomes:
                self._process_chromosome(chrom)
                progress.advance(task)
                
        logger.info(f"Processed {total_sites:,} variant sites across {len(chromosomes)} chromosomes")
        
    def _process_chromosome(self, chrom: str) -> None:
        """Process reads for a single chromosome."""
        tree = self.interval_trees[chrom]
        
        # Filter reads for this chromosome
        target_chrom_reads = self.target_reads[self.target_reads['chr'] == chrom]
        background_chrom_reads = self.background_reads[self.background_reads['chr'] == chrom]
        
        for interval in tree:
            pos = interval.data['pos']
            
            # Find overlapping target reads
            target_overlaps = target_chrom_reads[
                (target_chrom_reads['start'] <= pos) & 
                (target_chrom_reads['end'] >= pos)
            ].copy()
            
            # Find overlapping background reads
            background_overlaps = background_chrom_reads[
                (background_chrom_reads['start'] <= pos) & 
                (background_chrom_reads['end'] >= pos)
            ].copy()
            
            # Store in pools
            self.target_pools[chrom][pos] = target_overlaps
            self.background_pools[chrom][pos] = background_overlaps
            
            # Genotype background reads for this site
            self._genotype_site(chrom, pos, interval.data)
            
    def _genotype_site(self, chrom: str, pos: int, variant_data: Dict[str, Any]) -> None:
        """
        Genotype a variant site using background reads.
        
        Args:
            chrom: Chromosome name
            pos: Position of the variant
            variant_data: Dictionary containing ref and alt alleles
        """
        background_reads = self.background_pools[chrom][pos]
        
        if len(background_reads) == 0:
            # No reads covering this site - assign random genotype
            self.background_genotypes[chrom][pos] = np.random.choice(['0/0', '0/1', '1/1'])
            return
            
        ref_allele = variant_data['ref']
        alt_allele = variant_data['alt']
        
        # Count alleles based on sequence text
        ref_count = 0
        alt_count = 0
        
        for _, read in background_reads.iterrows():
            sequence = read['text']
            read_start = read['start']
            
            # Calculate position within the read sequence
            seq_pos = pos - read_start
            
            if 0 <= seq_pos < len(sequence):
                base = sequence[seq_pos].upper()
                if base == ref_allele.upper():
                    ref_count += 1
                elif base == alt_allele.upper():
                    alt_count += 1
        
        # Determine genotype based on allele frequencies
        total_count = ref_count + alt_count
        if total_count == 0:
            genotype = '0/0'  # Default to homozygous reference
        else:
            alt_freq = alt_count / total_count
            if alt_freq < 0.2:
                genotype = '0/0'
            elif alt_freq > 0.8:
                genotype = '1/1'
            else:
                genotype = '0/1'
                
        self.background_genotypes[chrom][pos] = genotype
        
    def simulate_mixtures(self) -> None:
        """Simulate read mixtures across all fetal fraction and depth combinations."""
        console.print("\n[bold blue]Simulating read mixtures...[/bold blue]")
        
        # Generate parameter combinations
        ff_values = np.arange(self.ff_min, self.ff_max + self.ff_step, self.ff_step)
        depth_values = np.arange(self.depth_min, self.depth_max + self.depth_step, self.depth_step)
        
        total_combinations = len(ff_values) * len(depth_values)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            main_task = progress.add_task("Simulating mixtures...", total=total_combinations)
            
            for ff in ff_values:
                for depth in depth_values:
                    self._simulate_single_mixture(ff, depth)
                    progress.advance(main_task)
                    
        logger.info(f"Generated {total_combinations} mixture files")
        
    def _simulate_single_mixture(self, ff: float, depth: int) -> None:
        """
        Simulate a single mixture with specified fetal fraction and depth.
        
        Args:
            ff: Fetal fraction
            depth: Target depth
        """
        mixed_reads = []
        
        for chrom in self.interval_trees:
            for interval in self.interval_trees[chrom]:
                pos = interval.data['pos']
                mixed_reads.extend(self._simulate_site_mixture(chrom, pos, ff, depth))
        
        # Create output DataFrame
        if mixed_reads:
            output_df = pd.concat(mixed_reads, ignore_index=True)
            
            # Save to parquet file
            output_file = f"{ff:.2f}_{depth}.parquet"
            output_df.to_parquet(output_file, index=False)
            
    def _simulate_site_mixture(self, chrom: str, pos: int, ff: float, depth: int) -> List[pd.DataFrame]:
        """
        Simulate mixture for a single site.
        
        Args:
            chrom: Chromosome name
            pos: Position
            ff: Fetal fraction
            depth: Target depth
            
        Returns:
            List of DataFrames containing mixed reads
        """
        target_reads = self.target_pools[chrom][pos]
        background_reads = self.background_pools[chrom][pos]
        
        # Adjust fetal fraction for triploid chromosome
        actual_ff = ff
        if self.triploid_chr and chrom == self.triploid_chr:
            actual_ff = 1.5 * ff
            
        # Sample depth using Poisson distribution
        sampled_depth = np.random.poisson(depth)
        if sampled_depth <= 0:
            return []
            
        # Calculate target and background read counts
        target_count = int(actual_ff * sampled_depth)
        background_count = sampled_depth - target_count
        
        # Ensure non-negative counts
        target_count = max(0, target_count)
        background_count = max(0, background_count)
        
        selected_reads = []
        
        # Sample target reads
        if target_count > 0 and len(target_reads) > 0:
            if len(target_reads) >= target_count:
                sampled_target = target_reads.sample(n=target_count, replace=False)
            else:
                sampled_target = target_reads.sample(n=target_count, replace=True)
            selected_reads.append(sampled_target)
            
        # Sample background reads
        if background_count > 0 and len(background_reads) > 0:
            selected_background = self._sample_background_reads(
                chrom, pos, background_reads, background_count
            )
            if not selected_background.empty:
                selected_reads.append(selected_background)
                
        return selected_reads
        
    def _sample_background_reads(
        self, 
        chrom: str, 
        pos: int, 
        background_reads: pd.DataFrame, 
        required_count: int
    ) -> pd.DataFrame:
        """
        Sample background reads with genotype-based bias for realistic simulation.
        
        Args:
            chrom: Chromosome name
            pos: Position
            background_reads: Available background reads
            required_count: Number of reads needed
            
        Returns:
            Sampled background reads
        """
        if len(background_reads) == 0:
            return pd.DataFrame()
            
        genotype = self.background_genotypes[chrom][pos]
        
        if len(background_reads) >= required_count:
            # Sufficient reads available - simple random sampling
            return background_reads.sample(n=required_count, replace=False)
        else:
            # Insufficient reads - sample with replacement and bias
            if genotype == '0/0':
                # Homozygous reference - prefer reads supporting reference
                weights = self._calculate_read_weights(background_reads, pos, 'ref_bias')
            elif genotype == '1/1':
                # Homozygous alternate - prefer reads supporting alternate
                weights = self._calculate_read_weights(background_reads, pos, 'alt_bias')
            else:
                # Heterozygous - balanced sampling
                weights = None
                
            return background_reads.sample(
                n=required_count, 
                replace=True, 
                weights=weights
            )
            
    def _calculate_read_weights(
        self, 
        reads: pd.DataFrame, 
        pos: int, 
        bias_type: str
    ) -> np.ndarray:
        """
        Calculate sampling weights for reads based on genotype bias.
        
        Args:
            reads: DataFrame of reads
            pos: Position
            bias_type: Type of bias ('ref_bias' or 'alt_bias')
            
        Returns:
            Array of weights for sampling
        """
        # For now, return uniform weights
        # In a more sophisticated implementation, this would analyze
        # the sequence content to assign weights
        return np.ones(len(reads))


@click.command()
@click.option(
    '--target',
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help='Parquet file with fetal reads'
)
@click.option(
    '--background',
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help='Parquet file with maternal reads'
)
@click.option(
    '--vcf',
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help='VCF file with variant sites'
)
@click.option(
    '--ff-min',
    required=True,
    type=float,
    help='Minimum fetal fraction'
)
@click.option(
    '--ff-max',
    required=True,
    type=float,
    help='Maximum fetal fraction'
)
@click.option(
    '--ff-step',
    required=True,
    type=float,
    help='Fetal fraction step size'
)
@click.option(
    '--depth-min',
    required=True,
    type=int,
    help='Minimum depth'
)
@click.option(
    '--depth-max',
    required=True,
    type=int,
    help='Maximum depth'
)
@click.option(
    '--depth-step',
    required=True,
    type=int,
    help='Depth step size'
)
@click.option(
    '--triploid-chr',
    default=None,
    help='Chromosome with triploid condition (1.5x fetal fraction)'
)
def main(
    target: Path,
    background: Path,
    vcf: Path,
    ff_min: float,
    ff_max: float,
    ff_step: float,
    depth_min: int,
    depth_max: int,
    depth_step: int,
    triploid_chr: Optional[str]
) -> None:
    """
    Simulate read mixtures from fetal and maternal genomic data.
    
    This tool generates mixed read datasets across different fetal fractions
    and depths, with support for triploid chromosomes and genotyping-based
    background read simulation.
    """
    # Display header
    console.print(Panel.fit(
        "[bold magenta]Read Mixture Simulator[/bold magenta]\n"
        "Simulating fetal-maternal read mixtures",
        title="🧬 Genomics Tool",
        title_align="left"
    ))
    
    # Validate parameters
    if ff_min >= ff_max:
        console.print("[red]Error: ff-min must be less than ff-max[/red]")
        sys.exit(1)
    if depth_min >= depth_max:
        console.print("[red]Error: depth-min must be less than depth-max[/red]")
        sys.exit(1)
        
    # Display parameters
    params_table = Table(title="Simulation Parameters")
    params_table.add_column("Parameter", style="cyan")
    params_table.add_column("Value", style="magenta")
    
    params_table.add_row("Target file", str(target))
    params_table.add_row("Background file", str(background))
    params_table.add_row("VCF file", str(vcf))
    params_table.add_row("Fetal fraction range", f"{ff_min} - {ff_max} (step: {ff_step})")
    params_table.add_row("Depth range", f"{depth_min} - {depth_max} (step: {depth_step})")
    if triploid_chr:
        params_table.add_row("Triploid chromosome", triploid_chr)
    
    console.print(params_table)
    
    try:
        # Initialize simulator
        simulator = ReadMixtureSimulator(
            target_file=str(target),
            background_file=str(background),
            vcf_file=str(vcf),
            ff_min=ff_min,
            ff_max=ff_max,
            ff_step=ff_step,
            depth_min=depth_min,
            depth_max=depth_max,
            depth_step=depth_step,
            triploid_chr=triploid_chr
        )
        
        # Execute simulation pipeline
        simulator.load_data()
        simulator.build_interval_trees()
        simulator.intersect_reads_with_sites()
        simulator.simulate_mixtures()
        
        console.print("\n[bold green]✓ Simulation completed successfully![/bold green]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Simulation interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == '__main__':
    main()
