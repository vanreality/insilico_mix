#!/usr/bin/env python
"""In-silico read mixer.

This script mixes reads from a target and a background file based on variation
sites defined in a VCF file. For each variation site, it groups reads that
cover the site, generates simulated depth and fetal fraction using Poisson
distributions, and samples reads accordingly.

Supports multiple input formats: TXT, TSV, and Parquet files are automatically
detected and handled appropriately. Parquet files are processed using polars
for efficient loading.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import click
import numpy as np
import pandas as pd
import polars as pl
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

console = Console()


def detect_file_format(file_path: Path) -> str:
    """Detects the file format based on file extension.
    
    Args:
        file_path: The path to the file.
        
    Returns:
        The detected format: 'parquet', 'tsv', or 'txt'.
    """
    suffix = file_path.suffix.lower()
    if suffix == '.parquet':
        return 'parquet'
    elif suffix in ['.tsv', '.txt']:
        return suffix[1:]  # Remove the dot
    else:
        # Default to txt for unknown extensions
        console.print(f"[yellow]Warning: Unknown file extension '{suffix}'. Treating as txt format.[/yellow]")
        return 'txt'


def load_vcf_file(vcf_path: Path) -> pd.DataFrame:
    """Loads a VCF file and extracts chromosome and position information.

    Args:
        vcf_path: The path to the VCF file.

    Returns:
        A DataFrame containing chromosome and position columns.
        
    Raises:
        SystemExit: If the file cannot be loaded or is empty.
    """
    try:
        console.print(f"Loading VCF file from: [cyan]{vcf_path}[/cyan]")
        vcf_df = pd.read_csv(
            vcf_path,
            sep="\t",
            header=None,
            comment="#",
            usecols=[0, 1],
            names=["chr", "pos"],
        )
        
        if vcf_df.empty:
            console.print("[bold red]Error: VCF file is empty.[/bold red]")
            sys.exit(1)
            
        # Ensure position is integer
        vcf_df["pos"] = vcf_df["pos"].astype(int)
        
        # Remove duplicates
        vcf_df = vcf_df.drop_duplicates()
        
        console.print(f"Loaded [bold yellow]{len(vcf_df):,}[/bold yellow] variation sites")
        return vcf_df
        
    except FileNotFoundError:
        console.print(f"[bold red]Error: VCF file not found at {vcf_path}[/bold red]")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        console.print(f"[bold red]Error: VCF file {vcf_path} is empty.[/bold red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]An error occurred while processing the VCF file: {e}[/bold red]")
        sys.exit(1)


def load_reads_file(file_path: Path, progress: Progress) -> pd.DataFrame:
    """Loads a reads file (TXT/TSV/Parquet) into a pandas DataFrame.

    Automatically detects the file format and loads accordingly.
    For TXT/TSV: assumes tab-separated file.
    For Parquet: uses polars for efficient loading, then converts to pandas.

    Args:
        file_path: The path to the read file.
        progress: A rich Progress instance for displaying progress.

    Returns:
        A pandas DataFrame containing the read data.
        
    Raises:
        SystemExit: If the file cannot be loaded or is empty.
    """
    task_id = progress.add_task(f"Loading [cyan]{file_path.name}[/cyan]", total=None)
    file_format = detect_file_format(file_path)
    
    try:
        if file_format == 'parquet':
            # Use polars for efficient parquet reading
            df_polars = pl.read_parquet(file_path)
            df = df_polars.to_pandas()
            console.print(f"  Loaded [cyan]{len(df):,}[/cyan] reads from parquet file")
        else:
            # Handle TXT/TSV files with pandas
            separator = "\t" if file_format in ['tsv', 'txt'] else "\t"
            df = pd.read_csv(file_path, sep=separator)
            console.print(f"  Loaded [cyan]{len(df):,}[/cyan] reads from {file_format.upper()} file")
        
        progress.update(task_id, completed=1, total=1)
        progress.stop_task(task_id)
        progress.update(task_id, visible=False)
        
        if df.empty:
            console.print(f"[bold red]Error: Input file {file_path} is empty.[/bold red]")
            sys.exit(1)
            
        # Ensure start and end columns are integers if they exist
        if 'start' in df.columns and 'end' in df.columns:
            df['start'] = df['start'].astype(int)
            df['end'] = df['end'].astype(int)
            
        return df
        
    except FileNotFoundError:
        console.print(f"[bold red]Error: Reads file not found at {file_path}[/bold red]")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        console.print(f"[bold red]Error: Reads file {file_path} is empty.[/bold red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]An error occurred while loading {file_path}: {e}[/bold red]")
        sys.exit(1)


def find_reads_covering_site(reads_df: pd.DataFrame, chromosome: str, position: int) -> pd.DataFrame:
    """Finds reads that cover a specific genomic position.
    
    Args:
        reads_df: DataFrame containing read information with chr, start, end columns.
        chromosome: The chromosome of the variation site.
        position: The genomic position of the variation site.
        
    Returns:
        DataFrame containing reads that cover the specified position.
    """
    # Filter reads by chromosome and position overlap
    covering_reads = reads_df[
        (reads_df['chr'] == chromosome) & 
        (reads_df['start'] <= position) & 
        (reads_df['end'] >= position)
    ]
    return covering_reads.copy()


def generate_simulated_parameters(vcf_df: pd.DataFrame, 
                                base_depth: int, 
                                base_ff: float, 
                                triploid_chr: str = None,
                                seed: int = 42) -> Tuple[Dict[Tuple[str, int], int], Dict[Tuple[str, int], float]]:
    """Generates simulated depth and fetal fraction for each variation site using Poisson distribution.
    
    Args:
        vcf_df: DataFrame containing variation sites.
        base_depth: Base depth value for Poisson distribution.
        base_ff: Base fetal fraction for simulation.
        triploid_chr: Chromosome with triploid condition (1.4x fetal fraction multiplier).
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple of (depth_dict, ff_dict) where keys are (chr, pos) tuples.
    """
    np.random.seed(seed)
    
    depth_dict = {}
    ff_dict = {}
    
    for _, row in vcf_df.iterrows():
        chr_pos = (row['chr'], row['pos'])
        
        # Generate simulated depth using Poisson distribution
        simulated_depth = max(1, np.random.poisson(base_depth))
        depth_dict[chr_pos] = simulated_depth
        
        # Generate simulated fetal fraction using Poisson distribution approach
        # We simulate fetal and maternal read counts separately, then calculate fraction
        expected_fetal_reads = base_ff * base_depth
        expected_maternal_reads = (1 - base_ff) * base_depth
        
        simulated_fetal_reads = max(0, np.random.poisson(expected_fetal_reads))
        simulated_maternal_reads = max(1, np.random.poisson(expected_maternal_reads))
        
        total_reads = simulated_fetal_reads + simulated_maternal_reads
        simulated_ff = simulated_fetal_reads / total_reads if total_reads > 0 else 0.01
        
        # Apply triploid chromosome multiplier if applicable
        if triploid_chr and row['chr'] == triploid_chr:
            simulated_ff = min(0.99, simulated_ff * 1.4)
            
        # Ensure fetal fraction is within reasonable bounds
        simulated_ff = max(0.001, min(0.999, simulated_ff))
        ff_dict[chr_pos] = simulated_ff
    
    return depth_dict, ff_dict


def sample_reads_for_site(target_covering: pd.DataFrame, 
                         background_covering: pd.DataFrame,
                         simulated_depth: int,
                         simulated_ff: float,
                         site_seed: int) -> pd.DataFrame:
    """Samples reads for a specific variation site based on simulated depth and fetal fraction.
    
    Args:
        target_covering: Target reads covering the site.
        background_covering: Background reads covering the site.
        simulated_depth: Simulated depth for this site.
        simulated_ff: Simulated fetal fraction for this site.
        site_seed: Random seed for this specific site.
        
    Returns:
        DataFrame containing sampled reads for this site.
    """
    np.random.seed(site_seed)
    
    # Calculate number of reads needed from each source
    n_target_needed = int(round(simulated_depth * simulated_ff))
    n_background_needed = simulated_depth - n_target_needed
    
    sampled_reads = []
    
    # Sample target reads
    if n_target_needed > 0 and len(target_covering) > 0:
        replace_target = n_target_needed > len(target_covering)
        
        sampled_target = target_covering.sample(
            n=n_target_needed,
            replace=replace_target,
            random_state=site_seed
        )
        sampled_reads.append(sampled_target)
    
    # Sample background reads with resampling if insufficient
    if n_background_needed > 0 and len(background_covering) > 0:
        replace_background = n_background_needed > len(background_covering)
        
        sampled_background = background_covering.sample(
            n=n_background_needed,
            replace=replace_background,
            random_state=site_seed + 1  # Different seed for background
        )
        sampled_reads.append(sampled_background)
    
    if sampled_reads:
        return pd.concat(sampled_reads, ignore_index=True)
    else:
        # Return empty DataFrame with same structure as one of the inputs
        if len(target_covering) > 0:
            return pd.DataFrame(columns=target_covering.columns)
        elif len(background_covering) > 0:
            return pd.DataFrame(columns=background_covering.columns)
        else:
            return pd.DataFrame()


@click.command()
@click.option(
    "--target",
    "target_path",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    required=True,
    help="Path to the target reads file (TXT/TSV/Parquet).",
)
@click.option(
    "--background",
    "background_path",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    required=True,
    help="Path to the background reads file (TXT/TSV/Parquet).",
)
@click.option(
    "--ff",
    "fetal_fraction",
    type=click.FloatRange(0.0, 1.0),
    required=True,
    help="Base fetal fraction (proportion of reads from target file).",
)
@click.option(
    "--depth",
    type=click.IntRange(min=1),
    required=True,
    help="Base average sequencing depth for Poisson sampling.",
)
@click.option(
    "--vcf",
    "vcf_path",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    required=True,
    help="VCF file defining variation sites of interest.",
)
@click.option(
    "--output",
    "output_prefix",
    type=click.Path(writable=True, dir_okay=False, path_type=Path),
    required=True,
    help="Prefix for the output Parquet file.",
)
@click.option(
    "--triploid-chr",
    type=str,
    default=None,
    help="Chromosome with triploid condition (1.4x fetal fraction multiplier).",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Random seed for reproducibility.",
)
def mix(
    target_path: Path,
    background_path: Path,
    fetal_fraction: float,
    depth: int,
    vcf_path: Path,
    output_prefix: Path,
    triploid_chr: str,
    seed: int,
):
    """Generates an in-silico mixture of reads based on variation sites from a VCF file.
    
    This tool groups reads by variation sites, generates simulated depth and fetal
    fraction for each site using Poisson distributions, and samples reads accordingly.
    If a triploid chromosome is specified, the fetal fraction for that chromosome
    is multiplied by 1.4.
    """
    console.print("[bold green]Starting read mixture process...[/bold green]")
    np.random.seed(seed)

    # 1. Load VCF file and get variation sites
    vcf_df = load_vcf_file(vcf_path)

    # 2. Load read files
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        transient=True,
    ) as progress:
        console.print("Loading target and background read files...")
        target_df = load_reads_file(target_path, progress)
        background_df = load_reads_file(background_path, progress)

    # Validate required columns
    required_cols = ['chr', 'start', 'end']
    for col in required_cols:
        if col not in target_df.columns:
            console.print(f"[bold red]Error: Target file missing required column '{col}'.[/bold red]")
            sys.exit(1)
        if col not in background_df.columns:
            console.print(f"[bold red]Error: Background file missing required column '{col}'.[/bold red]")
            sys.exit(1)

    # 3. Generate simulated parameters for each variation site
    console.print("Generating simulated depth and fetal fraction parameters...")
    depth_dict, ff_dict = generate_simulated_parameters(
        vcf_df, depth, fetal_fraction, triploid_chr, seed
    )
    
    if triploid_chr:
        triploid_sites = vcf_df[vcf_df['chr'] == triploid_chr]
        console.print(f"Applied 1.4x fetal fraction multiplier to [yellow]{len(triploid_sites):,}[/yellow] sites on [cyan]{triploid_chr}[/cyan]")

    # 4. Process each variation site and sample reads
    console.print("Processing variation sites and sampling reads...")
    all_sampled_reads = []
    sites_with_no_coverage = 0
    sites_processed = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task_id = progress.add_task("Processing sites", total=len(vcf_df))
        
        for idx, (_, row) in enumerate(vcf_df.iterrows()):
            chromosome = row['chr']
            position = row['pos']
            chr_pos = (chromosome, position)
            
            # Find reads covering this variation site
            target_covering = find_reads_covering_site(target_df, chromosome, position)
            background_covering = find_reads_covering_site(background_df, chromosome, position)
            
            if len(target_covering) == 0 and len(background_covering) == 0:
                sites_with_no_coverage += 1
                progress.update(task_id, advance=1)
                continue
            
            # Get simulated parameters for this site
            simulated_depth = depth_dict[chr_pos]
            simulated_ff = ff_dict[chr_pos]
            
            # Sample reads for this site
            sampled_reads = sample_reads_for_site(
                target_covering, 
                background_covering,
                simulated_depth,
                simulated_ff,
                seed + idx  # Unique seed for each site
            )
            
            if not sampled_reads.empty:
                all_sampled_reads.append(sampled_reads)
                sites_processed += 1
            
            progress.update(task_id, advance=1)

    # 5. Report processing statistics
    console.print(f"Processed [green]{sites_processed:,}[/green] sites successfully")
    if sites_with_no_coverage > 0:
        console.print(f"[yellow]Warning: {sites_with_no_coverage:,} sites had no read coverage and were skipped.[/yellow]")

    # 6. Combine all sampled reads and write to Parquet
    if not all_sampled_reads:
        console.print("[bold red]Error: No reads were sampled. Check VCF file and read coverage.[/bold red]")
        sys.exit(1)
    
    console.print("Combining samples and writing to output file...")
    combined_df = pd.concat(all_sampled_reads, ignore_index=True)

    # Rename text column to seq if it exists (for compatibility)
    rename_map = {'text': 'seq'}
    combined_df = combined_df.rename(columns=rename_map)

    # Ensure output directory exists
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_prefix.with_suffix(".parquet")

    try:
        combined_df.to_parquet(output_path, index=False)
        console.print(f"[bold green]✔ Success![/bold green] Mixed reads file saved to: [cyan]{output_path}[/cyan]")
        console.print(f"Total sampled reads: [bold yellow]{len(combined_df):,}[/bold yellow]")
        
        # Calculate some statistics
        avg_depth_achieved = len(combined_df) / sites_processed if sites_processed > 0 else 0
        console.print(f"Average depth achieved: [bold cyan]{avg_depth_achieved:.1f}[/bold cyan] reads per site")
        
    except Exception as e:
        console.print(f"[bold red]Failed to write Parquet file: {e}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    mix()
