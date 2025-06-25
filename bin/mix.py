#!/usr/bin/env python
"""In-silico read mixer.

This script mixes reads from a target and a background file based on a specified
fetal fraction (ff) and sequencing depth. It calculates the number of reads
needed from each source to achieve the desired depth over genomic regions
defined in a BED file, samples the reads, and saves the mixture to a
Parquet file.
"""

import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

console = Console()


def load_bed_file(bed_path: Path) -> int:
    """Loads a BED file and computes the total length of genomic regions.

    Args:
        bed_path: The path to the BED file.

    Returns:
        The total length of all regions in base pairs.
    """
    try:
        console.print(f"Loading BED file from: [cyan]{bed_path}[/cyan]")
        bed_df = pd.read_csv(
            bed_path,
            sep="\t",
            header=None,
            comment="#",
            usecols=[1, 2],
            names=["start", "end"],
        )
        total_length = (bed_df["end"] - bed_df["start"]).sum()
        if total_length <= 0:
            console.print("[bold red]Error: Total length from BED file is non-positive.[/bold red]")
            sys.exit(1)
        console.print(f"Total target region length (L): [bold yellow]{total_length:,}[/bold yellow] bp")
        return total_length
    except FileNotFoundError:
        console.print(f"[bold red]Error: BED file not found at {bed_path}[/bold red]")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        console.print(f"[bold red]Error: BED file {bed_path} is empty.[/bold red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]An error occurred while processing the BED file: {e}[/bold red]")
        sys.exit(1)


def load_reads_file(file_path: Path, progress: Progress) -> pd.DataFrame:
    """Loads a reads file (TXT/TSV) into a pandas DataFrame.

    Assumes a tab-separated file without a header, with columns for chromosome,
    start, end, and other read attributes.

    Args:
        file_path: The path to the read file.
        progress: A rich Progress instance for displaying progress.

    Returns:
        A pandas DataFrame containing the read data.
    """
    task_id = progress.add_task(f"Loading [cyan]{file_path.name}[/cyan]", total=None)
    try:
        # Assuming the file is not excessively large to be read into memory
        df = pd.read_csv(file_path, sep="\t", header=None)
        progress.update(task_id, completed=1, total=1)
        progress.stop_task(task_id)
        progress.update(task_id, visible=False)
        if df.empty:
            console.print(f"[bold red]Error: Input file {file_path} is empty.[/bold red]")
            sys.exit(1)
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


@click.command()
@click.option(
    "--target",
    "target_path",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    required=True,
    help="Path to the target reads file (TXT/TSV).",
)
@click.option(
    "--background",
    "background_path",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    required=True,
    help="Path to the background reads file (TXT/TSV).",
)
@click.option(
    "--ff",
    "fetal_fraction",
    type=click.FloatRange(0.0, 1.0),
    required=True,
    help="Fetal fraction (proportion of reads from target file).",
)
@click.option(
    "--depth",
    type=click.IntRange(min=1),
    required=True,
    help="Desired average sequencing depth.",
)
@click.option(
    "--bed",
    "bed_path",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    required=True,
    help="BED file defining genomic regions of interest.",
)
@click.option(
    "--output",
    "output_prefix",
    type=click.Path(writable=True, dir_okay=False, path_type=Path),
    required=True,
    help="Prefix for the output Parquet file.",
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
    bed_path: Path,
    output_prefix: Path,
    seed: int,
):
    """Generates an in-silico mixture of reads."""
    console.print("[bold green]Starting read mixture process...[/bold green]")
    np.random.seed(seed)

    # 1. Load BED and compute total region length (L)
    total_length = load_bed_file(bed_path)

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

    # Rename columns for clarity, assuming at least 3 columns exist
    num_cols = min(len(target_df.columns), len(background_df.columns))
    if num_cols < 3:
        console.print("[bold red]Error: Input files must have at least 3 columns (chr, start, end).[/bold red]")
        sys.exit(1)
    
    col_map = {0: 'chr', 1: 'start', 2: 'end'}
    target_df = target_df.rename(columns=col_map)
    background_df = background_df.rename(columns=col_map)


    # 3. Calculate reads to sample
    console.print("Calculating number of reads to sample...")

    # Calculate average read lengths
    avg_len_target = (target_df["end"] - target_df["start"]).mean()
    avg_len_background = (background_df["end"] - background_df["start"]).mean()

    if avg_len_target <= 0 or avg_len_background <= 0:
        console.print("[bold red]Error: Average read length is non-positive. Check start/end columns.[/bold red]")
        sys.exit(1)

    # Calculate total bases required and split by fetal fraction
    total_bases = depth * total_length
    target_bases = total_bases * fetal_fraction
    background_bases = total_bases * (1 - fetal_fraction)

    # Calculate number of reads needed from each source
    n_needed_target = int(round(target_bases / avg_len_target))
    n_needed_background = int(round(background_bases / avg_len_background))

    n_total_target = len(target_df)
    n_total_background = len(background_df)
    
    console.print(f"  - Target: [bold yellow]{n_needed_target:,}[/bold yellow] reads to be sampled from [bold cyan]{n_total_target:,}[/bold cyan] available.")
    console.print(f"  - Background: [bold yellow]{n_needed_background:,}[/bold yellow] reads to be sampled from [bold cyan]{n_total_background:,}[/bold cyan] available.")

    # 4. Sample reads
    console.print("Sampling reads...")
    
    # Determine if sampling should be with or without replacement
    replace_target = n_needed_target > n_total_target
    if replace_target:
        console.print("[yellow]Warning: Not enough target reads available. Sampling with replacement.[/yellow]")

    replace_background = n_needed_background > n_total_background
    if replace_background:
        console.print("[yellow]Warning: Not enough background reads available. Sampling with replacement.[/yellow]")
    
    sampled_target = target_df.sample(n=n_needed_target, replace=replace_target, random_state=seed)
    sampled_background = background_df.sample(n=n_needed_background, replace=replace_background, random_state=seed)

    # 5. Combine and write to Parquet
    console.print("Combining samples and writing to output file...")
    combined_df = pd.concat([sampled_target, sampled_background], ignore_index=True)

    # Ensure output directory exists
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_prefix.with_suffix(".parquet")

    try:
        combined_df.to_parquet(output_path, index=False)
        console.print(f"[bold green]✔ Success![/bold green] Mixed reads file saved to: [cyan]{output_path}[/cyan]")
    except Exception as e:
        console.print(f"[bold red]Failed to write Parquet file: {e}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    mix()
