"""
OpenKeywords CLI - Generate SEO keywords from the command line.
"""

import asyncio
import logging
import os
import sys

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .generator import KeywordGenerator
from .models import CompanyInfo, GenerationConfig

console = Console()


def setup_logging(verbose: bool):
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


@click.group()
@click.version_option(version="0.2.0")
def main():
    """
    OpenKeywords - AI-powered SEO keyword generation.

    Generate high-quality, clustered SEO keywords using Google Gemini.
    Optionally fetch real search volume data from SE Ranking.
    """
    pass


@main.command()
@click.option("--company", "-c", required=True, help="Company name")
@click.option("--url", "-u", default="", help="Company website URL")
@click.option("--industry", "-i", default=None, help="Industry category")
@click.option("--description", "-d", default=None, help="Company description")
@click.option("--services", "-s", default=None, help="Services (comma-separated)")
@click.option("--products", "-p", default=None, help="Products (comma-separated)")
@click.option("--audience", "-a", default=None, help="Target audience")
@click.option("--location", "-l", default=None, help="Target location")
@click.option("--count", "-n", default=50, help="Number of keywords to generate")
@click.option("--clusters", default=6, help="Number of clusters")
@click.option("--language", default="english", help="Target language")
@click.option("--region", default="us", help="Target region (country code)")
@click.option("--min-score", default=40, help="Minimum company-fit score")
@click.option("--with-gaps", is_flag=True, help="Enable SE Ranking gap analysis (requires URL)")
@click.option("--with-research", is_flag=True, help="Enable deep research (Reddit, Quora, forums)")
@click.option("--output", "-o", default=None, help="Output file (csv or json)")
@click.option("--competitors", default=None, help="Competitor URLs (comma-separated)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def generate(
    company: str,
    url: str,
    industry: str,
    description: str,
    services: str,
    products: str,
    audience: str,
    location: str,
    count: int,
    clusters: int,
    language: str,
    region: str,
    min_score: int,
    with_gaps: bool,
    with_research: bool,
    competitors: str,
    output: str,
    verbose: bool,
):
    """
    Generate SEO keywords for a company.

    Examples:

        openkeywords generate --company "Acme Software" --industry "B2B SaaS"

        openkeywords generate -c "Acme" -s "project management,collaboration" -n 100

        openkeywords generate -c "Acme" --with-research --output keywords.csv
    """
    setup_logging(verbose)

    # Check API keys
    if not os.getenv("GEMINI_API_KEY"):
        console.print("[red]Error: GEMINI_API_KEY environment variable not set[/red]")
        console.print("Set it with: export GEMINI_API_KEY='your-key'")
        sys.exit(1)

    if with_gaps and not os.getenv("SERANKING_API_KEY"):
        console.print("[yellow]Warning: SERANKING_API_KEY not set - gap analysis will be skipped[/yellow]")
        with_gaps = False

    if with_gaps and not url:
        console.print("[yellow]Warning: --url required for gap analysis - skipping[/yellow]")
        with_gaps = False

    # Build company info
    company_info = CompanyInfo(
        name=company,
        url=url,
        industry=industry,
        description=description,
        services=services.split(",") if services else [],
        products=products.split(",") if products else [],
        target_audience=audience,
        target_location=location,
        competitors=competitors.split(",") if competitors else [],
    )

    # Build config
    config = GenerationConfig(
        target_count=count,
        min_score=min_score,
        enable_clustering=True,
        cluster_count=clusters,
        language=language,
        region=region,
        enable_research=with_research,
    )

    console.print(f"\n[bold blue]OpenKeywords[/bold blue]")
    console.print(f"Generating {count} keywords for [green]{company}[/green]")
    if with_research:
        console.print("[bold magenta]Deep Research enabled (Reddit, Quora, forums)[/bold magenta]")
    console.print()

    # Run generation
    async def run():
        generator = KeywordGenerator()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Generating keywords...", total=None)
            result = await generator.generate(company_info, config)

        return result

    try:
        result = asyncio.run(run())
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

    # Display results
    console.print(f"\n[green]Generated {len(result.keywords)} keywords[/green]")
    console.print(f"  Processing time: {result.processing_time_seconds:.1f}s")
    console.print(f"  Average score: {result.statistics.avg_score:.1f}")
    console.print(f"  Clusters: {len(result.clusters)}")

    # Source breakdown (if research was enabled)
    if result.statistics.source_breakdown and len(result.statistics.source_breakdown) > 1:
        console.print("\n[bold]Keyword Sources:[/bold]")
        for source, src_count in result.statistics.source_breakdown.items():
            pct = (src_count / len(result.keywords)) * 100 if result.keywords else 0
            console.print(f"  {source}: {src_count} ({pct:.0f}%)")

    # Intent breakdown
    if result.statistics.intent_breakdown:
        console.print("\n[bold]Intent Distribution:[/bold]")
        for intent, int_count in result.statistics.intent_breakdown.items():
            pct = (int_count / len(result.keywords)) * 100 if result.keywords else 0
            console.print(f"  {intent}: {int_count} ({pct:.0f}%)")

    # Show top keywords
    console.print("\n[bold]Top 10 Keywords:[/bold]")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Keyword", style="cyan")
    table.add_column("Intent", style="green")
    table.add_column("Score", justify="right")
    table.add_column("Cluster", style="yellow")

    if with_gaps:
        table.add_column("Volume", justify="right")
        table.add_column("Difficulty", justify="right")

    if with_research:
        table.add_column("Source", style="magenta")

    for kw in result.keywords[:10]:
        row = [
            kw.keyword,
            kw.intent,
            str(kw.score),
            kw.cluster_name or "-",
        ]
        if with_gaps:
            row.extend([str(kw.volume), str(kw.difficulty)])
        if with_research:
            row.append(kw.source)
        table.add_row(*row)

    console.print(table)

    # Export if output specified
    if output:
        if output.endswith(".csv"):
            result.to_csv(output)
            console.print(f"\n[green]Exported to {output}[/green]")
        elif output.endswith(".json"):
            result.to_json(output)
            console.print(f"\n[green]Exported to {output}[/green]")
        else:
            console.print(f"[yellow]Unknown format. Use .csv or .json extension.[/yellow]")


@main.command()
def check():
    """
    Check API key configuration.
    """
    console.print("\n[bold blue]OpenKeywords - Configuration Check[/bold blue]\n")

    gemini_key = os.getenv("GEMINI_API_KEY")
    seranking_key = os.getenv("SERANKING_API_KEY")

    if gemini_key:
        console.print(f"[green]GEMINI_API_KEY: Set ({gemini_key[:8]}...)[/green]")
    else:
        console.print("[red]GEMINI_API_KEY: Not set[/red]")

    if seranking_key:
        console.print(f"[green]SERANKING_API_KEY: Set ({seranking_key[:8]}...)[/green]")
    else:
        console.print("[yellow]SERANKING_API_KEY: Not set (optional)[/yellow]")

    console.print("\n[bold]Setup Instructions:[/bold]")
    console.print("  export GEMINI_API_KEY='your-gemini-api-key'")
    console.print("  export SERANKING_API_KEY='your-seranking-key'  # Optional")


if __name__ == "__main__":
    main()
