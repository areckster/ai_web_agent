# main.py — 2025-07-01
import click, torch
from agent import run_research_agent
from crawler import crawl_site
from memory import Memory

@click.group()
def cli() -> None:
    """CLI for the local AI Web-Research Assistant."""
    pass

@cli.command()
def verify_gpu() -> None:
    """Quick Metal/MPS sanity-check."""
    if torch.backends.mps.is_available():
        click.echo(click.style("✅  Metal (MPS) backend detected.", fg="green"))
    else:
        click.echo(click.style("⚠️  Metal backend NOT detected (CPU fallback).", fg="yellow"))

@cli.command()
@click.argument("query", nargs=-1, required=True)
@click.option("-v", "--verbose", is_flag=True, help="Print agent thoughts each loop.")
def research(query: tuple[str, ...], verbose: bool) -> None:
    """Run a live web-research session."""
    question = " ".join(query)
    run_research_agent(question, verbose=verbose)

@cli.command()
@click.argument("site")
@click.option("--pages", default=40, show_default=True, help="Max pages to crawl")
def crawl(site: str, pages: int) -> None:
    """
    Warm-up crawler: pre-index an entire site into RAM so Recall is instant.
    """
    mem = Memory()
    click.echo(f"Crawling {site} …")
    pages_dict = crawl_site(site, max_pages=pages)
    for url, html in pages_dict.items():
        mem.add_web_content(url, html)
    click.echo(click.style(f"✓ Stored {len(pages_dict)} pages.", fg="green"))

if __name__ == "__main__":
    cli()