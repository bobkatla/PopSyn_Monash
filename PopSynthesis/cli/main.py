import click

@click.group()
def cli():
    """A command-line interface for the PopSynthesis project."""
    pass

@cli.command()
def hello():
    """Prints a greeting."""
    click.echo("Hello, World!")
