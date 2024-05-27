import click
import maze_ipp


@click.group()
@click.version_option(version=maze_ipp.__version__)
def main():
    pass
