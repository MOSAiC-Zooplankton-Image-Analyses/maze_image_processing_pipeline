import click
import maze_ipp


@click.group()
@click.version_option(version=maze_ipp.__version__)
def cli():
    pass


@cli.command()
@click.argument("task_fn", type=click.Path(exists=True))
def loki(task_fn):
    """LOKI (re-)segmentation pipeline."""

    from maze_ipp.loki.pipeline import main

    main(task_fn)


@cli.command()
def loki_config():
    """Generate default configuration for the LOKI (re-)segmentation pipeline."""

    from maze_ipp.loki.config_schema import generate_yaml_example

    generate_yaml_example()
