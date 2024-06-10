import click
import maze_ipp


@click.group()
@click.version_option(version=maze_ipp.__version__)
def cli():
    pass


@cli.command()
@click.argument(
    "task_fn",
    type=click.Path(exists=True),
)
def loki(task_fn):
    """LOKI (re-)segmentation pipeline."""

    from maze_ipp.loki.pipeline import Runner

    Runner.run(task_fn)


@cli.command()
@click.argument("module")
def config(module):
    """Generate default configuration."""

    from maze_ipp.gen_config import generate_yaml_example

    if module == "loki":
        from maze_ipp.loki.config_schema import SegmentationPipelineConfig as Schema

    elif module == "predict":
        from maze_ipp.predict.config_schema import PredictionPipelineConfig as Schema

    else:
        raise ValueError(f"Unknown module: {module}")

    print(generate_yaml_example(Schema))


@cli.command()
@click.argument(
    "task_fn",
    type=click.Path(exists=True),
)
def predict(task_fn):
    """Predict images using a PyTorch model."""

    from maze_ipp.predict.pipeline import Runner

    Runner.run(task_fn)
