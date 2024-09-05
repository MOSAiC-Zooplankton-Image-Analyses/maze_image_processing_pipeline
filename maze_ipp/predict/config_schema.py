from typing import List, Literal, OrderedDict, Sequence
from pydantic import BaseModel, ConfigDict, Field

from ..config import TrueToDefaultsModel


class EcoTaxaInputConfig(BaseModel):
    path: str = Field(
        description="Path to an input EcoTaxa archive. May contain wildcard characters ('?', '*')."
    )
    ignore_patterns: List[str] = Field(
        [],
        description="Ignore these directories. May contain wildcard characters ('?', '*').",
    )
    max_n_objects: int | None = Field(
        None,
        description="Maximum number of objects. (For debugging.)",
        json_schema_extra={"debug": True},
    )


class DataDescriptorSchema(BaseModel):
    channel_names: Sequence[str] | None = Field(
        None, description="List of channel names"
    )

    model_config = ConfigDict(
        extra="allow",
    )


class ModelMetaSchema(BaseModel):
    # inputs: OrderedDict[str, DataDescriptorSchema] = Field(
    #     description="Ordered mapping of input names to input descriptions."
    # )
    outputs: OrderedDict[str, DataDescriptorSchema] = Field(
        description='Ordered mapping of output names to output descriptions, e.g. {"pred": {"channel_names": ["Prosoma", "Oilsack"]}}. Only a single output is supported.'
    )

    model_config = ConfigDict(
        extra="allow",
    )


class TilingConfig(TrueToDefaultsModel):
    size: int = Field(1024, description="Edge length of one tile")
    stride: int = Field(
        896,
        description="Stride of the tiling. `size - stride` is the overlap of two consecutive tiles.",
    )


class ModelConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_fn: str = Field(
        description="A file containing a ScriptModule (or ScriptFunction) previously saved with :func:`torch.jit.save <torch.jit.save>`",
    )

    device: str = Field(
        "cpu",
        description="A device to load and execute the model (e.g. 'cpu' or 'cuda:0').",
    )
    n_threads: int = Field(
        0, description="Number of threads that each execute an instance of the model."
    )
    batch_size: int = Field(0, description="Batch size")
    autocast: bool = Field(
        False,
        description="Enable automatic mixed precision inference to improve performance.",
    )
    dtype: str = Field(
        "float32", description="Datatype to use for the processing (e.g. 'float32')"
    )

    meta: ModelMetaSchema | None = Field(None, description="Model metadata.")

    tiling: TilingConfig | Literal[False] = Field(
        False,
        description="Apply the model to square tiles on each input image. Required for semantic segmentation.",
    )


class SegmentationConfig(TrueToDefaultsModel):
    draw: bool = Field(False, description="Draw segments.")


class PolyTaxoConfig(BaseModel):
    poly_taxonomy_fn: str = Field(description="PolyTaxonomy filename.")
    ecotaxa_taxonomy_fn: str = Field(description="EcoTaxa project taxonomy filename.")
    compatible_predictions_only: bool = Field(
        True,
        description="Update validated object_annotation_category with compatible predictions. "
        "Incompatible predictions will not be added, even if they obtain higher scores than any compatible prediction.\n"
        "If false, the prediction only depends on the model output.",
    )
    skip_unchanged_objects: bool = Field(
        True,
        description="Save only objects with updated annotations and skip unchanged objects.",
    )
    filter_validated: str | None = Field(
        None,
        description="Filter expression to apply to validated objects.\n"
        "Objects not matching this filter are skipped.",
    )
    threshold: float = Field(
        0.9,
        description="Absolute threshold to apply to prediction scores. "
        "Any accepted prediction must obtain a higher score than `threshold`. "
        "If a score is below 1-threshold, a negative descriptor will be added to the description.",
    )
    threshold_relative: float = Field(
        0.0,
        description="Relative threshold to apply to prediction scores. Any accepted prediction must obtain a higher score than the next-best prediction's score + `threshold_relative`.",
    )
    taxonomy_augmentation_rules: OrderedDict[str, str] | None = Field(
        None,
        description="Augmentation rules to apply to previously validated annotations.\n"
        "These rules enrich already validated annotations by incorporating implicit defaults "
        "or taxonomic knowledge that could not be represented in EcoTaxa.\n"
        "These rules have the form `<query>: <update>`: "
        "If the query expression matches the description, the update expression is applied.",
    )
    prediction_constraint_rules: OrderedDict[str, str] | None = Field(
        None,
        description="Constraint rules to apply to predicted annotations.\n"
        "These rules limit or exclude certain predictions based on contextual factors "
        "or known exceptions within the taxonomy. "
        "The purpose is to prevent inaccurate or inappropriate predictions "
        "that do not align with known biological or taxonomic constraints.\n"
        "These rules have the form `<query>: <update>`: "
        "If the query expression matches the description, the update expression is applied.",
    )


class PredictionPipelineConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    input: EcoTaxaInputConfig = Field(description="Configuration of the input.")
    model: ModelConfig = Field(description="Configuration of the input.")

    save_raw_h5: bool = Field(
        False,
        description="Save raw predictions into an HDF5 file, e.g. for feature extraction.",
    )
    segmentation: SegmentationConfig | Literal[False] = Field(
        False,
        description="Measure predicted segments and store into EcoTaxa archive. (Only applies for semantic segmentation.)",
    )
    polytaxo: PolyTaxoConfig | Literal[False] = Field(
        False,
        description="Predict object properties using a PolyTaxo classifier and store into an EcoTaxa archive.",
    )

    target_dir: str = Field(description="Directory where the output files are created.")

    log_interval: str | float = Field(
        "60s", description="The interval at which progress is logged, e.g. 10s or 1m."
    )
