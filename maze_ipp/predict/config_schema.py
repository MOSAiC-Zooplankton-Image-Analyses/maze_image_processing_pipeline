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
        None, description="Maximum number of objects. (For debugging.)"
    )


class DataDescriptorSchema(BaseModel):
    channels: Sequence[str] = Field(description="")

    model_config = ConfigDict(
        extra="allow",
    )


class ModelMetaSchema(BaseModel):
    # inputs: OrderedDict[str, DataDescriptorSchema] = Field(
    #     description="Ordered mapping of input names to input descriptions."
    # )
    outputs: OrderedDict[str, DataDescriptorSchema] = Field(
        description='Ordered mapping of output names to output descriptions, e.g. {"pred": ["Prosoma", "Oilsack"]}. Only a single output is supported.'
    )

    model_config = ConfigDict(
        extra="allow",
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

    tiled: bool = Field(
        False,
        description="Apply the model to square tiles on each input image. Required for semantic segmentation.",
    )


class SegmentationConfig(TrueToDefaultsModel):
    draw: bool = Field(False, description="Draw segments.")


class PredictionPipelineConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    input: EcoTaxaInputConfig = Field(description="Configuration of the input.")
    model: ModelConfig = Field(description="Configuration of the input.")

    save_raw_predictions: bool = Field(
        False,
        description="Save raw predictions into an HDF5 file, e.g. for feature extraction or polytaxo classification.",
    )
    segmentation: SegmentationConfig | Literal[False] = Field(
        None,
        description="Measure predicted segments and store into EcoTaxa archive. (Only applies for semantic segmentation.)",
    )

    target_dir: str = Field(description="Directory where the output files are created.")
