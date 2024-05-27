from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Literal,
    Mapping,
    Tuple,
    Type,
    Union,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    model_validator,
    Field,
)


class DefaultModel(BaseModel):
    __default_field__: ClassVar[str]

    @model_validator(mode="before")
    @classmethod
    def parse_shortform(cls, data: Any):
        if not isinstance(data, Mapping):
            return {cls.__default_field__: data}
        return data


class TrueToDefaultsModel(BaseModel):
    @model_validator(mode="before")
    @classmethod
    def parse_shortform(cls, data: Any):
        if data is True:
            return {}
        return data


class SegmentationPostprocessingConfig(TrueToDefaultsModel):
    closing_radius: int = Field(
        0,
        description="Apply morphological closing (close small gaps) using this radius.",
    )
    opening_radius: int = Field(
        0,
        description="Apply morphological opening (remove small objects) using this radius.",
    )
    merge_segments_distance: int = Field(
        0,
        description="Merge segments closer than the specified distance.",
    )
    min_area: int = Field(
        0, description="Remove objects with an area below the specified threshold."
    )
    n_threads: int = Field(
        0, description="Use multiple threads to perform the post-processing."
    )
    clear_border: bool = Field(
        False, description="Clear objects touching the image border."
    )


class ThresholdSegmentationConfig(DefaultModel):
    __default_field__ = "threshold_brighter"

    threshold_brighter: float = Field(
        ..., description="Extract objects brighter than this threshold."
    )


class StitchConfig(TrueToDefaultsModel):
    skip_single: bool = Field(
        False, description="Remove stitched frames with only one object (debug)."
    )


class PytorchSegmentationConfig(DefaultModel):
    __default_field__ = "model_fn"

    model_config = ConfigDict(protected_namespaces=())

    # Stitching
    stitch: StitchConfig | Literal[False] = Field(
        True, description="Stitch objects to reconstruct frames."
    )

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

    # Post-processing
    postprocess: SegmentationPostprocessingConfig | Literal[False] = Field(
        False, description="Perform full-frame post-processing steps."
    )

    full_frame_archive_fn: str | None = Field(
        None,
        description="Write segmented full-frames to this file in the target directory (debug).",
    )

    # Settings for ExtractROI
    padding: int = Field(
        75,
        description="Pad extracted regions with this number of pixels on each border.",
    )
    min_intensity: int = Field(
        None, description="Minimum intensity of extracted regions."
    )
    apply_mask: bool = Field(
        False,
        description="Hide everything in a vignette that is not part of current object.",
    )
    background_color: Any = Field(
        0,
        description="Color for the background when hiding foreign object parts. Can be a scalar (`0`), a tuple (`(r,g,b)=(255,0,0)`), a color name (`'black'`) or a quantile (`'quantile:0.25'`).",
    )
    keep_background: bool = Field(
        True, description="When hiding non-object image regions, keep background."
    )


class SegmentationConfig(BaseModel):
    # Segmentation
    threshold: ThresholdSegmentationConfig | None = Field(
        None, description="Use thresholding for segmentation."
    )
    pytorch: PytorchSegmentationConfig | None = Field(
        None, description="Use a PyTorch model for segmentation."
    )

    @model_validator(mode="after")
    def parse_shortform(self):
        if (self.threshold is None and self.pytorch is None) or (
            self.threshold is not None and self.pytorch is not None
        ):
            raise ValueError("Exactly one of threshold and pytorch must be configured.")

        return self

    # Filtering
    filter_expr: str | None = Field(
        None, description="Filter objects by Python expression."
    )


class DetectDuplicatesConfig(BaseModel):
    min_similarity: float = Field(
        0.98, description="Minimum similarity of two objects."
    )
    max_age: int = Field(1, description="Maximum age of a previous object.")


DetectDuplicatesModelOrFalse = DetectDuplicatesConfig | Literal[False]


class MergeTelemetryConfig(BaseModel):
    # A string denoting a time difference (e.g. 5 minutes: "5m")
    tolerance: str | None = Field(
        default=None,
        description="Maximum delta between object time and telemetry time.",
    )


class LokiInputConfig(BaseModel):
    path: str = Field(
        description="Path to a LOKI input directory. May contain wildcard characters ('?', '*')."
    )
    discover: bool = Field(
        True,
        description="Try to discover all LOKI samples inside the specified path "
        "by looking for directories that contain 'Pictures' and 'Telemetrie' folders.",
    )
    ignore_patterns: List[str] = Field(
        [],
        description="Ignore these directories. May contain wildcard characters ('?', '*').",
    )

    filter_expr: str | None = Field(
        None, description="Filter input objects by Python expression."
    )

    slice: int | None = Field(
        None, description="Process only this many objects (for debugging)."
    )

    default_meta: Dict = Field({}, description="Default metadata for all objects.")
    valid_frame_id_fn: str | None = Field(
        None,
        description="Location of a file containing all valid `object_frame_id` s. "
        "Frames not in this file will be skipped. (Optional.)",
    )
    merge_telemetry: MergeTelemetryConfig | Literal[False] = Field(
        default_factory=MergeTelemetryConfig,
        description="Merge telemetry. (Default: true)",
    )
    save_meta: bool = Field(
        False,
        description="Save calculated input metadata in the target directory (for debugging).",
    )

    # Detect duplicates
    detect_duplicates: DetectDuplicatesModelOrFalse = Field(
        False, description="Detect duplicates. (Default: false)"
    )


class MergeAnnotationsConfig(DefaultModel):
    __default_field__ = "annotations_fn"

    annotations_fn: str = Field(
        description="EcoTaxa TSV file containing annotations for objects."
    )
    min_overlap: float | None = Field(
        None,
        description="Minimum overlap of object and annotation bounding box in IoU.",
    )
    min_validated_overlap: float | None = Field(
        None,
        description="Minimum overlap of object and annotation bounding so that the resulting annotation_status remains 'validated'.",
    )


class ScalebarConfig(BaseModel):
    px_per_mm: float = Field(description="Pixels per millimeter.")


class PostprocessingConfig(BaseModel):
    scalebar: ScalebarConfig | None = Field(
        None, description="Draw a scalebar on each object image."
    )

    slice: int | None = Field(
        None, description="Process only this many objects (for debugging)."
    )

    filter_expr: str | None = Field(
        None, description="Filter objects by Python expression."
    )

    # Detect duplicates
    detect_duplicates: DetectDuplicatesModelOrFalse = Field(
        False, description="Detect duplicates."
    )

    # Merge annotations
    merge_annotations: MergeAnnotationsConfig | None = Field(
        None, description="Merge annotations."
    )

    rescale_max_intensity: bool = Field(
        False,
        description="Rescale the image intensities so that the brightest value is white.",
    )


class EcoTaxaOutputConfig(BaseModel):
    target_dir: str = Field(
        description="Directory where the EcoTaxa archives are created."
    )
    skip_existing: bool = Field(False, description="Skip if archive already exists.")
    image_fn: str = Field(
        "{object_id}.jpg",
        description="Format string for the names of image files inside the archive. "
        "All fields in metadata can be used.",
    )
    store_mask: bool = Field(
        False, description="Store the mask of each object alongside its image."
    )
    type_header: bool = Field(
        True,
        description="Include a type header in the produced TSV file. "
        "(Required for successful import into EcoTaxa.)",
    )


class SegmentationPipelineConfig(BaseModel):
    input: LokiInputConfig = Field(description="Configuration of the input.")
    segmentation: SegmentationConfig = Field(
        description="Configuration of the segmentation."
    )
    postprocess: PostprocessingConfig = Field(
        description="Configuration of the post-processing."
    )
    output: EcoTaxaOutputConfig = Field(description="Configuration of the output.")


if __name__ == "__main__":
    from textwrap import indent, wrap
    from pydantic_core import PydanticUndefined
    from pydantic.fields import FieldInfo
    from typing import get_origin, get_args
    from types import UnionType, NoneType
    import json

    def generate_yaml_example(model: Type[BaseModel], depth=1) -> str:
        def get_yaml_example_field(name: str, field: FieldInfo) -> Tuple[str, str]:
            if field.annotation is None:
                raise ValueError(f"{name} has no annotation")

            if get_origin(field.annotation) in {Union, UnionType}:
                # Skip NoneType for optional fields
                union_types = [
                    t for t in get_args(field.annotation) if t is not NoneType
                ]

                union_examples = []
                for type_ in union_types:
                    if get_origin(type_) == Literal:
                        union_examples.append(
                            f"# {name}: {json.dumps(get_args(type_)[0])}\n"
                        )
                    elif get_origin(type_) is None and issubclass(type_, BaseModel):
                        union_examples.append(
                            f"# {name}:\n"
                            + indent(generate_yaml_example(type_, depth + 1), "#   "),
                        )
                    else:
                        union_examples.append(f"# {name}: ...\n")

                return (
                    "# ## OR ##\n".join(union_examples),
                    "optional",
                )

            if field.default is not PydanticUndefined:
                return (
                    f"# {name}: {json.dumps(field.default)}",
                    "optional",
                )

            if issubclass(field.annotation, BaseModel):
                return (
                    f"{name}:\n"
                    + indent(
                        generate_yaml_example(field.annotation, depth + 1),
                        "  " * depth,
                    ),
                    "required",
                )

            return f"{name}: ...", "required"

        result = []
        for name, field in model.model_fields.items():
            if field.description is None:
                raise ValueError(f"{name} has no description")

            example, modifier = get_yaml_example_field(name, field)

            result.append(
                indent("\n".join(wrap(f"[{modifier}] {field.description}")), "## ")
            )
            result.append(example)

        result.append("")

        return "\n".join(result)

    print(generate_yaml_example(SegmentationPipelineConfig))
