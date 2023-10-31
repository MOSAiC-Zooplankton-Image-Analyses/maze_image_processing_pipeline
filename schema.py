from typing import Mapping
from marshmallow import Schema, fields, pre_load


class DefaultSchema(Schema):
    __default_field__: str

    @pre_load
    def _convert_default(self, value, **_):
        if isinstance(value, Mapping):
            return value
        return {self.__default_field__: value}


class SegmentationPostprocessing(Schema):
    closing_radius = fields.Int(load_default=0)
    opening_radius = fields.Int(load_default=0)
    merge_labels = fields.Int(load_default=0)
    min_area = fields.Int(load_default=0)
    n_threads = fields.Int(load_default=0)


class ThresholdSegmentation(DefaultSchema):
    __default_field__ = "threshold"
    threshold = fields.Number()


class StoredSegmentation(DefaultSchema):
    __default_field__ = "pickle_fn"
    pickle_fn = fields.Str()
    full_frame_archive_fn = fields.Str(load_default=None)
    skip_single = fields.Bool(load_default=False)


class PytorchSegmentation(DefaultSchema):
    __default_field__ = "model_fn"
    model_fn = fields.Str(required=False)
    jit_model_fn = fields.Str(required=False)
    full_frame_archive_fn = fields.Str(load_default=None)
    skip_single = fields.Bool(load_default=False)
    device = fields.Str(load_default="cpu")
    stitch = fields.Bool(load_default=True)
    postprocess = fields.Nested(SegmentationPostprocessing, required=False)
    n_threads = fields.Int(load_default=0)
    batch_size = fields.Int(load_default=0)
    autocast = fields.Bool(load_default=False)
    dtype = fields.Str(load_default="float32")
    apply_mask = fields.Bool(load_default=False)
    min_intensity = fields.Int(load_default=None)


class SegmentationSchema(Schema):
    threshold = fields.Nested(ThresholdSegmentation, required=False)
    stored = fields.Nested(StoredSegmentation, required=False)
    pytorch = fields.Nested(PytorchSegmentation, required=False)


class GlobInputSchema(DefaultSchema):
    __default_field__ = "pattern"
    pattern = fields.Str()


class LokiInputSchema(Schema):
    glob = fields.Nested(GlobInputSchema, required=False)
    # Process only this many objects
    slice = fields.Int(required=False)
    meta = fields.Dict(required=False)
    filter_object_frame_id = fields.Str(load_default=None)
    ignore_patterns = fields.List(fields.Str, required=False)
    merge_telemetry = fields.Bool(load_default=True)


class InputSchema(Schema):
    loki = fields.Nested(LokiInputSchema)


class EcoTaxaOutputSchema(Schema):
    path = fields.Str()
    image_fn = fields.Str(required=False)
    scalebar = fields.Bool(load_default=False)


class OutputSchema(Schema):
    ecotaxa = fields.Nested(EcoTaxaOutputSchema)


class PipelineSchema(Schema):
    input = fields.Nested(LokiInputSchema, required=True)
    segmentation = fields.Nested(SegmentationSchema)
    output = fields.Nested(EcoTaxaOutputSchema, required=True)


if __name__ == "__main__":
    import sys
    import yaml

    with open(sys.argv[1]) as f:
        config = PipelineSchema()
        x = config.load(yaml.safe_load(f))

        print(x)
