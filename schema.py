from typing import Callable, Mapping
from marshmallow import Schema, fields, pre_load, post_load, post_dump

class DefaultSchema(Schema):
    __default_field__: str

    @pre_load
    def _convert_default(self, value, **_):
        if isinstance(value, Mapping):
            return value
        return {self.__default_field__: value}

class ThresholdSegmentation(DefaultSchema):
    __default_field__ = "threshold"
    threshold = fields.Number()


class SegmentationSchema(Schema):
    threshold = fields.Nested(ThresholdSegmentation, required=False)


class LokiInputSchema(Schema):
    path = fields.Str()
    segmentation = fields.Nested(SegmentationSchema)


class InputSchema(Schema):
    loki = fields.Nested(LokiInputSchema)

class EcoTaxaOutputSchema(Schema):
    path = fields.Str()

class OutputSchema(Schema):
    ecotaxa = fields.Nested(EcoTaxaOutputSchema)

class PipelineSchema(Schema):
    input = fields.Nested(LokiInputSchema, required=True)
    output = fields.Nested(EcoTaxaOutputSchema, required=True)

if __name__ == "__main__":
    import sys
    import yaml

    with open(sys.argv[1]) as f:
        config = PipelineConfig()
        x = config.load(yaml.safe_load(f))

        print(x)
        
    