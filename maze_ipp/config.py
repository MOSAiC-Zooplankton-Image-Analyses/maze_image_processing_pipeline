import json
import re
from textwrap import indent, wrap
from types import NoneType, UnionType
from typing import (
    Any,
    ClassVar,
    Literal,
    Mapping,
    Tuple,
    Type,
    Union,
    get_args,
    get_origin,
)

from pydantic import BaseModel, model_validator
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined


def generate_yaml_example(model: Type[BaseModel], depth=1) -> str:
    def get_yaml_example_field(name: str, field: FieldInfo) -> Tuple[str, str]:
        if field.annotation is None:
            raise ValueError(f"{name} has no annotation")

        if get_origin(field.annotation) in {Union, UnionType}:
            # Skip NoneType for optional fields
            union_types = [t for t in get_args(field.annotation) if t is not NoneType]

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

        if isinstance(field.annotation, type) and issubclass(
            field.annotation, BaseModel
        ):
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
        if (field.json_schema_extra is not None) and field.json_schema_extra.get(
            "debug", False
        ):
            continue

        if field.description is None:
            raise ValueError(f"{name} has no description")

        example, modifier = get_yaml_example_field(name, field)

        description = re.sub(
            r":attr:`([^`]*)`",
            lambda m: (
                m.group(1).rsplit(".")[-1] if m.group(1)[0] == "~" else m.group(1)
            ),
            field.description,
            flags=re.MULTILINE,
        )

        for line in f"[{modifier}] {description}".splitlines():
            result.append(indent("\n".join(wrap(line)), "## "))
        result.append(example)

    result.append("")

    return "\n".join(result)


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
