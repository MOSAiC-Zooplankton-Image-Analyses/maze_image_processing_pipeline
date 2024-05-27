from textwrap import indent, wrap
from pydantic_core import PydanticUndefined
from pydantic.fields import FieldInfo
from pydantic import BaseModel
from typing import get_origin, get_args, Type, Tuple, Union, Literal
from types import UnionType, NoneType
import json


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
