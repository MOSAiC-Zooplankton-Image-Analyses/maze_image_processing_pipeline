import contextlib
import fnmatch
import glob
import json
import logging
import os
import sys
import textwrap
from typing import Any, Collection, Dict, List, Mapping, Sequence, Tuple

import exceptiongroup
import natsort
import numpy as np
import pandas as pd
import polytaxo
import pydantic
import pyecotaxa.archive
import scipy.ndimage as ndi
import skimage
import skimage.color.colorlabel
import skimage.measure
import skimage.util
import torch
import torchvision.transforms.functional as tvtf
import yaml
from morphocut.batch import BatchedPipeline
from morphocut.contrib.ecotaxa import EcotaxaReader, EcotaxaWriter
from morphocut.core import Call, Pipeline, StreamObject, Variable
from morphocut.hdf5 import HDF5Writer
from morphocut.pipelines import DataParallelPipeline
from morphocut.stream import Filter
from morphocut.stream import Progress as LiveProgress
from morphocut.stream import Slice, Unpack
from morphocut.tiles import TiledPipeline
from morphocut.torch import PyTorch
from skimage.measure._regionprops import RegionProperties

from ..common import add_note, convert_img_dtype, recursive_update
from ..pipeline_runner import PipelineRunner
from .config_schema import ModelMetaSchema, PredictionPipelineConfig

logging.captureWarnings(True)
logger = logging.getLogger(__name__)

del exceptiongroup


def _find_files_glob(pattern: str, ignore_patterns: Collection | None = None):
    for fn in glob.iglob(pattern):
        if ignore_patterns is not None and any(
            fnmatch.fnmatch(fn, pat) for pat in ignore_patterns
        ):
            logger.info(f"Ignoring {fn}.")
            continue

        yield fn


def measure_segments(
    meta: Dict[str, Any],
    image: np.ndarray,
    probabilities: np.ndarray,
    channel_names: Sequence[str],
    draw: bool,
    fill_holes: bool | Sequence[str] = False,
    _properties=["area", "axis_major_length", "area_convex"],
) -> Tuple[Mapping[str, Any], List]:

    # Filter metadata keys to make sure that the archive is importable later
    meta = {
        k: v
        for k, v in meta.items()
        if k.split("_", maxsplit=1)[0] in pyecotaxa.archive.VALID_PREFIXES
    }

    # Convert probabilities to predictions
    predictions = (probabilities > 0.5).astype(bool)

    assert predictions.ndim == 3
    assert predictions.shape[-1] == len(channel_names)

    # Calculate raw area (without hole-filling and component selection)
    for c, channel_name in enumerate(channel_names):
        meta[f"object_{channel_name}_raw_area"] = predictions[..., c].sum()

    # Fill holes
    if fill_holes:
        for c, channel_name in enumerate(channel_names):
            if fill_holes is True or channel_name in fill_holes:
                # Use ndi.find_objects to narrow down the area to be processed by ndi.binary_fill_holes
                for slices in ndi.find_objects(predictions[..., c], 1):
                    if slices is None:
                        continue
                    ndi.binary_fill_holes(
                        predictions[..., c][slices], output=predictions[..., c][slices]
                    )

    # Keep only largest segment
    channel_props = {}
    for c, channel_name in enumerate(channel_names):
        labels = skimage.measure.label(predictions[..., c])
        regions: List[RegionProperties] = skimage.measure.regionprops(labels)
        if regions:
            regions.sort(key=lambda r: r.area, reverse=True)
            channel_props[channel_name] = props = regions[0]
            # Update predictions
            predictions[..., c] = props._label_image == props.label
        else:
            channel_props[channel_name] = None

    # Draw segments
    if draw:
        annotations = np.zeros(predictions.shape[:-1], dtype=int)
        for c in range(predictions.shape[-1]):
            annotations[predictions[..., c]] = c + 1

        colors = [
            skimage.color.colorlabel._rgb_vector(c)
            for c in skimage.color.colorlabel.DEFAULT_COLORS
        ]

        try:
            annotated_image = skimage.color.colorlabel.label2rgb(
                annotations, image, alpha=0.3, saturation=1, bg_color=None
            )
        except Exception as exc:
            add_note(
                exc,
                f"predictions.shape: {predictions.shape}, annotations.shape: {annotations.shape}, image.shape: {image.shape}",
            )
            raise
    else:
        annotated_image = None
        colors = None

    for c, channel_name in enumerate(channel_names):
        props = channel_props[channel_name]

        if props is None:
            for prop in _properties:
                meta[f"object_{channel_name}_{prop}"] = 0
            meta[f"object_{channel_name}_area_convex_ratio"] = 0
        else:
            for prop in _properties:
                meta[f"object_{channel_name}_{prop}"] = getattr(props, prop)

            meta[f"object_{channel_name}_area_convex_ratio"] = (
                (props.area / props.area_convex) if props.area_convex else 0
            )

            if annotated_image is not None:
                centroid_r, centroid_c = props.centroid
                vr = np.cos(props.orientation) * 0.5 * props.axis_major_length
                r0, r1 = centroid_r + vr, centroid_r - vr
                vc = np.sin(props.orientation) * 0.5 * props.axis_major_length
                c0, c1 = centroid_c + vc, centroid_c - vc

                max_r = annotated_image.shape[0] - 1
                max_c = annotated_image.shape[1] - 1

                rr, cc, val = skimage.draw.line_aa(
                    round(r0.clip(0, max_r)),
                    round(c0.clip(0, max_c)),
                    round(r1.clip(0, max_r)),
                    round(c1.clip(0, max_c)),
                )
                annotated_image[rr, cc] = (
                    val[..., None] * colors[c] + (1 - val[..., None]) * annotated_image[rr, cc]  # type: ignore
                )

    return meta, (
        []
        if annotated_image is None
        else [
            (
                meta["object_id"] + "_overlay.jpg",
                skimage.util.img_as_ubyte(annotated_image),
            )
        ]
    )


def _prepare_translation(
    ecotaxa_taxonomy_fn: str,
    poly_taxonomy: polytaxo.PolyTaxonomy,
):
    ecotaxa_taxonomy = pd.read_csv(ecotaxa_taxonomy_fn, index_col=False)

    def _parse_lineage(lineage: str) -> pd.Series:
        parts = lineage.split(">")
        n_parts = len(parts)
        try:
            description = poly_taxonomy.get_description(
                parts, ignore_missing_intermediaries=True, with_alias=True
            )
        except ValueError as exc:
            logger.warn(f"Could not parse lineage '{lineage}': {exc}")
            return pd.Series([None, n_parts])

        return pd.Series([description, n_parts])

    ecotaxa_taxonomy[["polytaxo_description_obj", "lineage_depth"]] = ecotaxa_taxonomy[
        "lineage"
    ].apply(
        _parse_lineage  # type: ignore
    )  # type: ignore

    # Filter out unparseable lineage
    ecotaxa_taxonomy = ecotaxa_taxonomy[
        ~pd.isna(ecotaxa_taxonomy["polytaxo_description_obj"])
    ]

    # I. Prepare *forward* translation (EcoTaxa => PolyTaxo)
    display_name_to_description = ecotaxa_taxonomy.set_index("display_name", drop=True)

    # II. Prepare *backward* translation (PolyTaxo => EcoTaxa)
    description_to_display_name = ecotaxa_taxonomy.copy()
    description_to_display_name["polytaxo_description"] = description_to_display_name[
        "polytaxo_description_obj"
    ].map(str)
    description_to_display_name = description_to_display_name

    # Remove wildcard descriptors
    wildcard_mask = description_to_display_name["polytaxo_description_obj"].map(
        lambda description: any(
            (
                any("*" in a for a in d.alias)
                if isinstance(d, polytaxo.PrimaryNode)
                else False
            )
            for d in description.descriptors
        )
    )
    description_to_display_name = description_to_display_name[~wildcard_mask]

    # Keep the shallowest category for each polytaxo_description
    description_to_display_name = description_to_display_name.sort_values(
        ["polytaxo_description", "lineage_depth"]
    ).drop_duplicates("polytaxo_description", keep="first")

    description_to_display_name = description_to_display_name.set_index(
        "polytaxo_description", drop=True
    )

    return display_name_to_description, description_to_display_name


def build_polytaxo_pipeline(
    config: PredictionPipelineConfig, meta: Variable, probabilites: Variable
):
    assert config.polytaxo is not False

    logger.info(
        f"Predicting object properties using PolyTaxonomy {config.polytaxo.poly_taxonomy_fn}."
    )

    with open(config.polytaxo.poly_taxonomy_fn, "r") as f:
        poly_taxonomy_dict = yaml.safe_load(f)

        if not isinstance(poly_taxonomy_dict, dict):
            raise ValueError(
                f"Unexpected content in {config.polytaxo.poly_taxonomy_fn}: {poly_taxonomy_dict}"
            )

    poly_taxonomy = polytaxo.PolyTaxonomy.from_dict(poly_taxonomy_dict)

    logger.info(poly_taxonomy.format_tree())

    logger.info(f"Using EcoTaxa taxonomy {config.polytaxo.ecotaxa_taxonomy_fn}")
    display_name_to_description, description_to_display_name = _prepare_translation(
        config.polytaxo.ecotaxa_taxonomy_fn, poly_taxonomy
    )

    if config.polytaxo.taxonomy_augmentation_rules is not None:
        taxonomy_augmentation_rules = [
            (
                poly_taxonomy.parse_expression(query),
                poly_taxonomy.parse_expression(update),
            )
            for query, update in config.polytaxo.taxonomy_augmentation_rules.items()
        ]
    else:
        taxonomy_augmentation_rules = None

    if config.polytaxo.prediction_constraint_rules is not None:
        prediction_constraint_rules = [
            (
                poly_taxonomy.parse_expression(query),
                poly_taxonomy.parse_expression(update),
            )
            for query, update in config.polytaxo.prediction_constraint_rules.items()
        ]
    else:
        prediction_constraint_rules = None

    if config.polytaxo.filter_validated is not None:
        filter_validated = poly_taxonomy.parse_expression(
            config.polytaxo.filter_validated
        )
    else:
        filter_validated = None

    def _update_meta(
        meta: Dict,
        probabilities,
        *,
        compatible_predictions_only=config.polytaxo.compatible_predictions_only,
        filter_validated=filter_validated,
        threshold=config.polytaxo.threshold,
        threshold_relative=config.polytaxo.threshold_relative,
        taxonomy_augmentation_rules=taxonomy_augmentation_rules,
        prediction_constraint_rules=prediction_constraint_rules,
        display_name_to_description=display_name_to_description,
        description_to_display_name=description_to_display_name,
        skip_unchanged_objects=config.polytaxo.skip_unchanged_objects,
    ):
        if (
            compatible_predictions_only
            and meta.get("object_annotation_status", "") == "validated"
        ):
            description_prev = display_name_to_description.at[
                meta["object_annotation_category"], "polytaxo_description_obj"
            ]

            # Skip objects that don't match the filter
            if filter_validated is not None and not filter_validated.match(
                description_prev
            ):
                return None

            # Apply taxonomy augmentation rules
            if taxonomy_augmentation_rules is not None:
                for query, update in taxonomy_augmentation_rules:
                    if query.match(description_prev):
                        description_prev = update.apply(description_prev)
        else:
            description_prev = None

        description: polytaxo.Description = poly_taxonomy.parse_probabilities(
            probabilities,
            baseline=description_prev,
            thr_pos_abs=threshold,
            thr_neg=1 - threshold,
            thr_pos_rel=threshold_relative,
        )

        # Exclude descriptors with predict==False
        _descriptors = (
            (
                q
                if (
                    not isinstance(q, (polytaxo.TagNode, polytaxo.PrimaryNode))
                    or (q.meta.get("predict", True))
                )
                else q.parent
            )
            for q in description.descriptors
        )
        # Rebuild description to avoid duplicate qualifiers (because we use q.parent)
        description = polytaxo.Description(poly_taxonomy.root).update(
            d for d in _descriptors if d is not None
        )
        del _descriptors

        # Apply prediction constraint rules
        if prediction_constraint_rules is not None:
            for query, update in prediction_constraint_rules:
                if query.match(description):
                    description = update.apply(description)

        # Re-add previous description in case a rule erased a previously validated annotation
        if description_prev is not None:
            description.add(description_prev)

        # If cut or multiple, no other qualifiers shall be used
        for q in description.qualifiers:
            if isinstance(q, (polytaxo.PrimaryNode, polytaxo.TagNode)) and q.meta.get(
                "singleton", False
            ):
                description.qualifiers = [q]
                break

        # Drop negated qualifiers
        description.qualifiers = [
            q
            for q in description.qualifiers
            if not isinstance(q, polytaxo.NegatedRealNode)
        ]

        # Lookup display_name for the description
        try:
            display_name = description_to_display_name.at[
                str(description), "display_name"
            ]
        except KeyError as exc:
            # If the name is not found, suggest creating a new category

            # Create a description that only includes the qualifiers (not the exact anchor)
            qualifier_description = polytaxo.Description(poly_taxonomy.root).update(
                description.qualifiers
            )

            # # TODO: Select next best
            # applicable_virtuals = description.anchor.get_applicable_virtuals()
            # if description_prev is not None:
            #     qualifier_description_prev = polytaxo.PolyDescription(
            #         poly_taxonomy.root
            #     ).update(description_prev.qualifiers)
            #     # Select only virtuals that imply (are an extension of) the previous annotation
            #     applicable_virtuals = [
            #         v
            #         for v in applicable_virtuals
            #         if qualifier_description_prev <= v.description
            #     ]
            # # Calculate match score
            # matching_virtuals = [
            #     (v, sum(1 for d in qualifier_description.descriptors if d <= ))
            #     for v in applicable_virtuals
            # ]

            matching_virtual = next(
                (
                    virtual
                    for virtual in description.anchor.get_applicable_virtuals()
                    if virtual.description == qualifier_description
                ),
                None,
            )
            if matching_virtual is not None:
                msg = f"Consider creating '{description.anchor.name}>{matching_virtual.name}' on EcoTaxa."
            else:
                msg = f"Consider creating an appropriate morpho-taxon on EcoTaxa and adding it to the list of virtuals."

            if meta.get("object_annotation_status", "") == "validated":
                msg += f"\nOriginal description was: {description_prev} ({meta['object_annotation_category']})"

            msg = textwrap.indent(msg, "  ")

            logger.error(
                f"Could not find description in EcoTaxa taxonomy: {exc}\n{msg}"
            )

            # Keep object_annotation_category unchanged
            display_name = meta["object_annotation_category"]

        if meta["object_annotation_category"] == display_name:
            # Don't include in the output if we didn't change anything
            if skip_unchanged_objects:
                return None
        else:
            meta.update(
                object_annotation_category=display_name,
                object_annotation_status="predicted",
            )

        meta = {
            k: v
            for k, v in meta.items()
            if k
            in {
                "object_id",
                "object_annotation_category",
                "object_annotation_status",
            }
        }

        return meta

    meta = Call(_update_meta, meta, probabilites)
    Filter(meta)
    return meta


class Runner(PipelineRunner):
    @staticmethod
    def _configure_and_run(config_dict):
        try:
            config = PredictionPipelineConfig.model_validate(config_dict)
        except pydantic.ValidationError as exc:
            logger.error(str(exc))
            return

        if sys.stdout.isatty():
            Progress = LiveProgress
        else:
            from functools import partial

            from ..log_progress import LogProgress

            log_interval = config.log_interval
            if isinstance(log_interval, str):
                log_interval = pd.Timedelta(log_interval).total_seconds()

            Progress = partial(LogProgress, log_interval=log_interval)

        os.makedirs(config.target_dir, exist_ok=True)

        with Pipeline() as p:
            process_meta_var = Variable("process_meta", p)
            process_meta = {}

            # Discover input archives
            input_archive_fns = list(
                _find_files_glob(config.input.path, config.input.ignore_patterns)
            )

            logger.info(
                f"Found {len(input_archive_fns):d} input archives in {config.input.path}"
            )

            input_archive_fn = Unpack(
                natsort.natsorted(
                    input_archive_fns, alg=natsort.PATH | natsort.IGNORECASE
                )
            )

            Progress(input_archive_fn)

            predictions_fn = Call(
                lambda input_archive_fn: os.path.join(
                    config.target_dir,
                    os.path.splitext(os.path.basename(input_archive_fn))[0] + ".h5",
                ),
                input_archive_fn,
            )

            measurements_fn = Call(
                lambda input_archive_fn: os.path.join(
                    config.target_dir,
                    os.path.splitext(os.path.basename(input_archive_fn))[0]
                    + ".segmentation.zip",
                ),
                input_archive_fn,
            )

            polytaxo_fn = Call(
                lambda input_archive_fn: os.path.join(
                    config.target_dir,
                    os.path.splitext(os.path.basename(input_archive_fn))[0]
                    + ".polytaxo.zip",
                ),
                input_archive_fn,
            )

            et_obj = EcotaxaReader(
                input_archive_fn,
                # query: RawOrVariable[Optional[str]] = None,
                # prepare_data: Optional[Callable[["pd.DataFrame"], "pd.DataFrame"]] = None,
                # verbose=False,
                # keep_going=False,
                # print_summary=False,
                # encoding="utf-8",
                # index_pattern="*ecotaxa_*",
                # columns: Optional[List] = None,
                # image_default_mode=None,
            )

            image = et_obj.image
            object_id = Call(lambda et_obj: et_obj.meta["object_id"], et_obj)

            if config.input.max_n_objects is not None:
                Slice(config.input.max_n_objects)

            Progress(object_id)

            ###

            extra_files = {"meta.json": None}
            model: torch.jit.ScriptModule = torch.jit.load(
                config.model.model_fn,
                map_location=config.model.device,
                _extra_files=extra_files,
            )
            model_meta_dict: Dict = (
                json.loads(extra_files["meta.json"]) if extra_files["meta.json"] else {}
            )

            # Merge config meta into model meta
            if config.model.meta is not None:
                model_meta_dict = recursive_update(
                    model_meta_dict, config.model.meta.model_dump()
                )

            try:
                model_meta = ModelMetaSchema.model_validate(model_meta_dict)
            except:
                logger.error(
                    f"Could not validate combined model metadata {model_meta_dict!r}"
                )
                raise

            # ((input_name, input_description),) = model_meta.inputs.items()
            ((output_name, output_description),) = model_meta.outputs.items()

            logger.info(model.code)
            # logger.info(f"Input channels '{input_name}': {input_description.channels}")
            logger.info(
                f"Output channels '{output_name}': {output_description.channel_names}"
            )

            # Convert model to the specified dtype
            torch_dtype = getattr(torch, config.model.dtype)
            np_dtype = np.dtype(config.model.dtype)
            model = model.to(torch_dtype)

            def pre_transform(
                img: np.ndarray, _center_crop=not config.model.tiling
            ) -> torch.Tensor:
                """Ensure RGB image, convert to specified dtype and transpose."""
                if img.ndim == 2:
                    img = skimage.color.gray2rgb(img)

                img = img.transpose((2, 0, 1))

                img = convert_img_dtype(img, np_dtype)

                tensor = torch.from_numpy(img)

                if _center_crop:
                    # Extract center 1024x1024 window (or pad)
                    tensor = tvtf.center_crop(tensor, 1024)

                return tensor.contiguous()

            def post_transform(predictions: torch.Tensor) -> np.ndarray:
                # Convert to channel last
                return predictions.cpu().movedim(1, -1).numpy()

            with contextlib.ExitStack() as context_stack:
                if config.model.tiling is not False:
                    context_stack.enter_context(
                        TiledPipeline(
                            (config.model.tiling.size, config.model.tiling.size),
                            image,
                            tile_stride=(
                                config.model.tiling.stride,
                                config.model.tiling.stride,
                            ),
                            blend_strategy="linear",
                        )
                    )

                if config.model.batch_size:
                    context_stack.enter_context(
                        BatchedPipeline(config.model.batch_size)
                    )
                    is_batch = True
                else:
                    is_batch = False

                if config.model.n_threads > 1:
                    context_stack.enter_context(
                        DataParallelPipeline(executor=config.model.n_threads)
                    )

                predictions = PyTorch(
                    model,
                    image,
                    device=config.model.device,
                    is_batch=is_batch,
                    # output_key=config.model.output_key,
                    # pin_memory=None,
                    pre_transform=pre_transform,
                    post_transform=post_transform,
                    # autocast=False,
                )

            if config.save_raw_h5:
                h5_mode_create = config.model.tiling
                HDF5Writer(
                    predictions_fn,
                    (
                        [(object_id, predictions)]
                        if h5_mode_create
                        else [("object_id", object_id), ("predictions", predictions)]
                    ),
                    dataset_mode="create" if h5_mode_create else "append",
                    compression="gzip",
                )

            if config.segmentation:
                if not config.model.tiling:
                    logger.warning(
                        "Segmentation is requested but tiling is not enabled."
                    )

                if output_description.channel_names is None:
                    raise ValueError(f"Supply channel_names for output '{output_name}'")

                meta, fnames_images = Call(
                    measure_segments,
                    et_obj.meta,
                    image,
                    predictions,
                    output_description.channel_names,
                    config.segmentation.draw,
                    config.segmentation.fill_holes,
                ).unpack(2)

                EcotaxaWriter(measurements_fn, fnames_images, meta=meta)

            if config.polytaxo is not False:
                # Extract relevant metadata
                meta = Call(
                    lambda et_obj: {
                        k: v
                        for k, v in et_obj.meta.items()
                        if k
                        in {
                            "object_id",
                            "object_annotation_status",
                            "object_annotation_category",
                            "object_annotation_hierarchy",
                        }
                    },
                    et_obj,
                )
                meta = build_polytaxo_pipeline(config, meta, predictions)
                EcotaxaWriter(polytaxo_fn, [], meta=meta)

        # Inject pipeline metadata into the stream
        obj = StreamObject(n_remaining_hint=1)
        obj[process_meta_var] = process_meta
        p.run(iter([obj]))
