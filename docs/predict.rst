Prediction
==========

The prediction module consumes EcoTaxa archive files and produces different kinds of output according to respective task.

Semantic segmentation
---------------------

The semantic segmentation task applies a PyTorch model for semantic segmentation to each input image
and produces an EcoTaxa archive with the original metadata, the measurements and (optionally) the detected segments
overlayed onto the original images.

.. code-block:: python
    
    maze-ipp predict semantic_segmentation.yaml

`semantic_segmentation.yaml` is configured in the following way:

.. TODO: Link model file

:attr:`~maze_ipp.predict.config_schema.ModelConfig.model_fn` must point to a trained semantic segmentation model.
:attr:`~maze_ipp.predict.config_schema.ModelConfig.tiled` must be `true` so that the model is applied to all image regions.
If :attr:`~maze_ipp.predict.config_schema.SegmentationConfig.draw` is `true`, the detected segments will be stored alongside the measurements.


Feature calculation
-------------------

.. code-block:: python
    
    maze-ipp predict extract_features.yaml

`extract_features.yaml` is configured in the following way:

.. TODO: Link model file

:attr:`~maze_ipp.predict.config_schema.ModelConfig.model_fn` must point to a trained feature extractor model.
:attr:`~maze_ipp.predict.config_schema.PredictionPipelineConfig.save_raw_predictions` must be `true` so that a HDF5 file is created.
The HDF5 file that is created will contain two datasets: `object_id` and `predictions`.


Polyhierarchical classification
-------------------------------

Polyhierarchical classification works by predicting scores for each image, generating polyhierarchical descriptions and mapping them back to EcoTaxa categories.

.. code-block:: python
    
    maze-ipp predict polytaxo.yaml

`polytaxo.yaml` is configured in the following way:

.. TODO: Link model file

:attr:`~maze_ipp.predict.config_schema.ModelConfig.model_fn` must point to a trained polytaxo classifier model.
:attr:`~maze_ipp.predict.config_schema.PredictionPipelineConfig.polytaxo` must be configured.

For more information see the `polytaxo GitHub page <https://github.com/MOSAiC-Zooplankton-Image-Analyses/polytaxo>`_.


Configuration Example
---------------------

Here is some general example configuration:

.. program-output:: maze-ipp config predict

The example configuration can be generated using the `maze-ipp config predict` command.


Configuration Schema
~~~~~~~~~~~~~~~~~~~~

This is the complete documentation for the configuration of the pipeline:

.. automodule:: maze_ipp.predict.config_schema
   :exclude-members: DefaultModel, TrueToDefaultsModel
   :members:
