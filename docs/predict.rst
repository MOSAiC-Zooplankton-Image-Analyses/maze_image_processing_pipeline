Prediction
==========

Semantic segmentation
---------------------

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

Polyhierarchical classification is implement as a two-step process: predicting scores for each image and generating polyhierarchical descriptions.

.. code-block:: python
    
    maze-ipp predict polytaxo.yaml

`polytaxo.yaml` is configured in the following way:

.. TODO: Link model file

:attr:`~maze_ipp.predict.config_schema.ModelConfig.model_fn` must point to a trained polytaxo classifier model.

To generate polyhierarchical descriptions and, optionally, to map them back to EcoTaxa categories, the `polytaxo <https://github.com/MOSAiC-Zooplankton-Image-Analyses/polytaxo>`_ library can be used.


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
