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

Polyhierarchical classification
-------------------------------

Our polyhierarchical classification system combines a primary phylogenetic classification, such as family, genus, or species, with further positive and negative descriptors.
These descriptors could include attributes, life stages, and behaviors.

The richness of description offered by PolyTaxo allows us to train multi-label classifiers with outputs for each primary (class) and secondary concept (tag).

Polyhierarchical classification is implement as a two-step process: predicting image features and generating polyhierarchical descriptions.

.. code-block:: python
    
    maze-ipp predict polytaxo.yaml

`polytaxo.yaml` is configured in the following way:

.. TODO: Link model file
:attr:`~maze_ipp.predict.config_schema.ModelConfig.model_fn` must point to a trained polytaxo classifier model.

Configuration
-------------

Here is some general example configuration:

.. program-output:: maze-ipp config predict

The example configuration can be generated using the `maze-ipp config predict` command.


Configuration Schema
~~~~~~~~~~~~~~~~~~~~

This is the complete documentation for the configuration of the pipeline:

.. automodule:: maze_ipp.predict.config_schema
   :exclude-members: DefaultModel, TrueToDefaultsModel
   :members:
