LOKI Resegmentation
===================

The `maze-ipp loki` command implements a image processing pipeline for the resegmentation of raw data captured by the LOKI imaging system.

It provides the following features:

- `YAML <https://en.wikipedia.org/wiki/YAML>`_ configuration.
- Merging of telemetry metadata.
- Segmentation using thresholding or a deep learning model.
- Duplicate Detection
- Merging of existing EcoTaxa annotations
- Generation of import-ready EcoTaxa archives
- Logging and error handling
- Progress reporting

Configuration
-------------

Here is an example configuration file:

.. program-output:: python ../config_schema.py

Configuration Schema
~~~~~~~~~~~~~~~~~~~~

This is the complete documentation for the configuration of the pipeline:

.. automodule:: config_schema
   :exclude-members: DefaultModel, TrueToDefaultsModel
   :members:
