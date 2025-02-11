LOKI Resegmentation
===================

The `maze-ipp loki` command implements an image processing pipeline for the resegmentation of raw data captured by the LOKI imaging system.

It provides the following features:

- Sample folder discovery
- Merging of telemetry metadata.
- Segmentation using thresholding or a deep learning model.
- Duplicate Detection
- Merging of existing EcoTaxa annotations
- Generation of import-ready EcoTaxa archives
- Logging and error handling
- Progress reporting
- `YAML <https://en.wikipedia.org/wiki/YAML>`_ configuration.

Sample folder discovery
-----------------------

By default, the input :attr:`~maze_ipp.loki.config_schema.LokiInputConfig.path` is searched for valid sample folders.
Sample folders are recognized if they contain the subfolders `"Telemetrie"` and `"Pictures"`.
This sample folder discovery can be disabled by setting :attr:`~maze_ipp.loki.config_schema.LokiInputConfig.discover` to `false`.

Configuration
-------------

Here is the example configuration:

.. program-output:: maze-ipp config loki

The example configuration can be generated using the `maze-ipp config loki` command.


Configuration Schema
~~~~~~~~~~~~~~~~~~~~

This is the complete documentation for the configuration of the pipeline:

.. automodule:: maze_ipp.loki.config_schema
   :exclude-members: DefaultModel, TrueToDefaultsModel
   :members:


Merging existing annotations
----------------------------

In the case of reprocessing, it is sometimes necessary to merge existing annotations into the new dataset.
These can be extracted from an EcoTaxa export using the `pyecotaxa extract-meta` helper command:

   pyecotaxa extract-meta --fix-bbox LOKI <INPUT_ARCH_FN> <OUTPUT_TSV_FN>
