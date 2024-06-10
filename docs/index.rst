Welcome to MAZE-IPP's documentation!
====================================

The MAZE Image Processing Pipeline (MAZE-IPP) is a comprehensive tool for advanced marine biological research, integrating three specialized modules to enhance zooplankton imagery analysis.
Its command-line interface (CLI) provides two modules:
The first module (`maze-ipp loki`) resegments LOKI (Lightframe On-sight Keyspecies Investigation) data for precise organism delineation.
The second module (`maze-ipp predict`) applys a deep learning model to images for semantic segmentation or classification.

MAZE-IPP interfaces with `MorphoCluster <https://github.com/morphocluster/morphocluster/>`_ and `EcoTaxa <https://ecotaxa.obs-vlfr.fr/>`_, two image annotation platforms.
MorphoCluster's clustering algorithms accelerate the annotation of images using the deep image features calculated by `maze-ipp predict`.
Segmented and classified images are exported to EcoTaxa for further validation and curation.
Fine-grained polyhierarchical identification of calanoid copepods and chaetognatha, detailing orientation, health, life stage, and sex is provided using the `polytaxo <https://github.com/MOSAiC-Zooplankton-Image-Analyses/polytaxo>`_ library.
Interfacing with EcoTaxa is implemented using `pyecotaxa <https://github.com/ecotaxa/pyecotaxa>`_.


.. graphviz:: components.dot
   :caption: Interaction of the pipeline components within the EcoTaxa ecosystem.


.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:
   
   installation
   loki
   predict
   morphocluster