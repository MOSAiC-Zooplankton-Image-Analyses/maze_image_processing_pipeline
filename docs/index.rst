Welcome to MAZE-IPP's documentation!
====================================

The MAZE Image Processing Pipeline (MAZE-IPP) is a comprehensive tool for advanced marine biological research, integrating three specialized modules to enhance zooplankton imagery analysis.
It is used via a command-line interface (CLI).
The first module (`maze-ipp loki`) resegments LOKI (Lightframe On-sight Keyspecies Investigation) data for precise organism delineation.
The second module (`maze-ipp segmseg`) targets semantic segmentation of calanoid copepods.
The third module (`maze-ipp polytaxo`) provides fine-grained polyhierarchical identification of calanoid copepods and chaetognatha, detailing orientation, health, life stage, and sex.

MAZE-IPP interfaces with `MorphoCluster <https://github.com/morphocluster/morphocluster/>`_ and `EcoTaxa <https://ecotaxa.obs-vlfr.fr/>`_, two image annotation platforms.
MorphoCluster's clustering algorithms accelerate the annotation of images.
Segmented and classified images are exported to EcoTaxa for further validation and curation.
Interfacing with EcoTaxa is implemented using `pyecotaxa <https://github.com/ecotaxa/pyecotaxa>`_.


.. graphviz:: flowchart.dot
   :caption: Interaction of the pipeline components within the EcoTaxa ecosystem.


.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:
   
   installation
   loki
   morphocluster
   ecotaxa
