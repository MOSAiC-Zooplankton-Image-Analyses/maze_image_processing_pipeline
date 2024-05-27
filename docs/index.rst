Welcome to MAZE-IPP's documentation!
====================================

The MAZE Image Processing Pipeline (MAZE-IPP) is a comprehensive tool designed for advanced marine biological research.
It integrates three specialized modules to enhance the analysis of marine zooplankton imagery.
The first module focuses on the resegmentation of LOKI (Lightframe On-sight Keyspecies Investigation) data, ensuring precise delineation of captured organisms.
The second module performs semantic segmentation, targeting calanoid copepods.
The third module providing fine-grained polyhierarchical identification of both calanoid copepods and chaetognatha, offering detailed insights on orientation, health, life stage, and sex.
The MAZE Image Processing Pipeline (MAZE-IPP) interfaces seamlessly with MorphoCluster and EcoTaxa, two prominent platforms in marine biological research. MAZE-IPP integrates with MorphoCluster by utilizing its robust clustering algorithms to group similar image data, enhancing the resegmentation process of LOKI data. This integration allows for more accurate and efficient sorting and analysis of large image datasets.

The MAZE-IPP interfaces seamlessly with MorphoCluster and EcoTaxa, two image annotation platforms.
MorphoCluster utilizes robust clustering algorithms to group similar image data, enhancing classification process of image data.
This integration allows for more accurate and efficient sorting and analysis of large image datasets.
EcoTaxa is used to streamline the fine-grained classification of organisms.
After the initial processing within MAZE-IPP, the segmented and classified images are exported to EcoTaxa, where they are further validated and curated.

.. TODO: Flowchart

.. graphviz:: flowchart.dot
   :caption: Interaction of the pipeline components within the ecosystem.


.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   loki
   morphocluster
   ecotaxa

