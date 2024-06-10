MorphoCluster
=============

MorphoCluster is utilized to pre-sort images through clustering.
Similar images are grouped together based on visual features, significantly enhancing efficiency and consistency of the classification.

1. Pull EcoTaxa data
--------------------

.. code:: sh
   
    # Get a full copy of EcoTaxa taxonomy tree
    # (The tree is stored in <taxonomy-fn>)
    pyecotaxa pull-taxonomy --format=<raw|yaml|...> <taxonomy-fn>

    # Pull one (or multiple) projects
    # (This stores the exported archives in the data/ directory.)
    # Arguments:
    #   -d: Working directory
    #   --with-images: Include images in the export.
    #   <project_id*>: A space-delimited list of project IDs
    pyecotaxa pull -d data --with-images <project_id*>

2. Annotate using MorphoCluster
-------------------------------

Annotation using MorphoCluster involves a series of structured steps to effectively cluster and annotate plankton images (see Schroeder et al. 2020).
First, images and metadata imported from an EcoTaxa archive.
Features of these images are then extracted using a deep learning model.
The images are annotated using the MorphoCluster web application, where users can validate and refine clusters.
In the end, annotations are exported.

3. Map MorphoCluster labels to EcoTaxa categories
-------------------------------------------------

.. code:: sh

   pyecotaxa map-categories --taxonomy <taxonomy-fn>

3a. Merge annotations back into project data
--------------------------------------------

(This step might be necessary if multiple EcoTaxa projects were
annotated jointly in MorphoCluster or if previous annotations should not
be overwritten.)

.. code:: sh

   # (This creates <output-fn> with )
   # Arguments:
   #   --annotation-only: Only `object_annotation_*` columns are contained in the output.
   #   --predicted-only: Only objects with annotation status "predicted" are contained in the output.
   #       (This avoids overwriting already validated annotations.)
   pyecotaxa merge --annotation-only --predicted-only <export-archive-fn> annotations.tsv <output-fn>

4. Push annotated data back to EcoTaxa
--------------------------------------

.. code:: sh

   pyecotaxa push --project-id <project_id> --update-annotations <data-fn>
