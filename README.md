# LOKI Image Processing Pipeline (LOKI -> EcoTaxa)

(Re-)Segmentation and meta-data collection of LOKI images. 

## Usage

Execute a task defined in a YAML configuration file:

```sh
python pipeline.py task.yaml
```

### Example configuration

```yaml
input:
    ## Input path. May contain one or multiple exports.
    path: path/to/loki-project

    ## Default metadata
    meta:
        sample_bottomdepth: -99
        sample_detail_location: XXX
        sample_region: XXX

    segmentation:
        ## One of the following:

        ## Simple segmentation using thresholding
        # threshold: 128

        ## Pickled segmenter
        # stored:
        #   pickle_fn: filename.pkl

        ## PyTorch model for segmentation
        pytorch:
            jit_model_fn: model.jit.pt
            device: cuda:0
      
output:
  path: path/to/output-directory
```