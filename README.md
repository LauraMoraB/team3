# M1 - Traffic Signs Detection


Introduction to Human and Computer Vision project

## Methods

### Connected Component Labeling

Two methods have been implemented:

- **Method 1**

    - Color Segmentation
    - Contour Detection
    - BBox Definition

- **Method 2**

    - Color Segmentation
    - Morphology Filters
    - Fill Holes
    - Contour Detection
    - BBox Definition

### Sliding Window

- **Method 3**: SLW approach using square windows

- **Method 4**: SLW approach using rectangular windows

- **Method 5**: SLW Fast approach using recursive analysis of the image

## Code Execution

```bash
$ python main.py -h

usage: main.py [-h] [-ld] [-ps] [-t {SLW2,SLW3,SLW_FAST,CCL}]
               [--test | --train | --validate]

optional arguments:
  -h, --help            show this help message and exit
  --test                test excludes train, validate
  --train               train excludes test, validate
  --validate            validate excludes test, train

General arguments:
  -ld, --load_data      Load data
  -ps, --plot_slots     Ploting slots
  -t {SLW2,SLW3,SLW_FAST,CCL}, --task {SLW2,SLW3,SLW_FAST,CCL}

```

Parameters:

- ld: load data into dataframes and compute filling ratios, etc
- ps: plot image features
- t: method to be executed
- train/test/validate: data which is going to be analysed. Only one at a time can be used

The paths to find the data has to be changed in the code.
