# Tubular Segmentation
This repository contains the code to extract topological features, i.e. cycles and compoenents, from tubular networks. Doing so requires following these steps.

## 1. Data
The raw data format are in either`.lsm`or`.czi`formats. We need to save the tubular channel for each timepoint and save them as a `.tif`. To do so run
```
./scripts/convert_data.sh
```
You can replace the`--raw_file` argument to point to the desired raw data. You can add `--make_beta` to save beta channel as well. The argument `--tp_max` can also be specified
to ignore later timepoints.
## 2. Segmentation
Three segmentation models are already defined under`segmentation/models`:

`unet`is a fully-supervised network,

`ae`is a U-net whose encoder has been pretrained on an autoencoder,

`semi`is a semisuprvised U-net which trains a U-net and an autoencoder with a shared encoder jointly.

To run these models, you can execute
```
./scripts/unet.sh
./scripts/ae.sh
./scripts/semi.sh
```
These assume the models are already trained and the weights are saved in`.log/{model_name}/weights`. Adjust`--final_epoch`to select which epoch to use to make predictions.
The probability maps are saved to`ts_duct_path/../prob`.

To train these models,`--train`should be added in the arguments along with`--tr_duct_path`and`--tr_label_path` with proper input.
## 3. Skeletonization
We use the PyGEL3D module to extract skeletons from the binary segmentations as follows:
```
./scripts/skeletonize.sh
```
## 4. Cycles and Components
From the skeletons, we can detect cycles and components in each 3d image. To do so run
```
./scripts/cyc_cmp.sh
```
## 5. Filtering
To use the temporal information, we filter out cycles and components which are likely to be false positivies.

To filter out cycles, they are tracked over time and then those trajectories which are short-lived are removed.  How many timepoints is considered short is determined by `--thr`
in
```
./scripts/cyc_filtering.sh
```
To filter out components, every frame is compared to its immediate neighbors. If there is no component in its vicinity in the previous and the next frames, than that component
removed from that frame. What determines the vicinity is the `--thr` in the following scripts. The smaller its value the more aggresive the filtering.
```
./scripts/cyc_filtering.sh
```
## 6. Topology as Tif
To easily visualize the results, the cycles and components can be saved as a 5D tif file, assuming both the original files and their filtered versions
are saved in the same directroy specified by`--cyc_path`or`--cmp_path`. To make maximum intensity projection along the z axis, use`--make_mip` in
```
./scripts/cyc2tif.sh
./scripts./cmp2tif.sh
```
The red channel has been used to indicate the cycles or components which have been filtered out. The coloring in tif file for cycles is according to their tracking, while for components they are assigned at random.
