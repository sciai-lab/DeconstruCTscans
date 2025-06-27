# DeconstruCTscans: an automated pipeline for parsing complex CT assemblies
![fig1](https://github.com/user-attachments/assets/794b306f-ea7c-4d02-ba74-13f7add70056)

This repository contains code of an effective and fully automated pipeline for parsing large-scale, complex 3D assemblies from computed tomography (CT) scans into their individual parts. The pipeline combines a 3D deep boundary detection network trained only on simulated CT scans with efficient graph partitioning to segment the 3D scans. The predicted instance segments are matched and aligned with a known part catalog to form a set of candidate part poses. The subset of these proposals that jointly best reconstructs the assembly is found by solving an instance of the maximum weighted independent set problem. It produces high-fidelity reconstructions of assemblies of up to 3600 parts, accurately detecting both small and large parts.

## Conda environment
The code submission includes an `environment.yml` listing all the required packages. Setup the environement using

```conda env create -f environment.yml```

Navigate into the cloned DeconstruCTscans folder and install the package into the activated `deconstruct` environment using:

```pip install .```

### Dependencies
Similarly, clone and install the following dependencies:

https://github.com/imagirom/embeddingutils  
https://github.com/PeteLipp/stl-to-voxel  
https://github.com/PeteLipp/elf 

## Code structure
The main python module is called `deconstruct`. The code is strucutred into four major submodules:
- `data_generation`: contains all the code for creating and processing the data. It contains scripts to prepare the input for aRTist (CT scan simulation) and to generate ground truth annotations. It also contains code to create the HDF5 dataset (as provided in the Zenodo repository, see below).
- `instance_segmentation`: contains all the code for training and inference of the instance segmentation step.
- `proposal_generation`: contains all the code for generating proposals by matching and aligning the catalog parts with the instance segments.
- `proposal_selection`: contains all the code for selecting the best part proposals (solving a maximum weighted independent set problem) and assembling them into a final reconstruction.

## Downloading data
The full dataset of annotated seven annotated CT scans of complex assemblies can be downloaded from the following [Zenodo repository](https://zenodo.org/records/15730445).
A exemplary datasset of a small assembly is available there as `ct_assembly_dataset_small.zip`.

The dataset for each assembly (e.g. `first_assembly`) is structured as follows:
- `first_assembly_x10y5z5_dataset.h5`: an HDF5 file with the raw CT scan and ground truth instance annotation and semantic labels.
- `stl_catalog`: a folder which contains all the meshes of the part catalog and a `first_assembly_info.json` file which contains all information on how to assemble the parts in the scene. Some parts could not be used in the simulation of the CT scans because their meshes where not watertight.

Furthermore the repository includes:
- `unet_model`: a folder which contains the best validation check point of the trained model and a config-file for training and inference.
- `stls_watertight_replacements`: a folder which contains manually fixed versions of non-watertight catalog parts.

### Details on the HDF5 file structure and contents

Each file contains a raw scan (`raw_input_volume`), a corresponding ground-truth segmentation (`gt_instance_volume`), together with metadata about the scan setup, scaling, clipping offsets, and part semantics:

#### Datasets
- `raw_input_volume` (`uint16`):  
  The raw volumetric scan data.
- `gt_instance_volume` (`uint16`):  
  Ground-truth instance labels for each voxel.

#### Attributes
- `name` (`str`): Name of the scanned assembly, e.g. `"first_assembly"`.
- `semantic_label_list` (`np.ndarray[str]`): List of semantic class identifiers corresponding to instance IDs in the volume.
- `raw_min` / `raw_max` (`np.uint16`): Intensity range of the raw input volume.
- `clipping_min_corner` (`np.ndarray[int]`): Origin of the cropped region in the larger raw CT scan volume, needed for alignment in the 3D scene.
- `relative_scale_to_artist` (`float`): Only needed if a scaling was applied to part meshes before simulating the CT scans (`1.0` for all datasets).  
- `shift_to_place_meshes_in_volume` (`np.ndarray[float]`): Offsets for aligning meshes extracted from volumes and meshes from the 3D scene.
- `streak_str_addition` (`str`): String indicating the rotation performed on the assembly to avoid streak artifacts in the CT scan, e.g. `"x10y5z5"` meaning that a rotation by 10° around the x-axis, by 5° around the y-axis and by 5° around the z-axis.
- `voxelization_scale` (`np.float64`): Resolution of the voxel grid (`5.185185185185185` for all datasets).


## Pipeline demo: small assembly
<p align="center">
<img src=https://github.com/user-attachments/assets/6ee8aa7c-9d57-4360-92c1-15ddd66290a1 style="width: 60%;">
</p>

Even for the _small_ assembly the volume is larged compare to 2D data. Executing this demo therefore requires a GPU 
with at least 24GB of memory, about 10GB of disk space and 32 GB of RAM.  

To run the demo execute the following script:

```python full_inference.py --path_to_h5_data <path_to_h5_data>```

where `path_to_h5_data` is the path to the h5-dataset, e.g. `first_assembly_x10y5z5_dataset.h5` containing the CT scan and the groundtruth labels.
The script will run the full pipeline (instance segmentation, proposal generation and proposal selection) 
and save the results in an `output` folder. 

The `reconstruction_result.ply` file contains the final reconstruction and can be viewed with standard mesh viewers such as [meshlab](https://www.meshlab.net/) or [blender](https://www.blender.org/).

