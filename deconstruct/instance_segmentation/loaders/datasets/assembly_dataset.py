import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
from deconstruct.instance_segmentation.loaders.transforms import configure_transforms
from deconstruct.instance_segmentation.loaders.transforms.custom_transforms import TargetToAffinities
from deconstruct.instance_segmentation.inference.stitching import calculate_viable_chunking

from dataclasses import dataclass


@dataclass
class InsSegAffinityData:
    """all tensors are of shape (B, C, *img_shape)"""
    raw: torch.Tensor
    gt_insseg: torch.Tensor
    affs: torch.Tensor
    left_corner: torch.Tensor  # of shape (B, len(img_shape))
    path_h5_dataset: list[str]  # of length B

    pred: torch.Tensor = None

    @staticmethod
    def collate_fn(batch):
        raw_batch = torch.cat([data.raw for data in batch], dim=0)
        gt_insseg_batch = torch.cat([data.gt_insseg for data in batch], dim=0)
        affs_batch = torch.cat([data.affs for data in batch], dim=0)

        left_corner_batch = torch.stack([data.left_corner for data in batch], dim=0)
        data_file_batch = [file for data in batch for file in data.path_h5_dataset]

        return InsSegAffinityData(raw_batch, gt_insseg_batch, affs_batch, left_corner_batch, data_file_batch)

    def get_sample(self, index):
        return InsSegAffinityData(self.raw[index].unsqueeze(0),
                                  self.gt_insseg[index].unsqueeze(0),
                                  self.affs[index].unsqueeze(0),
                                  self.left_corner[index].unsqueeze(0),
                                  [self.path_h5_dataset[index]],
                                  pred=self.pred[index].unsqueeze(0) if self.pred is not None else None)

    def __getitem__(self, key):
        return getattr(self, key)

    def detach(self):
        self.raw = self.raw.detach()
        self.gt_insseg = self.gt_insseg.detach()
        self.affs = self.affs.detach()
        self.left_corner = self.left_corner.detach()
        if self.pred is not None:
            self.pred = self.pred.detach()
        return self

    def to(self, device):
        self.raw = self.raw.to(device)
        self.gt_insseg = self.gt_insseg.to(device)
        self.affs = self.affs.to(device)
        self.left_corner = self.left_corner.to(device)
        if self.pred is not None:
            self.pred = self.pred.to(device)
        return self


class SingleCarDataset(Dataset):
    """
    input:  - data_file:
            - datasets have three keys: "raw_input_volume", "GT_instance_volume", "GT_semantic_volume" of same shape
            - transforms used for data augmentation
            - index_range: list off lowest and highest index

    getitem: returns single image and target image with shape (C=1, img_shape)
    exactly as 2d only the transformations are different.
    """

    def __init__(self, path_h5_dataset, affinity_transform_config, chunk_shape, transform_configs=None,
                 background_label=0, rescale_input=True, load_all_data_in_memory=False, gt_is_given=True):
        """
        Dataset for a single car volume.

        Parameters
        ----------
        path_h5_dataset : str
            hdf5 generated with data generation notebook. all images are type np.uint16.
        num_random_chunks : int
            sets the length of the dataset, specifies number of random chunks taken from the volume
        chunk_shape : tuple-like
            Specifies size of random chunks.
        transform : function, optional
            Takes input and target img and transforms them accordingly for data augmentation. Default : no trafo.
        transform_dict : dict, optional
            Additional parameters that transform function takes besides input and target img.
        seed_factor : int, optional
            Used to specifies np.random.seed so that the cropping becomes deterministic.
            --> Get item returns the same crop for the same index.
        rescale_input : bool, optional
            input volume is scaled to be float between [0,1]. Default = True.
        reject_empty_chunks_probability : float, optional
            between (0,1). probability to reject completely empty/background chunks. Default = .95.
        background_label : int, optional
            Label for background, default = 0.
        load_all_data_in_memory: bool, optional
            If True, the whole dataset is loaded into memory. Default = False.
        """
        self.path_h5_dataset = path_h5_dataset
        self.gt_is_given = gt_is_given
        self.transforms = None if transform_configs is None else configure_transforms(transform_configs)
        self.affinity_transform = TargetToAffinities(**affinity_transform_config)
        self.background_label = background_label
        self.load_all_data_in_memory = load_all_data_in_memory
        self.rescale_input = rescale_input

        # get general info about the dataset:
        with h5py.File(self.path_h5_dataset, "r") as f:
            assert "raw_input_volume" in f, "raw_input_volume not found in data file"
            assert "gt_instance_volume" in f or not self.gt_is_given, "gt_instance_volume not found in data file"
            self.img_shape = f["raw_input_volume"].shape
            self.raw_min = f.attrs["raw_min"]
            self.raw_max = f.attrs["raw_max"]

        if self.load_all_data_in_memory:
            with h5py.File(self.path_h5_dataset, "r") as f:
                self.gt_insseg = np.array(f["gt_instance_volume"], dtype=np.int32) if self.gt_is_given else None
                self.raw = np.array(f["raw_input_volume"], dtype=np.int32)  # bc uint16 can't be converted to torch
        else:
            self.raw = None
            self.gt_insseg = None

        self.info_from_ds_to_model = {"offsets": self.affinity_transform.offsets,
                                      "pad_value": self.affinity_transform.pad_value}

        assert (np.array(chunk_shape) <= np.array(self.img_shape)).all(), f"chunk_shape {chunk_shape} must be " \
                                                                          f"smaller than img_shape {self.img_shape}"
        self.chunk_shape = tuple(chunk_shape)

    def get_input_and_target(self, left_corner):
        sl = tuple([slice(left_corner[i], left_corner[i] + s) for i, s in enumerate(self.chunk_shape)])
        if self.load_all_data_in_memory:
            input_volume = torch.from_numpy(self.raw[sl]).view((1,) + self.chunk_shape)
            target_volume = torch.from_numpy(self.gt_insseg[sl]).view((1,) + self.chunk_shape) if self.gt_is_given \
                else None
        else:
            with h5py.File(self.path_h5_dataset, "r") as f:
                input_volume = f["raw_input_volume"][sl]
                input_volume = torch.from_numpy(input_volume.astype(np.int32)).view((1,) + self.chunk_shape)
                if self.gt_is_given:
                    target_volume = f["gt_instance_volume"][sl]
                    target_volume = torch.from_numpy(target_volume.astype(np.int32)).view((1,) + self.chunk_shape)
                else:
                    target_volume = None
        return input_volume, target_volume

    def common_getitem(self, input_volume, target_volume, left_corner):
        if self.rescale_input:
            # return input volume as float between [0,1]:
            input_volume = (input_volume - self.raw_min) / (self.raw_max - self.raw_min)
        else:
            # only convert to float:
            input_volume = input_volume.float()

        if self.transforms is not None:
            # input and target need to be transformed in the same way, combine them in one tensor:
            # expects input tensor of shape (C,img_shape)
            input_volume, target_volume = self.transforms(input_volume, target_volume)

        # compute affinities:
        affs = self.affinity_transform(target_img=target_volume)[1] if self.gt_is_given else None

        # init data instance:
        img_shape = input_volume.shape[1:]
        data = InsSegAffinityData(raw=input_volume.view(1, -1, *img_shape),
                                  gt_insseg=None if target_volume is None else target_volume.view(1, -1, *img_shape),
                                  affs=None if affs is None else affs.view(1, -1, *img_shape),
                                  left_corner=torch.tensor(left_corner).view(1, len(img_shape)),
                                  path_h5_dataset=[self.path_h5_dataset])

        return data

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


class SingleCarRandomDataset(SingleCarDataset):

    def __init__(self, num_random_chunks, seed_factor=None, reject_empty_chunks_probability=0.95, **super_kwargs):
        super().__init__(**super_kwargs)

        self.seed_factor = seed_factor
        self.num_random_chunks = num_random_chunks
        self.reject_empty_chunks_probability = reject_empty_chunks_probability

    def __len__(self):
        return self.num_random_chunks

    def __getitem__(self, index):
        # take a random chunk of the volume:
        # sample a random center voxel:
        # set a seed if validation set:
        if self.seed_factor is not None:
            np.random.seed(index * self.seed_factor)

        # make sure that chunk is not completely empty
        while True:
            left_corner = []
            for i in range(3):
                # ensure that no padding is required
                left_corner_max = max(1, self.img_shape[i] - self.chunk_shape[i] - 1)
                left_corner.append(np.random.randint(left_corner_max))

            input_volume, target_volume = self.get_input_and_target(left_corner)

            chunk_is_empty = (target_volume == self.background_label).all()

            # if not self.reject_empty_chunks or not chunk_is_empty:
            if np.random.random() > self.reject_empty_chunks_probability or not chunk_is_empty:
                # if chunk_is_empty:
                #     print("empty chunk was accepted")
                break

        data = self.common_getitem(input_volume, target_volume, left_corner)

        # reset random seed:
        np.random.seed()

        return data


class SingleCarGridDataset(SingleCarDataset):

    def __init__(self, overlap_shape, shrinking_shape=None, **super_kwargs):
        super().__init__(**super_kwargs)

        self.overlap_shape = overlap_shape
        self.shrinking_shape = shrinking_shape
        self.all_left_corners, self.list_slices = self.create_slicing_of_volume()

    def create_slicing_of_volume(self):
        """create slicing of volume"""
        if self.shrinking_shape is None:
            effective_overlap_shape = self.overlap_shape
        else:
            effective_overlap_shape = tuple(np.asarray(self.overlap_shape) + 2 * np.asarray(self.shrinking_shape))
        print(f"affinity prediction using overlap shape {self.overlap_shape}, effective overlap shape "
              f"{effective_overlap_shape}, shrinking shape {self.shrinking_shape} and "
              f"chunk shape {self.chunk_shape}.")
        all_left_corners, _ = calculate_viable_chunking(img_shape=self.img_shape,
                                                              chunk_shape=self.chunk_shape,
                                                              overlap_shape=effective_overlap_shape)

        # create a list of slices:
        list_slices = []
        for left_corner in all_left_corners:
            list_slices.append(tuple([slice(left_corner[i], left_corner[i] + s)
                                      for i, s in enumerate(self.chunk_shape)]))

        return all_left_corners, list_slices

    def __len__(self):
        return len(self.list_slices)

    def __getitem__(self, index):
        left_corner = self.all_left_corners[index]
        input_volume, target_volume = self.get_input_and_target(left_corner)
        data = self.common_getitem(input_volume, target_volume, left_corner)

        return data


class MultiCarDataset(Dataset):

    def __init__(self, list_data_files, num_random_chunks_list, seed_factor_list=None, **single_car_kwargs):
        """
        Takes the inputs that the single data set does, plus a list of data_files to construct a joint dataset with
        torch.utils.data.ConcatDataset().


        Parameters
        ----------
        list_data_files : list[str]
            Lists of data file paths.
        num_random_chunks_list : list[int] or int
            Specifies how many chunks should be taken from which car. If num_random_chunks_list is just a number,
             it's assumed that all datasets should have the same number of random chunks
        seed_factor_list : list[int or None], optional
        single_car_kwargs

        Returns
        -------
        torch.ConcatDataset
            Concatenated dataset consisting of chunks from the different cars.
        """

        if not isinstance(num_random_chunks_list, list):
            num_random_chunks_list = [num_random_chunks_list, ] * len(list_data_files)
        if not isinstance(seed_factor_list, list):
            seed_factor_list = [seed_factor_list, ] * len(list_data_files)

        assert len(seed_factor_list) == len(list_data_files), "len of seed list must match number of data_files"
        assert len(num_random_chunks_list) == len(list_data_files), "len of num_random_chunks_list must " \
                                                                    "match number of data_files"

        # init each individual dataset:
        list_datasets = []
        for i, data_file in enumerate(list_data_files):
            list_datasets.append(SingleCarRandomDataset(
                path_h5_dataset=data_file,
                num_random_chunks=num_random_chunks_list[i],
                seed_factor=seed_factor_list[i],
                **single_car_kwargs
            ))

        self.info_from_ds_to_model = list_datasets[0].info_from_ds_to_model
        self.concat_dataset = torch.utils.data.ConcatDataset(list_datasets)

    def __len__(self):
        return len(self.concat_dataset)

    def __getitem__(self, index):
        return self.concat_dataset[index]

