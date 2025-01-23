import torch
import numpy as np
from torch.utils.data import Dataset
from monai.transforms import Compose
import os
from pathlib import Path
import random
import monai.transforms as T


class BasicDataset(Dataset):
    def __init__(
        self,
        root_path,
        test_mode=False,
        randomly_drop_modalities=False,
        dataset_type="BRATS",
        return_context=False,
        context_dim=10, # 6 modalities + 4 diseases
    ):
        self.dataset_info = { # (modalities, number of training data)
            "BRATS": [["FLAIR", "T1", "T1c", "T2"], 444],
            "ATLAS": [["T1"], 459],
            "MSSEG": [["FLAIR", "T1", "T1c", "T2", "PD"], 37],
            "ISLES": [["FLAIR", "T1", "T2", "DWI"], 20],
            "WMH": [["FLAIR", "T1"], 42],
        }
        self.return_context = return_context
        self.disease = { # we have categoriesed the datasets into 4 diseases
            "BRATS": torch.tensor([1, 0, 0, 0]),
            "ATLAS": torch.tensor([0, 1, 0, 0]),
            "MSSEG": torch.tensor([0, 0, 1, 0]),
            "ISLES": torch.tensor([0, 1, 0, 0]),
            "WMH": torch.tensor([0, 0, 0, 1]),
        }
        # self.size = {  # from big to small
        #     "BRATS": torch.tensor([1, 0, 0]),
        #     "ATLAS": torch.tensor([0, 1, 0]),
        #     "MSSEG": torch.tensor([0, 0, 1]),
        #     "ISLES": torch.tensor([0, 1, 0]),
        #     "WMH": torch.tensor([0, 0, 1]),
        # }
        self.total_modalities = self._get_total_modalities()
        self.dataset_type = self._validate_dataset_type(dataset_type)
        self.test_mode = test_mode
        self.root_path = root_path
        self.img_paths, self.seg_paths = self._get_image_and_label_paths()
        self.transform = self._get_transforms(test_mode)
        self.randomly_drop_modalities = randomly_drop_modalities
        self.modality_indices = self._map_channels(
            self.dataset_info[self.dataset_type][0], self.total_modalities
        )
        if self.return_context:
            # assert context_dim in [10, 13]  # 6 modalities + 4 diseases + 3 sizes
            assert context_dim == 10  # 6 modalities + 4 diseases
            self.context_dim = context_dim
        print(f"Dataset {self.dataset_type} created with {len(self)} images.")
        # print(f"Modality indices: {self.modality_indices}")

    def _get_total_modalities(self):
        total_modalities = set()
        for modalities in self.dataset_info.values():
            total_modalities.update(modalities[0])
        return sorted(list(total_modalities))

    def _validate_dataset_type(self, dataset_type):
        if dataset_type not in self.dataset_info.keys():
            raise ValueError(f"Dataset type {dataset_type} not recognized.")
        return dataset_type

    def _get_image_and_label_paths(self):
        img_paths = sorted(
            Path(os.path.join(self.root_path, self.dataset_type, "images")).glob(
                "*.nii.gz"
            )
        )
        seg_paths = sorted(
            Path(os.path.join(self.root_path, self.dataset_type, "labels")).glob(
                "*.nii.gz"
            )
        )
        assert len(img_paths) == len(
            seg_paths
        ), "Number of images and labels must be the same"

        val_size = len(img_paths) - self.dataset_info[self.dataset_type][1]
        if self.test_mode:
            return img_paths[-val_size:], seg_paths[-val_size:]
        else:
            return img_paths[:-val_size], seg_paths[:-val_size]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        data = {"image": self.img_paths[index], "seg": self.seg_paths[index]}
        data = self.transform(data)
        img, target = data["image"], data["seg"]

        if (
            (self.randomly_drop_modalities)
            and (self.dataset_type != "ATLAS")
            and (self.test_mode == False)
        ):
            _, modified_img = self.rand_set_channels_to_zero(
                self.dataset_info[self.dataset_type][0], img
            )
            # print(f"remaining modalities: {remaning_modalities_index}")
        else:
            modified_img = img

        complete_img = torch.zeros(
            (len(self.total_modalities), *modified_img.shape[1:]), dtype=torch.float32
        )
        complete_img[self.modality_indices, :, :, :] = modified_img

        # print(f"type of complete_img: {type(complete_img)}")
        if self.return_context:
            modality_context = torch.any(complete_img != 0, dim=(1, 2, 3))
            disease_context = self.disease[self.dataset_type]
            # size_context = self.size[self.dataset_type]
            if self.context_dim == 10:
                context = torch.cat((modality_context, disease_context), dim=0)
            # else:
            #     context = torch.cat(
            #         (modality_context, disease_context, size_context), dim=0
            #     )
            # print(f"Context: {context}")
            # print(f"context shape: {context.shape}")
            return (
                np.array(complete_img),
                np.array(target),
                np.array(context, dtype=np.float32),
            )
        else:
            return (
                np.array(complete_img),
                np.array(target),
            ) 

    def _get_transforms(self, test_mode):
        if not test_mode:
            return Compose(
                [
                    T.LoadImaged(keys=["image", "seg"]),
                    T.EnsureChannelFirstd(keys=["image", "seg"]),
                    T.RandSpatialCropd(
                        keys=["image", "seg"],
                        roi_size=(
                            128,
                            128,
                            128,
                        ),
                    ),
                    T.RandRotate90d(
                        keys=["image", "seg"], prob=0.1, spatial_axes=(0, 2)
                    ),
                    T.CastToTyped(
                        keys=["image", "seg"], dtype=(torch.float32, torch.uint8)
                    ),
                ]
            )
        else:
            return Compose(
                [
                    T.LoadImaged(keys=["image", "seg"]),
                    T.EnsureChannelFirstd(keys=["image", "seg"]),
                    T.CastToTyped(
                        keys=["image", "seg"], dtype=(torch.float32, torch.uint8)
                    ),
                ]
            )

    def _map_channels(self, modalities, total_modalities):
        channel_map = []
        for channel in modalities:
            for index, modality in enumerate(total_modalities):
                if channel == modality:
                    channel_map.append(index)
        return channel_map

    def rand_set_channels_to_zero(self, dataset_modalities: list, img: torch.Tensor):
        modalities_remaining = []
        number_of_dropped_modalities = np.random.randint(0, len(dataset_modalities))
        modalities_dropped = random.sample(
            list(np.arange(len(dataset_modalities))), number_of_dropped_modalities
        )
        modalities_dropped.sort()
        img[modalities_dropped, :, :, :] = 0.0
        modalities_remaining = list(
            set(np.arange(len(dataset_modalities))) - set(modalities_dropped)
        )
        return modalities_remaining, img

    def _get_name(self):
        return self.dataset_type


if __name__ == "__main__":
    test_dataset = BasicDataset(
        root_path="/hdd/Continual_learning_data/FINAL",
        test_mode=False,
        dataset_type="BRATS",
        randomly_drop_modalities=True,
    )
    img, seg = test_dataset[0]
    print(img.shape, seg.shape)
    # tell me on which channel the maximum value is zero
    for i in range(img.shape[0]):
        if img[i].max() == 0:
            print(f"Channel {i} has max value of zero")
