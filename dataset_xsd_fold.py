import os
import SimpleITK as sitk
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


BASE_DIR = os.environ.get("IRCADB_BASE_DIR", r"/home/gpuserver/zhz/Datasets-3/3Dircadb/fold")


DEFAULT_SPLITS_DIR = os.environ.get(
    "IRCADB_SPLITS_DIR",
    os.path.join(os.path.dirname(__file__), "3Dircadb_5fold_splits_ssl")
)


DEFAULT_FOLD = int(os.environ.get("IRCADB_FOLD", 1))


DEFAULT_LABEL_RATIO = int(os.environ.get("IRCADB_LABEL_RATIO", 50))


# Extract fixed-size patches from a 3D volume.
def extract_patches(image, patch_size, stride):
    # Slide a 3D window over the volume.
    patches = []
    D, H, W = image.shape


    for d in range(0, D - patch_size[0] + 1, stride[0]):
        for h in range(0, H - patch_size[1] + 1, stride[1]):
            for w in range(0, W - patch_size[2] + 1, stride[2]):
                patch = image[d:d + patch_size[0], h:h + patch_size[1], w:w + patch_size[2]]
                patches.append(patch)

    return patches


def pad_image_to_fit_patch(image, patch_size, stride):
    D, H, W = image.shape


    pad_d = (stride[0] - (D - patch_size[0]) % stride[0]) % stride[0]
    pad_h = (stride[1] - (H - patch_size[1]) % stride[1]) % stride[1]
    pad_w = (stride[2] - (W - patch_size[2]) % stride[2]) % stride[2]


    padding = [
        (pad_d // 2, pad_d - pad_d // 2),
        (pad_h // 2, pad_h - pad_h // 2),
        (pad_w // 2, pad_w - pad_w // 2)
    ]


    padded_image = np.pad(image, padding, mode='constant', constant_values=0)
    return padded_image, padding


# Dataset that serves labeled and unlabeled 3D patches.
class CustomDataset(Dataset):
    def __init__(self, data_dir, patients, patch_size=(16, 16, 16), stride=(8, 8, 8), transform=None,
                 unlabeled=False):

        self.data_dir = data_dir


        self.transform = transform


        self.patients = patients


        # If True, only CT patches are returned.
        self.unlabeled = unlabeled

        # Patch extraction settings.
        self.patch_size = patch_size
        self.stride = stride


        self.ct_patches = []


        self.gt_patches = []


        self.paddings = []


        for patient in self.patients:
            patient_dir = os.path.join(data_dir, patient)


            ct_path = os.path.join(patient_dir, 'resampled_liver.nii.gz')
            ct_img = sitk.ReadImage(ct_path)
            ct_img = sitk.GetArrayFromImage(ct_img).astype(np.float32)


            ct_img = ct_img / np.max(ct_img)


            ct_img, padding = pad_image_to_fit_patch(ct_img, self.patch_size, self.stride)
            self.paddings.append(padding)
            ct_patches = extract_patches(ct_img, self.patch_size, self.stride)

            if not self.unlabeled:

                gt_path = os.path.join(patient_dir, 'resampled_portal.nii.gz')
                gt_img = sitk.ReadImage(gt_path)
                gt_img = sitk.GetArrayFromImage(gt_img).astype(np.int32)


                gt_img, _ = pad_image_to_fit_patch(gt_img, self.patch_size, self.stride)
                gt_patches = extract_patches(gt_img, self.patch_size, self.stride)


                # Keep only informative labeled patches.
                for ct_patch, gt_patch in zip(ct_patches, gt_patches):
                    if np.sum(gt_patch) > 0:
                        self.ct_patches.append(ct_patch)
                        self.gt_patches.append(gt_patch)
            else:


                for ct_patch in ct_patches:
                    if np.sum(ct_patch) > 0:
                        self.ct_patches.append(ct_patch)


    def __len__(self):
        return len(self.ct_patches)


    def __getitem__(self, idx):
        ct_patch = self.ct_patches[idx]


        if self.transform:
            ct_patch = self.transform(ct_patch)


        ct_patch = ct_patch.unsqueeze(0)


        if self.unlabeled:
            return ct_patch


        gt_patch = self.gt_patches[idx]
        if self.transform:
            gt_patch = self.transform(gt_patch)


        gt_patch = gt_patch.unsqueeze(0)
        return ct_patch, gt_patch


transform = transforms.Compose([transforms.ToTensor()])


def read_patient_list(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def get_split_patients(splits_dir=DEFAULT_SPLITS_DIR, fold=DEFAULT_FOLD, label_ratio=DEFAULT_LABEL_RATIO):

    if label_ratio not in (20, 50):
        raise ValueError("label_ratio must be 20 or 50")


    if fold not in (1, 2, 3, 4, 5):
        raise ValueError("fold must be in [1, 5]")


    train_labeled_patients = read_patient_list(
        os.path.join(splits_dir, f"fold{fold}_train_labeled_{label_ratio}.txt")
    )


    train_unlabeled_patients = read_patient_list(
        os.path.join(splits_dir, f"fold{fold}_train_unlabeled_{label_ratio}.txt")
    )


    val_patients = read_patient_list(os.path.join(splits_dir, f"fold{fold}_val.txt"))
    test_patients = read_patient_list(os.path.join(splits_dir, f"fold{fold}_test.txt"))


    # Package datasets, loaders, and split metadata for training.
    return {
        "train_labeled": train_labeled_patients,
        "train_unlabeled": train_unlabeled_patients,
        "val": val_patients,
        "test": test_patients,
    }


# Create train/val/test datasets for one fold.
def create_datasets(
    base_dir=BASE_DIR,
    splits_dir=DEFAULT_SPLITS_DIR,
    fold=DEFAULT_FOLD,
    label_ratio=DEFAULT_LABEL_RATIO,
    patch_size=(16, 16, 16),
    stride=(8, 8, 8),
    transform_fn=transform,
):

    split_patients = get_split_patients(splits_dir=splits_dir, fold=fold, label_ratio=label_ratio)


    train_labeled_dataset = CustomDataset(
        base_dir,
        split_patients["train_labeled"],
        patch_size=patch_size,
        stride=stride,
        transform=transform_fn,
        unlabeled=False,
    )


    train_unlabeled_dataset = CustomDataset(
        base_dir,
        split_patients["train_unlabeled"],
        patch_size=patch_size,
        stride=stride,
        transform=transform_fn,
        unlabeled=True,
    )


    val_dataset = CustomDataset(
        base_dir,
        split_patients["val"],
        patch_size=patch_size,
        stride=stride,
        transform=transform_fn,
        unlabeled=False,
    )


    test_dataset = CustomDataset(
        base_dir,
        split_patients["test"],
        patch_size=patch_size,
        stride=stride,
        transform=transform_fn,
        unlabeled=False,
    )

    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset


def create_dataloaders(
    base_dir=BASE_DIR,
    splits_dir=DEFAULT_SPLITS_DIR,
    fold=DEFAULT_FOLD,
    label_ratio=DEFAULT_LABEL_RATIO,
    patch_size=(16, 16, 16),
    stride=(8, 8, 8),
    labeled_batch_size=16,
    unlabeled_batch_size=16,
    val_batch_size=16,
    test_batch_size=16,
    shuffle_train=True,
):

    train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset = create_datasets(
        base_dir=base_dir,
        splits_dir=splits_dir,
        fold=fold,
        label_ratio=label_ratio,
        patch_size=patch_size,
        stride=stride,
        transform_fn=transform,
    )


    train_labeled_dataloader = DataLoader(
        train_labeled_dataset,
        batch_size=labeled_batch_size,
        shuffle=shuffle_train,
        drop_last=False,
    )


    train_unlabeled_dataloader = DataLoader(
        train_unlabeled_dataset,
        batch_size=unlabeled_batch_size,
        shuffle=shuffle_train,
        drop_last=False,
    )


    val_dataloader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        drop_last=False,
    )


    test_dataloader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
    )


    return {
        "train_labeled_dataset": train_labeled_dataset,
        "train_unlabeled_dataset": train_unlabeled_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "train_labeled_dataloader": train_labeled_dataloader,
        "train_unlabeled_dataloader": train_unlabeled_dataloader,
        "val_dataloader": val_dataloader,
        "test_dataloader": test_dataloader,
        "split_patients": {
            "train_labeled": train_labeled_dataset.patients,
            "train_unlabeled": train_unlabeled_dataset.patients,
            "val": val_dataset.patients,
            "test": test_dataset.patients,
        },
    }


# Backward-compatible placeholders.


train_labeled_dataset = None
train_unlabeled_dataset = None
val_dataset = None
test_dataset = None
train_labeled_dataloader = None
train_unlabeled_dataloader = None
val_dataloader = None
test_dataloader = None


if os.path.isdir(DEFAULT_SPLITS_DIR):
    try:
        _default_objects = create_dataloaders()
        train_labeled_dataset = _default_objects["train_labeled_dataset"]
        train_unlabeled_dataset = _default_objects["train_unlabeled_dataset"]
        val_dataset = _default_objects["val_dataset"]
        test_dataset = _default_objects["test_dataset"]
        train_labeled_dataloader = _default_objects["train_labeled_dataloader"]
        train_unlabeled_dataloader = _default_objects["train_unlabeled_dataloader"]
        val_dataloader = _default_objects["val_dataloader"]
        test_dataloader = _default_objects["test_dataloader"]
    except Exception:


        pass


if __name__ == "__main__":
    objects = create_dataloaders()
    print(f"Using fold {DEFAULT_FOLD}, label ratio {DEFAULT_LABEL_RATIO}%")
    print('Number of samples in labeled train set: ', len(objects["train_labeled_dataset"]))
    print('Number of samples in unlabeled train set: ', len(objects["train_unlabeled_dataset"]))
    print('Number of samples in validation set: ', len(objects["val_dataset"]))
    print('Number of samples in test set: ', len(objects["test_dataset"]))
    print()

    print('Patients in labeled train set:')
    for patient in objects["split_patients"]["train_labeled"]:
        print(patient)
    print()

    print('Patients in unlabeled train set:')
    for patient in objects["split_patients"]["train_unlabeled"]:
        print(patient)
    print()

    print('Patients in validation set:')
    for patient in objects["split_patients"]["val"]:
        print(patient)
    print()

    print('Patients in test set:')
    for patient in objects["split_patients"]["test"]:
        print(patient)
    print()
