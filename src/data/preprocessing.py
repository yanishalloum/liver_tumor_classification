"""
Conversion des volumes NIfTI 3D (LiTS dataset) en slices 2D exploitables.

Différences clés avec la v1 (preprocess.py) :
  1. Split train/val/test fait par PATIENT et non par slice -> pas de fuite
     de données (deux slices voisines du même foie sont presque identiques ;
     les séparer entre train et test triche sur le score).
  2. Les masques sont conservés avec leurs valeurs originales (0/1/2), pas
     convertis en PNG "colorés" -> nécessaire pour faire de la segmentation
     multi-classe et pas juste de la classification.
  3. Utilisable en script (CLI) OU importable comme module, testable.
  4. Pas de chemins Windows en dur : tout vient de configs/config.yaml.

Ce module NE fait QUE le prétraitement -> génère un CSV d'index + des
fichiers .npy (plus rapides à charger que du JPEG/PNG pour du entraînement
répété, et sans perte de précision contrairement au JPEG).
"""
import os
import csv
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import yaml
from tqdm import tqdm


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def read_nii(file_path: str) -> np.ndarray:
    """Charge un volume NIfTI et applique la rotation standard LiTS."""
    scan = nib.load(file_path)
    return np.rot90(np.array(scan.get_fdata()))


def apply_windowing(image: np.ndarray, level: int, width: int) -> np.ndarray:
    """
    Fenêtrage DICOM : recentre l'intensité des pixels sur la plage utile
    pour le foie, puis normalise en [0, 1]. Sans ça, le foie a un contraste
    très faible au milieu de la plage complète des Hounsfield Units.
    """
    lo, hi = level - width // 2, level + width // 2
    windowed = np.clip(image, lo, hi)
    return (windowed - lo) / (hi - lo)


def find_all_volumes(dataset_roots: list) -> dict:
    """
    Parcourt tous les sous-dossiers volume_pt1, volume_pt2... de PLUSIEURS
    datasets racines et retourne un dict patient_id -> chemin complet du
    fichier volume. Nécessaire car le dataset LiTS complet (131 patients)
    est réparti sur deux datasets Kaggle distincts (0-50 et 51-130).
    """
    volumes = {}
    for dataset_root in dataset_roots:
        for entry in os.listdir(dataset_root):
            full_entry = os.path.join(dataset_root, entry)
            if os.path.isdir(full_entry) and entry.startswith("volume_pt"):
                for fname in os.listdir(full_entry):
                    if fname.startswith("volume-") and fname.endswith(".nii"):
                        patient_id = fname.replace("volume-", "").replace(".nii", "")
                        volumes[patient_id] = os.path.join(full_entry, fname)
    return volumes


def match_volumes_and_segmentations(dataset_roots: list, segmentations_root: str, segmentations_subdir: str) -> list:
    """Associe chaque volume-N.nii (dans volume_pt*/ de n'importe quel dataset_root) à sa segmentation-N.nii."""
    volumes = find_all_volumes(dataset_roots)
    segmentations_dir = os.path.join(segmentations_root, segmentations_subdir)

    pairs = []
    for patient_id, vol_path in sorted(volumes.items(), key=lambda x: int(x[0])):
        seg_path = os.path.join(segmentations_dir, f"segmentation-{patient_id}.nii")
        if os.path.exists(seg_path):
            pairs.append({
                "patient_id": patient_id,
                "volume_path": vol_path,
                "seg_path": seg_path,
            })
    return pairs


def split_by_patient(patient_ids: list, train_ratio: float, val_ratio: float, seed: int) -> dict:
    """
    Split au niveau patient (pas slice). Retourne un dict patient_id -> split.
    C'est LE correctif le plus important par rapport à la v1.
    """
    rng = np.random.default_rng(seed)
    ids = sorted(set(patient_ids))
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    split_map = {}
    for i, pid in enumerate(ids):
        if i < n_train:
            split_map[pid] = "train"
        elif i < n_train + n_val:
            split_map[pid] = "valid"
        else:
            split_map[pid] = "test"
    return split_map


def process_dataset(config: dict) -> str:
    """Pipeline complet : lit les .nii, génère slices + masques + CSV d'index."""
    data_cfg = config["data"]
    paths_cfg = config["paths"]

    processed_dir = Path(paths_cfg["processed_dir"])
    slices_dir = processed_dir / "slices"
    masks_dir = processed_dir / "masks"
    slices_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    pairs = match_volumes_and_segmentations(
        paths_cfg["raw_dataset_roots"], paths_cfg["raw_segmentations_root"], paths_cfg["raw_segmentations_subdir"]
    )
    if not pairs:
        raise FileNotFoundError(
            "Aucune paire volume/segmentation trouvée. Vérifie raw_dataset_roots "
            "dans config.yaml (sur Kaggle: Add Input -> ajoute à la fois "
            "andrewmvd/liver-tumor-segmentation ET andrewmvd/liver-tumor-segmentation-part-2)."
        )
    print(f"{len(pairs)} paires volume/segmentation trouvées.")

    split_map = split_by_patient(
        [p["patient_id"] for p in pairs],
        data_cfg["train_ratio"],
        data_cfg["val_ratio"],
        data_cfg["seed"],
    )

    records = []
    for pair in tqdm(pairs, desc="Patients"):
        volume = read_nii(pair["volume_path"])
        seg = read_nii(pair["seg_path"])
        split = split_map[pair["patient_id"]]

        for s in range(volume.shape[2]):
            mask_slice = seg[..., s]
            if 1 not in mask_slice and 2 not in mask_slice:
                continue  # slice sans foie -> pas informative, on l'écarte

            img_slice = apply_windowing(
                volume[..., s].astype(np.float32),
                data_cfg["window_level"],
                data_cfg["window_width"],
            )

            slice_name = f"patient{pair['patient_id']}_slice{s}"
            img_path = slices_dir / f"{slice_name}.npy"
            mask_path = masks_dir / f"{slice_name}.npy"
            np.save(img_path, img_slice.astype(np.float32))
            np.save(mask_path, mask_slice.astype(np.uint8))

            records.append({
                "patient_id": pair["patient_id"],
                "slice_path": str(img_path),
                "mask_path": str(mask_path),
                "has_tumor": int(2 in mask_slice),
                "split": split,
            })

    csv_path = processed_dir / "index.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "slice_path", "mask_path", "has_tumor", "split"])
        writer.writeheader()
        writer.writerows(records)

    print(f"{len(records)} slices générées depuis {len(pairs)} patients.")
    print(f"Index sauvegardé : {csv_path}")
    for split in ["train", "valid", "test"]:
        n_patients = sum(1 for v in split_map.values() if v == split)
        n_slices = sum(1 for r in records if r["split"] == split)
        print(f"  {split}: {n_patients} patients, {n_slices} slices")

    return str(csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    process_dataset(cfg)
