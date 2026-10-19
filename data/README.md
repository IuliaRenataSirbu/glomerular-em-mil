# Data

This directory shows the required data structure. No real patient data is included.

## Required files

### `images/`
Directory containing patient EM image subdirectories. Each patient has a folder with one or more `.tif` images:
```
images/
  patient_id_1/
    patient_id_1-image_1.tif
    patient_id_1-image_2.tif
  patient_id_2/
    ...
```

### `images_index.json`
A JSON list indexing all images with metadata. See `images_index_example.json` for the required fields. The key fields used by the pipeline are `fullpath`, `patient_id`, and `diagnosis`.

### `folds.json`
Cross-validation split. See `folds_example.json` for format. Contains:
- `config`: dataset statistics
- `test`: list of patient IDs held out as the test set
- `folds`: dict of fold index → `{train: [...], val: [...]}` patient ID lists
- `patient_labels`: dict mapping patient ID to class name

### `if_markers.csv`
Immunofluorescence marker values per patient. See `if_markers_example.csv` for format. Required columns: `ID` (patient ID) plus one binary column per marker: `IgAbin_corrected`, `IgGbin_corrected`, `IgMbin_corrected`, `C1qbin_corrected`, `C3cbin_corrected`, `Kbin_corrected`, `lambdabin_corrected`, `Fbbin_corrected`.

Only required when using models with IF fusion (`TAB_FUSION: true`).

## Setting paths

After preparing your data files, set their paths in `code/mil_classifier/configs/paths.yml` (copy from `paths.yml.example`).
