# Dataset card

Original creator: Jun Cheng. Source: https://doi.org/10.6084/m9.figshare.1512427.v5
License: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/

The original dataset is not authored by this project's author. It is used and
transformed with attribution. Publisher metadata is preserved in `source.json`.

Original MATLAB `cjdata.label` maps 1=meningioma, 2=glioma, 3=pituitary.
The importer maps these to zero-based model labels 0,1,2. `cjdata.PID` supplies
the patient group; `cjdata.image` is the only image input. MATLAB array axes are
transposed on import through h5py. Tumor borders/masks are not used.

Transformations: per-image linear intensity scaling to 8-bit grayscale PNG;
whole-image resizing to 224x224 during loading; mild affine augmentation in the
training split only. Train/validation/test manifests are generated with patient
groups and pixel-duplicate connected groups. The publisher's slice CV indices are
downloaded for provenance but are not used for this project's split.

Limitations: only tumor-bearing slices and three diagnoses are represented.
No healthy control class, external evaluation or clinical validation is included.
Patient-disjoint splitting does not establish hospital-disjoint evaluation.
Near-duplicate and acquisition-site leakage require additional study.
