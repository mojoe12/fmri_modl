# fMRI SENSE

Unfold GE SMS-EPI fMRI k-space with a SENSE inverse (coil sensitivity maps plus the multiband / PE encoding matrix). Training optionally stacks a residual CNN on that unfold and uses mutual information against a T1 volume already in EPI space.

This is **not** MoDL. It does not implement the Iowa MoDL paper, and it is not a knee/brain parallel-imaging demo.

## Acquisition this code assumes

Hard-coded in `saved_sf2.py` for the GE SMS-EPI exams this tree was written against:

| Parameter | Value |
| --- | --- |
| Matrix | 90 × 90 × 60 |
| Multiband | 6 (11 packs, FOV-shift 3) |
| Coils | 32 |
| Acquired ky | 66 |
| Time points | 450 |

Input k-space is an Orchestra-style HDF5 (`kspace.h5`) with `kspace_data/volume_*`, calibration `surface_images`, and `multiband_info/multiband_slices`. The T1 is a NIfTI.

## Training

Edit the paths in `trn.py` (`smriFilenames`, `acqFilenames`) to a T1 (EPI-space) and a `kspace.h5`, then:

```bash
python trn.py
```

`trn.py` loads coil maps and the SMS encoding from `saved_sf2.getData`, unfolds with a regularized pinv (`model.py`), and trains with `mi_customloss` (histogram MI vs the T1, plus a small magnitude MSE term). Checkpoints go under `savedModels/` (gitignored).

Useful knobs at the top of `trn.py`: `epochs`, `K` (unfold / CNN cycles), `nLayers`, `nTimepoints`, `minibatchSize`.

## Files

| File | Role |
| --- | --- |
| `trn.py` | Training entry point |
| `model.py` | SENSE/pinv unfold, residual CNN, MI loss |
| `saved_sf2.py` | k-space I/O, coil maps, SMS encoding matrix |
| `displayInv.py` | Quick look at an unfold |
| `tstDemo.py`, `supportingFunctions.py` | Leftover demo code; not used by `trn.py` |
| `saved_model.py`, `new_saved_model.py`, `saved_display.py` | Older copies of the model / display scripts |

## Dependencies

TensorFlow 2, NumPy, h5py, nibabel, scikit-image, matplotlib, tqdm. A machine-specific freeze is in `requirements.txt`.

## Contact

Joseph Hutter, `josephahutter@gmail.com`
