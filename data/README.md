# Data

The code expects the prepared TreeTM splits:

```
data/TreeTM_datasets/wos/prepared.npz      data/TreeTM_datasets/wos/meta.json
data/TreeTM_datasets/alerce/prepared.npz   data/TreeTM_datasets/alerce/meta.json
```

Alternatively, set `GGTH_DATA=/path/to/TreeTM_datasets`.

`prepared.npz` holds the Booleanized inputs (`Btr`, `Bva`, `Bte`, uint8) and leaf labels (`ytr`, `yva`, `yte`, int64). Both datasets use stratified splits: 80% train and 20% test, with 10% of the training split held out for validation (seed 0).

| Dataset | Bits | Classes | Train / Val / Test | Encoding |
|---|---|---|---|---|
| WOS | 2000 | 133 | 33,829 / 3,759 / 9,397 | binary bag of words, 2000 most frequent training words, English stop words removed |
| ALeRCE | 1172 | 15 | 88,916 / 9,880 / 24,700 | thermometer code over seven quantile levels, plus missing-value bits |

**Sources:**
- **WOS:** Web of Science WOS-46985 (Kowsari et al., ICMLA 2017). Mendeley Data, DOI 10.17632/9rw3vkcfy4.6, CC BY 4.0.
- **ALeRCE:** light-curve classifier labeled set (Sánchez-Sáez et al., AJ 2021). Zenodo, DOI 10.5281/zenodo.4279623, CC BY 4.0.

**Download:** the prepared splits are a release asset of this repository (both datasets are CC BY 4.0, redistributed with attribution):

```bash
curl -L -o gg-th_datasets.zip https://github.com/leonid-yavits/gradient-guided-TM/releases/download/data-v1/gg-th_datasets.zip
unzip gg-th_datasets.zip -d data/      # creates data/TreeTM_datasets/{wos,alerce}
```

SHA-256 of `gg-th_datasets.zip`: `2fced0055421d2f8f62b283858d035420d51f990de3d67234027b0efecc40182`. Per-file checksums are in `TreeTM_datasets/SHA256SUMS`.

`fptm/export_bin.py` writes the raw binaries for the FPTM driver to `data/bin/`. These are not tracked by git.
