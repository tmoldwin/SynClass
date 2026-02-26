# MICrONS minnie65 synapses and hand-proofread cells

Data sources:

- **Synapse graph (all synapses)**: `Data/synapses_pni_2.csv`  
  - https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/synapse_graph/synapses_pni_2.csv  
  - ~47.5 GB; columns include `id`, `pre_pt_root_id`, `post_pt_root_id`, `ctr_pt_position_*`.
- **Proofreading status**: `Data/proofreading_status_public_release.csv`  
  - https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/proofreading_status/proofreading_status_public_release.csv
- **EM volume**: streamed via CloudVolume; do not download the full volume.

E/I (excitatory/inhibitory) for the **pre-synaptic** cell comes from CAVE cell-type tables (e.g. `aibs_metamodel_celltypes_v661`), not from the static CSVs.

---

## 1. Download the two CSVs

From the repo root (PowerShell):

```powershell
# Proofreading (small)
curl.exe -L "https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/proofreading_status/proofreading_status_public_release.csv" -o Data/proofreading_status_public_release.csv

# Synapse graph (47.5 GB)
curl.exe -L "https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/synapse_graph/synapses_pni_2.csv" -o Data/synapses_pni_2.csv
```

---

## 2. Full pipeline: E/I from CAVE → filter synapses → download images

Use only MICrONS IDs and public data: get E/I from CAVE, stream the synapse CSV to keep synapses where the **pre** has known E/I and at least one partner is proofread, then download EM crops in SynClass format and write `synapse_data.csv`.

### Dependencies

```powershell
pip install cloud-volume pandas numpy tqdm caveclient
```

CAVE token for `fetch_pre_ei_map.py`: [CAVEclient setup](https://tutorial.microns-explorer.org/quickstart_notebooks/em_py_01_caveclient_setup.html).

### Step A: Export pre-synaptic E/I from CAVE

```powershell
python scripts/fetch_pre_ei_map.py --out_csv Data/pre_root_id_to_ei.csv
```

Creates `Data/pre_root_id_to_ei.csv` (`pt_root_id`, `pre_clf_type` E/I).

### Step B: Filter synapse table to proofread + known E/I

```powershell
python scripts/filter_synapses_ei.py --out_csv Data/synapses_to_download.csv --max_synapses 5000
```

Omit `--max_synapses` to keep all matching synapses. Requires `Data/synapses_pni_2.csv`, `Data/proofreading_status_public_release.csv`, `Data/pre_root_id_to_ei.csv`.

### Step C: Download EM crops (SynClass format)

```powershell
python scripts/download_synapse_crops.py --manifest_csv Data/synapses_to_download.csv --out_dir Data/proofread_synapses --max_synapses 2000
```

Writes `{id}_syn.npy`, `{id}_pre_syn_n_mask.npy`, `{id}_post_syn_n_mask.npy` (masks zeros) and `Data/proofread_synapses/synapse_data.csv` (`id_`, `pre_clf_type`).

### Step D: Train

```powershell
python synapse_classifier_2dcnn_multichannel.py --csv_path Data/proofread_synapses/synapse_data.csv --data_dir Data/proofread_synapses --epochs 50 --batch_size 32
```

Or:

```powershell
python synapse_classifier_transformer.py --csv_path Data/proofread_synapses/synapse_data.csv --data_dir Data/proofread_synapses --epochs 50 --batch_size 32
```

---

## 3. Alternative: filter your existing CSV by proofread partners

If you already have `synapse_data.csv` with `id_`, `pre_clf_type`, `pre_id`, `post_id` and want to train only on proofread synapses (same volumes, subset of labels):

```powershell
curl.exe -L "https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/proofreading_status/proofreading_status_public_release.csv" -o Data/proofreading_status_public_release.csv
python scripts/build_proofread_ei_dataset.py
python synapse_classifier_2dcnn_multichannel.py --csv_path Data/synpase_raw_em/proofread_synapse_data.csv --epochs 50
```

This does **not** download new images; it only filters your CSV by proofread root IDs.
