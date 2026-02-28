"""
Fetch pre-synaptic root_id -> E/I from CAVE cell type table.
Saves Data/pre_root_id_to_ei.csv (pt_root_id, pre_clf_type) for use by filter_synapses_ei.py.

Requires: pip install caveclient
CAVE token: https://tutorial.microns-explorer.org/quickstart_notebooks/em_py_01_caveclient_setup.html
"""

import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Export pre_pt_root_id -> E/I from CAVE cell type table."
    )
    parser.add_argument(
        "--out_csv",
        default="Data/pre_root_id_to_ei.csv",
        help="Output CSV: pt_root_id, pre_clf_type (E or I)",
    )
    parser.add_argument(
        "--table",
        default="aibs_metamodel_celltypes_v661",
        help="CAVE table with pt_root_id and classification_system (excitatory_neuron, inhibitory_neuron, nonneuron).",
    )
    parser.add_argument(
        "--no_balance",
        action="store_true",
        help="Keep all E/I cells (do not cap to min(E,I)). Use for bbox/slab filtering.",
    )
    args = parser.parse_args()

    try:
        from caveclient import CAVEclient
    except ImportError:
        raise SystemExit(
            "CAVEclient required: pip install caveclient\n"
            "Then set up token: https://tutorial.microns-explorer.org/quickstart_notebooks/em_py_01_caveclient_setup.html"
        )

    import time
    import requests

    max_attempts = 10
    for attempt in range(max_attempts):
        try:
            client = CAVEclient("minnie65_public")
            print(f"Querying {args.table}...")
            df = client.materialize.query_table(
                args.table,
                select_columns=["pt_root_id", "classification_system"],
            )
            break
        except requests.exceptions.HTTPError as e:
            if e.response.status_code in (502, 503, 429):
                wait = 2 ** attempt
                print(f"CAVE unavailable ({e.response.status_code}), retry in {wait}s ({attempt + 1}/{max_attempts})...")
                time.sleep(wait)
            else:
                raise
    else:
        raise SystemExit("CAVE still unavailable after retries. Run again later.")
    # Keep only neurons; map to E/I
    neuron = df["classification_system"].isin(["excitatory_neuron", "inhibitory_neuron"])
    df = df.loc[neuron].copy()
    df["pre_clf_type"] = df["classification_system"].map(
        {"excitatory_neuron": "E", "inhibitory_neuron": "I"}
    )
    out = df[["pt_root_id", "pre_clf_type"]].drop_duplicates(subset=["pt_root_id"])
    if args.no_balance:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        out.to_csv(args.out_csv, index=False)
        print(f"Wrote {len(out)} root IDs (all E/I) -> {args.out_csv}")
        print(out["pre_clf_type"].value_counts().to_string())
    else:
        # Balance E and I to the smaller class size (e.g. 7892 each)
        e_ids = out.loc[out["pre_clf_type"] == "E", "pt_root_id"].tolist()
        i_ids = out.loc[out["pre_clf_type"] == "I", "pt_root_id"].tolist()
        n = min(len(e_ids), len(i_ids))
        rng = __import__("random").Random(42)
        rng.shuffle(e_ids)
        rng.shuffle(i_ids)
        balanced = pd.DataFrame(
            {"pt_root_id": e_ids[:n] + i_ids[:n], "pre_clf_type": ["E"] * n + ["I"] * n}
        )
        balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        balanced.to_csv(args.out_csv, index=False)
        print(f"Wrote {len(balanced)} root IDs (balanced {n} E + {n} I) -> {args.out_csv}")
        print(balanced["pre_clf_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
