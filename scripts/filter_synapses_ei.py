"""
Build manifest of synapses to download: proofread + known E/I (pre).
Output: Data/synapses_to_download.csv (id_, pre_clf_type, ctr_x, ctr_y, ctr_z).

Two modes:
- --use_cave: query CAVE synapse_query(pre_ids=...) in batches. No 47GB CSV needed.
- default: stream Data/synapses_pni_2.csv (requires download).
Requires: proofreading_status_public_release.csv, pre_root_id_to_ei.csv
"""

import argparse
import os

import pandas as pd


def load_proofread_roots(proof_csv: str) -> set[int]:
    df = pd.read_csv(proof_csv)
    if "pt_root_id" not in df.columns:
        df.columns = [
            "id", "valid", "pt_x", "pt_y", "pt_z", "pt_supervoxel_id",
            "pt_root_id", "valid_id", "status_dendrite", "status_axon"
        ]
    mask = df["status_axon"].isin(["clean", "extended"]) | df["status_dendrite"].isin(
        ["clean", "extended"]
    )
    return set(df.loc[mask, "pt_root_id"].astype("int64").tolist())


def load_ei_map(ei_csv: str) -> dict[int, str]:
    df = pd.read_csv(ei_csv)
    df = df[df["pre_clf_type"].isin(["E", "I"])]
    return dict(zip(df["pt_root_id"].astype("int64"), df["pre_clf_type"]))


def main():
    parser = argparse.ArgumentParser(
        description="Filter synapse table to proofread synapses with known E/I (pre)."
    )
    parser.add_argument(
        "--synapse_csv",
        default="Data/synapses_pni_2.csv",
        help="Full path to synapses_pni_2.csv (streamed in chunks).",
    )
    parser.add_argument(
        "--proof_csv",
        default="Data/proofreading_status_public_release.csv",
        help="Proofreading status CSV.",
    )
    parser.add_argument(
        "--ei_csv",
        default="Data/pre_root_id_to_ei.csv",
        help="Pre root_id -> E/I from fetch_pre_ei_map.py.",
    )
    parser.add_argument(
        "--out_csv",
        default="Data/synapses_to_download.csv",
        help="Output: id_, pre_clf_type, ctr_x, ctr_y, ctr_z.",
    )
    parser.add_argument(
        "--max_synapses",
        type=int,
        default=None,
        help="Max number of synapses to output (default: all).",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=500_000,
        help="Chunk size for streaming synapse CSV.",
    )
    parser.add_argument(
        "--use_cave",
        action="store_true",
        help="Query synapses from CAVE (synapse_query) instead of CSV. No 47GB download.",
    )
    parser.add_argument(
        "--pre_id_batch",
        type=int,
        default=100,
        help="When --use_cave: number of pre root IDs per CAVE query batch.",
    )
    parser.add_argument(
        "--bounds",
        type=str,
        default=None,
        help="Only output synapses inside volume: x_min,x_max,y_min,y_max,z_min,z_max (voxels). Leaves margin for default crop.",
    )
    parser.add_argument(
        "--slab_demo",
        action="store_true",
        help="With --bounds: query slab by bbox and assign dummy E/I (no proofread/EI filter). For pipeline test.",
    )
    parser.add_argument(
        "--no_proofread",
        action="store_true",
        help="Use all E/I cells (do not restrict to proofread). Use with --max_e/--max_i for scale (e.g. 65k E, 8k I).",
    )
    parser.add_argument(
        "--max_e",
        type=int,
        default=None,
        help="Cap number of E synapses in output (e.g. 65000).",
    )
    parser.add_argument(
        "--max_i",
        type=int,
        default=None,
        help="Cap number of I synapses in output (e.g. 7892).",
    )
    args = parser.parse_args()

    for p in [args.ei_csv]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Required file not found: {p}")
    if not args.no_proofread and not os.path.isfile(args.proof_csv):
        raise FileNotFoundError(f"Required file not found: {args.proof_csv}")
    if not args.use_cave and not os.path.isfile(args.synapse_csv):
        raise FileNotFoundError(
            f"Synapse CSV not found: {args.synapse_csv}. Download it or use --use_cave to query CAVE instead."
        )

    proof_roots = load_proofread_roots(args.proof_csv) if os.path.isfile(args.proof_csv) else set()
    ei_map = load_ei_map(args.ei_csv)
    print(f"Proofread roots: {len(proof_roots)}; pre E/I known: {len(ei_map)}")

    # Optional volume bounds (x_min,x_max,y_min,y_max,z_min,z_max); margin for crop 256x256 xy, 64 z
    bounds_margin_xy, bounds_margin_z = 128, 32
    bbox_api = None  # for CAVE bounding_box query when --bounds set
    if args.bounds:
        parts = [int(x.strip()) for x in args.bounds.split(",")]
        if len(parts) != 6:
            raise ValueError("--bounds must be x_min,x_max,y_min,y_max,z_min,z_max")
        x_min, x_max, y_min, y_max, z_min, z_max = parts
        bbox_api = [[x_min, y_min, z_min], [x_max, y_max, z_max]]
        x_lo, x_hi = x_min + bounds_margin_xy, x_max - bounds_margin_xy
        y_lo, y_hi = y_min + bounds_margin_xy, y_max - bounds_margin_xy
        z_lo, z_hi = z_min + bounds_margin_z, z_max - bounds_margin_z
        in_bounds = lambda cx, cy, cz: (x_lo <= cx <= x_hi and y_lo <= cy <= y_hi and z_lo <= cz <= z_hi)
        print(f"Filtering to volume bounds (with margin): x=[{x_lo},{x_hi}] y=[{y_lo},{y_hi}] z=[{z_lo},{z_hi}]")
    else:
        in_bounds = lambda cx, cy, cz: True

    rows = []

    if args.use_cave:
        try:
            from caveclient import CAVEclient
            import time
            import requests
        except ImportError:
            raise SystemExit("--use_cave requires: pip install caveclient (and token setup).")
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                client = CAVEclient("minnie65_public")
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
        if bbox_api is not None:
            # Query all synapses in volume bounds, then filter to proofread + E/I (or slab_demo: dummy E/I)
            print("Querying CAVE synapse_query(bounding_box=...) for synapses in volume...")
            syn_df = client.materialize.synapse_query(
                bounding_box=bbox_api,
                split_positions=True,
            )
            if syn_df is not None and len(syn_df) > 0:
                for idx, (_, r) in enumerate(syn_df.iterrows()):
                    if not args.slab_demo:
                        pre_rid = int(r["pre_pt_root_id"])
                        if pre_rid not in ei_map or ei_map.get(pre_rid) not in ("E", "I"):
                            continue
                        if pre_rid not in proof_roots and int(r["post_pt_root_id"]) not in proof_roots:
                            continue
                    else:
                        pre_rid = int(r["pre_pt_root_id"])
                        # Dummy E/I for slab demo (alternate so dataset has both classes)
                        pre_clf = "E" if (pre_rid % 2 == 0) else "I"
                    pos = r.get("ctr_pt_position")
                    if pos is None and "ctr_pt_position_x" in r.index:
                        cx = r["ctr_pt_position_x"]
                        cy = r["ctr_pt_position_y"]
                        cz = r["ctr_pt_position_z"]
                    elif hasattr(pos, "__len__") and len(pos) >= 3:
                        cx, cy, cz = float(pos[0]), float(pos[1]), float(pos[2])
                    else:
                        continue
                    if not in_bounds(cx, cy, cz):
                        continue
                    rows.append({
                        "id_": int(r["id"]),
                        "pre_clf_type": pre_clf if args.slab_demo else ei_map[pre_rid],
                        "ctr_x": cx, "ctr_y": cy, "ctr_z": cz,
                        "pre_pt_root_id": int(r["pre_pt_root_id"]),
                        "post_pt_root_id": int(r["post_pt_root_id"]),
                    })
                    if args.max_synapses is not None and len(rows) >= args.max_synapses:
                        break
            else:
                print("No synapses returned from bounding_box query.")
        else:
            if args.no_proofread:
                pre_with_ei = list(ei_map.keys())
                print(f"Pre cells (all E/I, no proofread): {len(pre_with_ei)}")
            else:
                pre_with_ei = list(proof_roots & set(ei_map.keys()))
                print(f"Pre cells (proofread + E/I): {len(pre_with_ei)}")
            if not pre_with_ei:
                print("No pre-synaptic cells with E/I. Check E/I CSV (run fetch_pre_ei_map.py --no_balance).")
            else:
                for i in range(0, len(pre_with_ei), args.pre_id_batch):
                    batch = pre_with_ei[i : i + args.pre_id_batch]
                    for _attempt in range(max_attempts):
                        try:
                            syn_df = client.materialize.synapse_query(
                                pre_ids=batch,
                                split_positions=True,
                            )
                            break
                        except requests.exceptions.HTTPError as e:
                            if e.response.status_code in (502, 503, 429):
                                wait = 2 **_attempt
                                time.sleep(wait)
                            else:
                                raise
                    else:
                        raise SystemExit("CAVE synapse_query unavailable after retries.")
                    if syn_df is None or len(syn_df) == 0:
                        continue
                    for _, r in syn_df.iterrows():
                        sid = int(r["id"])
                        pre_rid = int(r["pre_pt_root_id"])
                        pre_clf = ei_map.get(pre_rid)
                        if pre_clf not in ("E", "I"):
                            continue
                        pos = r.get("ctr_pt_position")
                        if pos is None and "ctr_pt_position_x" in r.index:
                            cx = r["ctr_pt_position_x"]
                            cy = r["ctr_pt_position_y"]
                            cz = r["ctr_pt_position_z"]
                        elif hasattr(pos, "__len__") and len(pos) >= 3:
                            cx, cy, cz = float(pos[0]), float(pos[1]), float(pos[2])
                        else:
                            continue
                        if not in_bounds(cx, cy, cz):
                            continue
                        rows.append({"id_": sid, "pre_clf_type": pre_clf, "ctr_x": cx, "ctr_y": cy, "ctr_z": cz,
                                     "pre_pt_root_id": int(r["pre_pt_root_id"]), "post_pt_root_id": int(r["post_pt_root_id"])})
                        if args.max_synapses is not None and len(rows) >= args.max_synapses:
                            break
                    if args.max_synapses is not None and len(rows) >= args.max_synapses:
                        break
                    # Early exit when we have enough per class for --max_e/--max_i
                    if args.max_e is not None or args.max_i is not None:
                        n_e = sum(1 for r in rows if r["pre_clf_type"] == "E")
                        n_i = sum(1 for r in rows if r["pre_clf_type"] == "I")
                        if (args.max_e is None or n_e >= args.max_e) and (args.max_i is None or n_i >= args.max_i):
                            break
    else:
        first_chunk = pd.read_csv(args.synapse_csv, nrows=1)
        if "ctr_pt_position_x" in first_chunk.columns:
            pos_cols = ["ctr_pt_position_x", "ctr_pt_position_y", "ctr_pt_position_z"]
        elif "ctr_pt_position" in first_chunk.columns:
            pos_cols = ["ctr_pt_position"]
        else:
            raise ValueError("synapses_pni_2.csv must have ctr_pt_position or ctr_pt_position_x/y/z")
        usecols = ["id", "pre_pt_root_id", "post_pt_root_id"] + pos_cols
        total = 0
        for chunk in pd.read_csv(args.synapse_csv, usecols=usecols, chunksize=args.chunk_size):
            pre_ok = chunk["pre_pt_root_id"].isin(proof_roots)
            post_ok = chunk["post_pt_root_id"].isin(proof_roots)
            proofread = chunk[pre_ok | post_ok]
            pre_has_ei = proofread["pre_pt_root_id"].isin(ei_map)
            sub = proofread[pre_has_ei]
            for _, r in sub.iterrows():
                sid = int(r["id"])
                pre_clf = ei_map[int(r["pre_pt_root_id"])]
                if len(pos_cols) == 3:
                    cx, cy, cz = r["ctr_pt_position_x"], r["ctr_pt_position_y"], r["ctr_pt_position_z"]
                else:
                    pos = r["ctr_pt_position"]
                    if isinstance(pos, str):
                        pos = [float(x) for x in pos.strip("[]").split()]
                    cx, cy, cz = pos[0], pos[1], pos[2]
                if not in_bounds(cx, cy, cz):
                    continue
                rows.append({"id_": sid, "pre_clf_type": pre_clf, "ctr_x": cx, "ctr_y": cy, "ctr_z": cz,
                             "pre_pt_root_id": int(r["pre_pt_root_id"]), "post_pt_root_id": int(r["post_pt_root_id"])})
                total += 1
                if args.max_synapses is not None and total >= args.max_synapses:
                    break
            if args.max_synapses is not None and total >= args.max_synapses:
                break

    # Cap by class if --max_e / --max_i (e.g. 65k E, 8k I)
    if rows and (args.max_e is not None or args.max_i is not None):
        import random
        rng = random.Random(42)
        e_rows = [r for r in rows if r["pre_clf_type"] == "E"]
        i_rows = [r for r in rows if r["pre_clf_type"] == "I"]
        if args.max_e is not None:
            rng.shuffle(e_rows)
            e_rows = e_rows[: args.max_e]
        if args.max_i is not None:
            rng.shuffle(i_rows)
            i_rows = i_rows[: args.max_i]
        rows = e_rows + i_rows
        rng.shuffle(rows)

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(out_df)} synapses -> {args.out_csv}")
    if len(out_df):
        print(out_df["pre_clf_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
