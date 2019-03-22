import sys
import random
import _pickle
import argparse
import xarray as xr
from time import clock

import numpy as np

import xbin
from derp_learning.util import InProcessExecutor, fnames_from_arg_list, cpu_count


def load(fname):
    with open(fname, "rb") as f:
        return _pickle.load(f)


def build_dataset_from_files(fnames):
    if len(fnames) == 0:
        print("no files specified")
        sys.exit()

    raw = [load(fname) for fname in fnames]

    # per structure
    pdb = [x["fname"] for x in raw]
    chains = [x["chains"] for x in raw]
    com = np.stack([x["coords"]["com"] for x in raw])
    rg = np.array([x["coords"]["rg"] for x in raw])
    nres = np.array([len(x["coords"]["cb"]) for x in raw])

    # per res
    ncac = np.concatenate([x["coords"]["ncac"] for x in raw])
    cb = np.concatenate([x["coords"]["cb"] for x in raw])
    stubs = np.concatenate([x["coords"]["stubs"] for x in raw])

    resdata = dict()
    resdata["pdbno"] = [np.repeat(i, len(x["coords"]["cb"])) for i, x in enumerate(raw)]
    resdata["pdbno"] = np.concatenate(resdata["pdbno"]).astype("i4")
    resdata["resno"] = [np.arange(len(x["coords"]["cb"])) for x in raw]
    resdata["resno"] = np.concatenate(resdata["resno"]).astype("i4")
    for name in raw[0]["resdata"].keys():
        if type(raw[0]["resdata"][name]) is str:
            dat = np.concatenate([list(x["resdata"][name]) for x in raw])
        else:
            dat = np.concatenate([x["resdata"][name] for x in raw])
        resdata[name] = dat
    resdata = xr.Dataset(resdata)

    pdb_res_offsets = np.cumsum([len(x["resdata"]["phi"]) for x in raw])
    pdb_res_offsets[1:] = pdb_res_offsets[:-1]
    pdb_res_offsets[0] = 0
    sanity_check_res_data(**vars())

    pairdata = dict()
    for name in raw[0]["pairdata"]:
        pairdata[name] = np.concatenate([x["pairdata"][name] for x in raw])
    pairdata["resi"] = pairdata["resi"].astype("i4")
    pairdata["resj"] = pairdata["resj"].astype("i4")

    pdb_pair_offsets = np.cumsum([len(x["pairdata"]["dist"]) for x in raw])
    pdb_pair_offsets[1:] = pdb_pair_offsets[:-1]
    pdb_pair_offsets[0] = 0
    tmp = [np.repeat(i, len(x["pairdata"]["dist"])) for i, x in enumerate(raw)]
    pairdata["pdbno"] = np.concatenate(tmp).astype("i4")
    sanity_check_pair_data(**vars())

    tot_nres = len(resdata["phi"])
    tot_pairs = len(pairdata["dist"])
    print(f"nstruct {len(pdb):,} nres {tot_nres:,} npairs {tot_pairs:,}")

    pair_resi_idx = pdb_res_offsets[pairdata["pdbno"]] + pairdata["resi"]
    pair_resj_idx = pdb_res_offsets[pairdata["pdbno"]] + pairdata["resj"]

    pair_stubi = stubs[pair_resi_idx]
    pair_stubj = stubs[pair_resj_idx]
    xreli = np.linalg.inv(pair_stubi) @ pair_stubj
    xrelj = np.linalg.inv(pair_stubj) @ pair_stubi
    assert np.allclose(np.linalg.inv(xreli), xrelj, atol=1e-3)  # floats kinda suck...

    cbi = cb[pair_resi_idx]
    cbj = cb[pair_resj_idx]
    dhat = np.linalg.norm(cbi - cbj, axis=1)
    assert np.allclose(dhat, pairdata["dist"], atol=1e-3)

    cart_resl = [0.5, 1.0, 2.0]
    ori_resl = [7.5, 15, 30]
    hash_params = [(c, o) for c in cart_resl for o in ori_resl]

    t = clock()
    indexers = [
        xbin.gu_xbin_indexer(cart_resl=2.0, ori_resl=15.0) for c, o in hash_params
    ]
    print("numba compile time", clock() - t)
    t = clock()
    for i, indexer in enumerate(indexers):
        pairdata["xibin" + str(i)] = indexer(xreli)
        pairdata["xjbin" + str(i)] = indexer(xrelj)
    print("binning time", clock() - t)

    # neighbors within radii???

    return


# (XA)inv * (XB)
# Ainv * Xinv * X * B = Ainv * B
# A = subs[]


def sanity_check_res_data(nres, pdb_res_offsets, resdata, **kw):
    for i, o in enumerate(pdb_res_offsets):
        lb = pdb_res_offsets[i]
        ub = (
            pdb_res_offsets[i + 1]
            if i + 1 < len(pdb_res_offsets)
            else len(resdata["phi"])
        )
        assert np.all(resdata["pdbno"][lb:ub] == i)
        assert np.all(resdata["resno"][lb:ub] < nres[i])
        assert np.all(resdata["resno"][lb] == 0)


def sanity_check_pair_data(nres, pdb_pair_offsets, pairdata, **kw):
    for i, o in enumerate(pdb_pair_offsets):
        pdbi = pairdata["pdbno"] == i
        for tmp in ("resi", "resj"):
            residx = pairdata[tmp][pdbi]
            assert max(residx) < nres[i]
            assert min(residx) == 0 or tmp == "resj"
            lb = pdb_pair_offsets[i]
            ub = (
                pdb_pair_offsets[i + 1]
                if i + 1 < len(pdb_pair_offsets)
                else len(pairdata["dist"])
            )
            residx = pairdata[tmp][lb:ub]
            assert max(residx) < nres[i]
            assert min(residx) == 0 or tmp == "resj"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", default=-1, type=int)
    parser.add_argument("pdbs", nargs="*")
    args = parser.parse_args(sys.argv[1:])
    fnames = fnames_from_arg_list(args.pdbs, [".pickle"])
    build_dataset_from_files(fnames)


if __name__ == "__main__":
    main()