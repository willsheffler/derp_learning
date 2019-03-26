import os
import _pickle
import random
import numpy as np
import xarray as xr
import numba as nb
from derp_learning.data_types import ResPairData


def load_which_takes_forever():
    with open("pdb_res_pair_data.pickle", "rb") as inp:
        return _pickle.load(inp)


if hasattr(os, "a_very_unique_name"):
    respairdat = os.a_very_unique_name
else:
    respairdat = load_which_takes_forever()
    os.a_very_unique_name = respairdat

if "ppdbidp" in respairdat.data:
    print("renaming ppdbidp")
    respairdat.data.rename(inplace=True, ppdbidp="pdbidp")


if "seq" in respairdat.data:
    id2aa = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
    aa2id = xr.DataArray(np.arange(20, dtype="i4"), [("aa", id2aa)])
    aaid = aa2id.sel(aa=respairdat.seq).values.astype("i4")
    respairdat.data["aaid"] = xr.DataArray(aaid, dims=["resid"])
    respairdat.data = respairdat.data.drop("seq")
    respairdat.data["id2aa"] = xr.DataArray(id2aa, [aa2id], ["aai"])
    respairdat.data["aa2id"] = xr.DataArray(aa2id, [id2aa], ["aa"])

    id2ss = np.array(list("EHL"))
    ss2id = xr.DataArray(np.arange(3, dtype="i4"), [("ss", id2ss)])
    ssid = ss2id.sel(ss=respairdat.ss).values.astype("i4")
    respairdat.data["ssid"] = xr.DataArray(ssid, dims=["resid"])
    respairdat.data = respairdat.data.drop("ss")
    respairdat.data["id2ss"] = xr.DataArray(id2ss, [ss2id], ["ssi"])
    respairdat.data["ss2id"] = xr.DataArray(ss2id, [id2ss], ["ss"])

print(respairdat)


import sys

sys.exit()


nb.njit(nogil=True, fastmath=True)


def identical_ranges(x):
    assert x.ndim == 1
    prev = -1
    vals = np.zeros(len(x), dtype=np.int64) - 1
    ranges = np.zeros((len(x), 2), dtype=np.int32) - 1
    n = 0
    for i in range(len(x)):
        if x[i] != prev:
            ranges[n][0] = i
            vals[n] = x[i]
            if n > 0:
                ranges[n - 1][1] = i
            n += 1
            prev = x[i]
    ranges[n - 1, 1] = len(x)
    assert n == len(set(x))
    return vals[:n].copy(), ranges[:n].copy()


def main():
    rp = respairdat.subset(10)
    print("sanity_check")
    rp.sanity_check()

    bintype = "xijbin_2.0_30"
    bins = rp.data[bintype].values
    print("argsort bin")
    order = np.argsort(bins)
    sbins = bins[order]
    print("identical_ranges")
    vals, ranges = identical_ranges(sbins)
    ninbin = ranges[:, 1] - ranges[:, 0]
    assert np.all(ninbin > 0)
    print("frac not one", np.sum(ninbin != 1) / len(ranges))
    print(ninbin[ninbin != 1])

    print(ranges.shape)
    binner = xr.DataArray(ranges, [("binidx", vals), ("lbub", ["lb", "ub"])])
    print(binner)
    sel = binner.sel(binidx=sbins)
    print(ranges.shape)
    print(sel.shape)

    for i in np.random.choice(len(bins), 10):
        lb, ub = binner.sel(binidx=bins[i]).values
        assert np.all(rp.data[bintype][order[lb:ub]] == bins[i])

    binsp = xr.DataArray(np.zeros(len(vals), 20))


if __name__ == "__main__":
    main()
