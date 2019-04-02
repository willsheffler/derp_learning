import os
import _pickle
import random
import numpy as np
import xarray as xr
import numba as nb
from derp_learning.data_types import ResPairData
from derp_learning.khash import KHashi8i8


def load_which_takes_forever():
    with open("datafiles/pdb_res_pair_data.pickle", "rb") as inp:
        return _pickle.load(inp)


if hasattr(os, "a_very_unique_name"):
    respairdat = os.a_very_unique_name
else:
    respairdat = load_which_takes_forever()
    os.a_very_unique_name = respairdat


@nb.njit(nogil=True, fastmath=True)
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


def make_xarray_binner(rp, bintype):
    binid = rp.data[bintype].values
    bin_srange2pairid = np.argsort(binid)
    sorted_binids = binid[bin_srange2pairid]
    binkeys, binvals = identical_ranges(sorted_binids)
    bin_srange = xr.DataArray(binvals, [("binidx", binkeys), ("bound", ["lb", "ub"])])
    return bin_srange, bin_srange2pairid


def make_numba_binner(rp, bintype):
    binid = rp.data[bintype].values
    bin_srange2pairid = np.argsort(binid)
    sorted_binids = binid[bin_srange2pairid]
    binkeys, ranges = identical_ranges(sorted_binids)
    binvals = ranges[:, 1].astype("i8") + np.left_shift(ranges[:, 0].astype("i8"), 32)
    bin_srange = KHashi8i8()
    bin_srange.update2(binkeys, binvals)

    assert np.all(ranges >= 0)
    assert np.all(binvals >= 0)
    assert bin_srange.size() == len(binkeys)
    assert np.all(bin_srange.array_get(binkeys) == binvals)
    assert np.all(np.right_shift(binvals, 32) == ranges[:, 0])
    assert np.all(binvals.astype("i4") == ranges[:, 1])

    return bin_srange, bin_srange2pairid


def get_bin_info(rp, bintype):
    bin_srange, bin_srange2pairid = make_xarray_binner(rp, bintype)

    ninbin = bin_srange.sel(bound="ub") - bin_srange.sel(bound="lb")
    assert np.all(ninbin > 0)
    # print("frac not one", np.sum(ninbin != 1) / len(binvals))
    # h = np.histogram(ninbin, bins="sturges")
    ssep = rp.p_resj - rp.p_resi  # offsets cancel
    unique, counts = np.unique(ninbin, return_counts=True)
    print("bin_count: nbins_wtih_count", dict(zip(unique, counts)))

    topbins = np.argsort(-ninbin).values[:10]
    for ibin in topbins:
        sep = ssep[rp[bintype] == bin_srange.binidx[ibin]].values
        print("xbin size", ninbin[ibin].values, "seqsep", np.mean(sep))

    # sanity check, all members of bin have bind == binkey
    binid = rp[bintype]
    sel = bin_srange.sel(binidx=binid[bin_srange2pairid])
    for pairid in np.random.choice(len(rp.pairid), 100):
        lb, ub = bin_srange.sel(binidx=binid[pairid]).values
        assert np.all(rp.data[bintype][bin_srange2pairid[lb:ub]] == binid[pairid])


@nb.njit(nogil=True, fastmath=True)
def pairs_to_seqprof(
    vbinidx,
    v_p_resi,
    tbinidx,
    p_resi,
    p_resj,
    p_etot,
    aaid,
    bin_trange,
    bin_trange2pairid,
    prof,
):
    nmissing, nfound = 0, 0
    for ivpair in range(len(p_resi)):
        trange = bin_trange.get(vbinidx[ivpair])
        if trange < 0:
            nmissing += 1
            continue
        nfound += 1
        lb = np.right_shift(trange, 32)
        ub = np.int32(trange)
        v_resi = v_p_resi[ivpair]
        for itrange in range(lb, ub):
            itpair = bin_trange2pairid[itrange]
            assert tbinidx[itpair] == vbinidx[ivpair]
            aa = aaid[p_resi[itpair]]
            prof[v_resi, aa] += 1  # -p_etot[itpair]

    return nmissing, nfound


def make_seq_prof_xbins(t, v, bintype):
    prof = np.zeros((len(v.resid), 20), np.float32)
    bin_trange, bin_trange2pairid = make_numba_binner(t, bintype)
    nmissing, nfound = pairs_to_seqprof(
        v[bintype].values,
        v.p_resi.values,
        t[bintype].values,
        t.p_resi.values,
        t.p_resj.values,
        t.p_etot.values,
        t.aaid.values,
        bin_trange,
        bin_trange2pairid,
        prof,
    )
    # prof /= np.sum(prof, axis=1)[:, np.newaxis]
    print(nmissing, nfound, v.aaid[0].values, prof[0])
    aahat = np.argmax(prof, axis=1)
    frac = np.sum(aahat == v.aaid.values)
    print("sequence recovery", frac / len(aahat))


def main():

    # respairdat.sanity_check()

    rp = respairdat
    # rp = respairdat.subset(100, random=0, sanity_check=True)
    # binners = {bintype: make_numba_binner(rp, bintype) for bintype in rp.xbin_types}
    # print(binners)
    # get_bin_info(rp, "xijbin_2.0_30")

    train, valid = rp.split(0.5, random=False, sanity_check=True)
    make_seq_prof_xbins(train, valid, "xijbin_1.0_15")


if __name__ == "__main__":
    main()
