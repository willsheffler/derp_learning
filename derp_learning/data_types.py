import os

import xarray as xr
import numpy as np


class ResPairData:
    def __init__(self, data, sanity_check=None):

        if isinstance(data, xr.Dataset):
            self.data = data
        elif isinstance(data, ResPairData):
            self.data = data.data
        if hasattr(self, "data"):
            if sanity_check is True:
                self.sanity_check()
            return

        # loading from raw dicts
        assert isinstance(data, dict)

        raw = data
        pdbdata = raw["pdbdata"]
        coords = raw["coords"]
        resdata = raw["resdata"]
        pairdata = raw["pairdata"]
        pdb_res_offsets = raw["pdb_res_offsets"]
        pdb_pair_offsets = raw["pdb_pair_offsets"]
        bin_params = raw["bin_params"]

        # put this stuff in dataset

        self.res_ofst = pdb_res_offsets
        self.pair_ofst = pdb_pair_offsets
        self.bin_params = bin_params

        pdbdata["file"] = pdbdata["pdb"]
        pdbdata["pdb"] = _get_pdb_names(pdbdata["file"])
        for k, v in pdbdata.items():
            pdbdata[k] = (["pdbid"], v)
        pdbdata["com"] = (["pdbid", "xyzw"], pdbdata["com"][1])

        for k, v in resdata.items():
            resdata[k] = (["resid"], v)
        # resdata["n"] = (["resid", "xyzw"], coords["ncac"][:, 0])
        # resdata["ca"] = (["resid", "xyzw"], coords["ncac"][:, 1])
        # resdata["c"] = (["resid", "xyzw"], coords["ncac"][:, 2])
        resdata["n"] = (["resid", "xyzw"], coords["n"])
        resdata["ca"] = (["resid", "xyzw"], coords["ca"])
        resdata["c"] = (["resid", "xyzw"], coords["c"])
        resdata["o"] = (["resid", "xyzw"], coords["o"])
        resdata["cb"] = (["resid", "xyzw"], coords["cb"])
        resdata["stub"] = (["resid", "hrow", "hcol"], coords["stubs"])
        resdata["r_pdbid"] = resdata["pdbno"]
        del resdata["pdbno"]

        for k, v in pairdata.items():
            pairdata[k] = (["pairid"], v)
        pairdata["p_pdbid"] = pairdata["pdbno"]
        del pairdata["pdbno"]

        if len(pdbdata.keys() & resdata.keys()):
            print(pdbdata.keys() & resdata.keys())
            assert 0 == len(pdbdata.keys() & resdata.keys())
        if len(pdbdata.keys() & pairdata.keys()):
            print(pdbdata.keys() & pairdata.keys())
            assert 0 == len(pdbdata.keys() & pairdata.keys())
        if len(pairdata.keys() & resdata.keys()):
            print(pairdata.keys() & resdata.keys())
            assert 0 == len(pairdata.keys() & resdata.keys())

        data = {**pdbdata, **resdata, **pairdata}
        assert len(data) == len(pdbdata) + len(resdata) + len(pairdata)
        self.data = xr.Dataset(
            data,
            coords=dict(
                xyzw=["x", "y", "z", "w"],
                hrow=["x", "y", "z", "w"],
                hcol=["x", "y", "z", "t"],
            ),
            attrs=dict(
                pdb_res_offsets=pdb_res_offsets,
                pdb_pair_offsets=pdb_pair_offsets,
                xbin_params=bin_params,
                xbin_types=raw["xbin_types"],
                xbin_swap_type=raw["xbin_swap_type"],
                eweights=raw["eweights"],
            ),
        )

        self.change_seq_ss_to_ids()

        res_ofst = self.pdb_res_offsets[self.p_pdbid]
        self.data["p_resi"] += res_ofst
        self.data["p_resj"] += res_ofst

        assert self.data.stub.sel(hcol="t").shape[1] == 4
        assert np.all(self.data.ca.sel(xyzw="w") == 1)
        assert np.all(self.data.cb.sel(xyzw="w") == 1)

        if sanity_check is not False:
            self.sanity_check()

    def __getattr__(self, k):
        if k == "data":
            raise AttributeError
        return getattr(self.data, k)

    def __getitem__(self, k):
        return self.data[k]

    def __str__(self):
        return "ResPairData with data = " + str(self.data).replace("\n", "\n  ")

    def change_seq_ss_to_ids(self):
        dat = self.data

        id2aa = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
        aa2id = xr.DataArray(np.arange(20, dtype="i4"), [("aa", id2aa)])
        aaid = aa2id.sel(aa=dat.seq).values.astype("i4")
        dat["aaid"] = xr.DataArray(aaid, dims=["resid"])
        dat = dat.drop("seq")
        dat["id2aa"] = xr.DataArray(id2aa, [aa2id], ["aai"])
        dat["aa2id"] = xr.DataArray(aa2id, [id2aa], ["aa"])

        id2ss = np.array(list("EHL"))
        ss2id = xr.DataArray(np.arange(3, dtype="i4"), [("ss", id2ss)])
        ssid = ss2id.sel(ss=dat.ss).values.astype("i4")
        dat["ssid"] = xr.DataArray(ssid, dims=["resid"])
        dat = dat.drop("ss")
        dat["id2ss"] = xr.DataArray(id2ss, [ss2id], ["ssi"])
        dat["ss2id"] = xr.DataArray(ss2id, [id2ss], ["ss"])
        self.data = dat

    def subset(self, keep=None, random=True, sanity_check=False):
        if isinstance(keep, (int, np.int32, np.int64)):
            if random:
                keep = np.random.choice(len(self.pdb), keep, replace=False)
            else:
                keep = np.arange(keep)
        else:
            keep = np.array(list(keep))
        rp = self.data
        pdbs = rp.pdb[keep].values
        mask = np.isin(rp.pdb, pdbs)
        old2new = np.zeros(len(rp.pdb), "i4") - 1
        old2new[keep] = np.arange(len(keep))
        residx = np.isin(rp.r_pdbid, keep)
        pairidx = np.isin(rp.p_pdbid, keep)
        rpsub = rp.sel(pdbid=keep, resid=residx, pairid=pairidx)

        # update relational stuff
        rpsub.r_pdbid.values = old2new[rpsub.r_pdbid]
        rpsub.p_pdbid.values = old2new[rpsub.p_pdbid]
        new_pdb_res_offsets = np.concatenate([[0], np.cumsum(rpsub.nres)])
        rpsub.attrs["pdb_res_offsets"] = new_pdb_res_offsets
        tmp = np.cumsum(rpsub.p_pdbid.groupby(rpsub.p_pdbid).count())
        new_pdb_pair_offsets = np.concatenate([[0], tmp])
        rpsub.attrs["pdb_pair_offsets"] = new_pdb_pair_offsets
        old_res_ofst = rp.pdb_res_offsets[keep[rpsub.p_pdbid]]
        new_res_ofst = new_pdb_res_offsets[rpsub.p_pdbid]
        rpsub["p_resi"] += new_res_ofst - old_res_ofst
        rpsub["p_resj"] += new_res_ofst - old_res_ofst

        rp = ResPairData(rpsub)
        if sanity_check:
            rp.sanity_check()
        return rp

    def split(self, frac, random=True, **kw):
        n1 = int(len(self.pdb) * frac)
        n2 = len(self.pdb) - n1
        if random:
            part1 = np.random.choice(len(self.pdb), n1, replace=False)
            part2 = np.array(list(set(range(len(self.pdb))) - set(part1)))
            np.random.shuffle(part2)
        else:
            part1 = np.arange(n1)
            part2 = np.arange(n1, len(self.pdb))
        parts = [part1, part2]
        return [self.subset(part, **kw) for part in parts]

    def sanity_check(self):
        rp = self.data
        Npdb = len(self.pdb)
        for ipdb in np.random.choice(Npdb, min(Npdb, 50), replace=False):
            rlb, rub = rp.pdb_res_offsets[ipdb : ipdb + 2]
            if rlb > 0:
                assert rp.r_pdbid[rlb - 1] == ipdb - 1
            assert rp.r_pdbid[rlb] == ipdb
            assert rp.r_pdbid[rub - 1] == ipdb
            if ipdb + 1 < len(rp.pdb):
                assert rp.r_pdbid[rub] == ipdb + 1
            p_resi = rp.resno[rp.r_pdbid == ipdb]
            assert np.min(p_resi) == 0
            assert np.max(p_resi) == rp.nres[ipdb] - 1
            plb, pub = rp.pdb_pair_offsets[ipdb : ipdb + 2]
            p_resi = rp.p_resi[plb:pub] - rp.pdb_res_offsets[ipdb]
            p_resj = rp.p_resj[plb:pub] - rp.pdb_res_offsets[ipdb]
            assert np.min(p_resi) == 0
            assert np.max(p_resi) < rp.nres[ipdb]
            assert 0 < np.min(p_resj) < rp.nres[ipdb]

            resi_this_ipdb = rp.p_resi[rp.p_pdbid == ipdb]
            assert np.all(rp.r_pdbid[resi_this_ipdb] == ipdb)
            if np.min(resi_this_ipdb) - 1 >= 0:
                assert rp.r_pdbid[np.min(resi_this_ipdb) - 1] != ipdb

        assert np.all(rp.p_resi - rp.pdb_res_offsets[rp.p_pdbid] == rp.resno[rp.p_resi])

        # sanity check pair distances vs res-res distances
        cbi = rp.cb[rp.p_resi]
        cbj = rp.cb[rp.p_resj]
        dhat = np.linalg.norm(cbi - cbj, axis=1)
        assert np.allclose(dhat, rp.p_dist, atol=1e-3)

        assert np.max(rp.p_resi) < len(rp.resid)
        assert np.max(rp.p_resj) < len(rp.resid)


def _get_pdb_names(files):
    base = [os.path.basename(f) for f in files]
    assert all(b[4:] == "_0001.pdb" for b in base)
    return [b[:4] for b in base]
