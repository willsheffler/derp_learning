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
        resdata["pdbidr"] = resdata["pdbno"]
        del resdata["pdbno"]

        for k, v in pairdata.items():
            pairdata[k] = (["pairid"], v)
        pairdata["pdbidp"] = pairdata["pdbno"]
        del pairdata["pdbno"]

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
            ),
        )

        self.change_seq_ss_to_ids()

        res_ofst = self.pdb_res_offsets[self.pdbidp]
        self.data["resi"] += res_ofst
        self.data["resj"] += res_ofst

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
        if "ppdbidp" in dat:
            print("renaming ppdbidp to pdbidp")
            dat.rename(inplace=True, ppdbidp="pdbidp")

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
                keep = sorted(np.random.choice(len(self.pdb), keep, replace=False))
            else:
                keep = np.arange(keep)
        rp = self.data
        pdbs = rp.pdb[keep].values
        mask = np.isin(rp.pdb, pdbs)
        old2new = np.zeros(len(rp.pdb), "i4") - 1
        old2new[keep] = np.arange(len(keep))
        residx = np.isin(rp.pdbidr, keep)
        pairidx = np.isin(rp.pdbidp, keep)
        rpsub = rp.sel(pdbid=keep, resid=residx, pairid=pairidx)

        # update relational stuff
        rpsub.pdbidr.values = old2new[rpsub.pdbidr]
        rpsub.pdbidp.values = old2new[rpsub.pdbidp]
        rpsub.attrs["pdb_res_offsets"] = np.concatenate([[0], np.cumsum(rpsub.nres)])
        tmp = np.cumsum(rpsub.pdbidp.groupby(rpsub.pdbidp).count())
        rpsub.attrs["pdb_pair_offsets"] = np.concatenate([[0], tmp])
        old_res_ofst = rp.pdb_res_offsets[rpsub.pdbidp]
        new_res_ofst = rpsub.pdb_res_offsets[rpsub.pdbidp]
        rpsub["resi"] += new_res_ofst - new_res_ofst
        rpsub["resj"] += new_res_ofst - new_res_ofst

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
        parts = [part1, part2]
        return [self.subset(part, **kw) for part in parts]

    def sanity_check(self):
        rp = self.data
        Npdb = len(self.pdb)
        for ipdb in np.random.choice(Npdb, min(Npdb, 100), replace=False):
            rlb, rub = rp.pdb_res_offsets[ipdb : ipdb + 2]
            if rlb > 0:
                assert rp.pdbidr[rlb - 1] == ipdb - 1
            assert rp.pdbidr[rlb] == ipdb
            assert rp.pdbidr[rub - 1] == ipdb
            if ipdb + 1 < len(rp.pdb):
                assert rp.pdbidr[rub] == ipdb + 1
            resi = rp.resno[rp.pdbidr == ipdb]
            assert np.min(resi) == 0
            assert np.max(resi) == rp.nres[ipdb] - 1
            plb, pub = rp.pdb_pair_offsets[ipdb : ipdb + 2]
            resi = rp.resi[plb:pub] - rp.pdb_res_offsets[ipdb]
            resj = rp.resj[plb:pub] - rp.pdb_res_offsets[ipdb]
            assert np.min(resi) == 0
            assert np.max(resi) < rp.nres[ipdb]
            assert 0 < np.min(resj) < rp.nres[ipdb]

        assert np.all(rp.resi - rp.pdb_res_offsets[rp.pdbidp] == rp.resno[rp.resi])

        # sanity check pair distances vs res-res distances
        cbi = rp.cb[rp.resi]
        cbj = rp.cb[rp.resj]
        dhat = np.linalg.norm(cbi - cbj, axis=1)
        assert np.allclose(dhat, rp.dist, atol=1e-3)


def _get_pdb_names(files):
    base = [os.path.basename(f) for f in files]
    assert all(b[4:] == "_0001.pdb" for b in base)
    return [b[:4] for b in base]
