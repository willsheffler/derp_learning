import os

import xarray as xr
import numpy as np


class ResPairData:
    def __init__(self, raw=None, data=None):

        if data is not None:
            self.data = data
            return

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
        resdata["bb"] = (["resid", "ncac", "xyzw"], coords["ncac"])
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
                ncac=["n", "ca", "c"],
            ),
            attrs=dict(
                pdb_res_offsets=pdb_res_offsets,
                pdb_pair_offsets=pdb_pair_offsets,
                xbin_params=bin_params,
            ),
        )

        assert self.data.stub.sel(hcol="t").shape[1] == 4
        assert np.all(self.data.cb.sel(xyzw="w") == 1)
        assert np.all(self.data.bb.sel(ncac="ca", xyzw="w") == 1)

    def __getattr__(self, k):
        if k == "data":
            raise AttributeError
        return getattr(self.data, k)

    def __str__(self):
        return "ResPairData with data = " + str(self.data).replace("\n", "\n  ")

    def subset(self, keep=None):
        rp = self.data
        pdbs = rp.pdb[keep].values
        mask = np.isin(rp.pdb, pdbs)
        old2new = np.zeros(len(rp.pdb), "i4") - 1
        old2new[keep] = np.arange(len(keep))
        residx = np.isin(rp.pdbidr, keep)
        pairidx = np.isin(rp.pdbidp, keep)
        rpsub = rp.sel(pdbid=keep, resid=residx, pairid=pairidx)
        rpsub.pdbidr.values = old2new[rpsub.pdbidr]
        rpsub.pdbidp.values = old2new[rpsub.pdbidp]
        rpsub["pdb_res_offsets"] = np.concatenate([[0], np.cumsum(rpsub.nres)])
        tmp = np.cumsum(rpsub.pdbidp.groupby(rpsub.pdbidp).count())
        rpsub["pdb_pair_offsets"] = np.concatenate([[0], tmp])
        return ResPairData(data=rpsub)

    def sanity_check(self):
        rp = self.data
        for ipdb in range(len(rp.pdb)):
            rlb, rub = rp.pdb_res_offsets.values[ipdb : ipdb + 2]
            if rlb > 0:
                assert rp.pdbidr[rlb - 1] == ipdb - 1
            assert rp.pdbidr[rlb] == ipdb
            assert rp.pdbidr[rub - 1] == ipdb
            if ipdb + 1 < len(rp.pdb):
                assert rp.pdbidr[rub] == ipdb + 1
            resi = rp.resno[rp.pdbidr == ipdb]
            assert np.min(resi) == 0
            assert np.max(resi) == rp.nres[ipdb] - 1
            plb, pub = rp.pdb_pair_offsets.values[ipdb : ipdb + 2]
            resi = rp.resi[plb:pub]
            resj = rp.resj[plb:pub]
            assert np.min(resi) == 0
            assert np.max(resi) < rp.nres[ipdb]
            assert 0 < np.min(resj) < rp.nres[ipdb]

        # sanity check pair distances vs res-res distances
        pair_resi_idx = rp.pdb_res_offsets[rp.pdbidp] + rp.resi
        pair_resj_idx = rp.pdb_res_offsets[rp.pdbidp] + rp.resj
        cbi = rp.cb[pair_resi_idx]
        cbj = rp.cb[pair_resj_idx]
        dhat = np.linalg.norm(cbi - cbj, axis=1)
        assert np.allclose(dhat, rp.dist, atol=1e-3)


def _get_pdb_names(files):
    base = [os.path.basename(f) for f in files]
    assert all(b[4:] == "_0001.pdb" for b in base)
    return [b[:4] for b in base]
