import os

import xarray as xr
import numpy as np


class ResPairData:
    def __init__(
        self,
        pdbdata,
        coords,
        resdata,
        pairdata,
        pdb_res_offsets,
        pdb_pair_offsets,
        bin_params,
    ):

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
        pairdata["ppdbidp"] = pairdata["pdbno"]
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
        return getattr(self.data, k)

    def __str__(self):
        return "ResPairData with data = " + str(self.data).replace("\n", "\n  ")


def _get_pdb_names(files):
    base = [os.path.basename(f) for f in files]
    assert all(b[4:] == "_0001.pdb" for b in base)
    return [b[:4] for b in base]
