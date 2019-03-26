import os
import _pickle
import random
import numpy as np
import xarray as xr

from derp_learning.data_types import ResPairData


def load_which_takes_forever():
    with open("pdb_res_pair_data.pickle", "rb") as inp:
        return _pickle.load(inp)


if hasattr(os, "a_very_unique_name"):
    rp = os.a_very_unique_name
else:
    rp = load_which_takes_forever()
    os.a_very_unique_name = rp

if "ppdbidp" in rp.data:
    print("renaming ppdbidp")
    rp.data.rename(inplace=True, ppdbidp="pdbidp")


def main():
    subidx = sorted(np.random.choice(len(rp.pdb), 10))
    rpsub = rp.subset(subidx)
    rpsub.sanity_check()
    print(rpsub)


if __name__ == "__main__":
    main()
