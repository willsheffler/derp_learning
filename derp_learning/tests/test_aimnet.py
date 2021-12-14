from derp_learning.aimnet import load_AIMNetMT_ens, load_AIMNetSMD_ens, AIMNetCalculator
from derp_learning.aimnet import models
from derp_learning import tests

import os
import rdkit.Chem
import ase
import ase.optimize
import ase.md
import numpy as np
import torch
import random
from openbabel import pybel

def pybel2ase(mol):
   coord = np.asarray([a.coords for a in mol.atoms])
   numbers = np.asarray([a.atomicnum for a in mol.atoms])
   return ase.Atoms(positions=coord, numbers=numbers)

def test_aimnet_charges():
   niter = 10
   charges = np.zeros((niter, 20))
   volumes = np.zeros((niter, 20))
   smiles = 'O=C([O-])[C@@H]([NH3+])Cc1c[nH]cn1'  # Histidine
   mol = pybel.readstring('smi', smiles)
   mol.make3D()
   atoms = pybel2ase(mol)

   model = load_AIMNetMT_ens().cuda()
   # model = load_AIMNetSMD_ens().cuda()
   calc = AIMNetCalculator(model)
   # calc = AIMNetCalculator(model)
   # tests.save_test_data('test_aimnet_charges_mol', atoms)
   atoms = tests.load_test_data('test_aimnet_charges_mol')
   atoms.set_calculator(calc)
   opt = ase.optimize.BFGS(atoms, trajectory='gas_opt.traj')
   opt.run(0.1)

   charge = calc.results['elmoments'][0, :, 0]
   vol = calc.results['volume'][0]

   refcharge = [
      -0.62481374, 0.7630918, -0.5691881, -0.02539759, -0.9038821, -0.35443392, 0.26744002,
      -0.24668668, -0.38005802, 0.24390018, -0.5313212, 0.13167161, 0.3751933, 0.47297472,
      0.38505638, 0.16893908, 0.16073106, 0.16768984, 0.36870983, 0.11017318
   ]
   refvol = [
      25.55421, 21.514797, 25.581034, 32.403996, 37.66971, 36.973072, 27.242733, 36.787865,
      28.910318, 28.584164, 33.01521, 2.6676216, 1.4759812, 1.2925818, 1.3805472, 2.546827,
      2.6276596, 2.582686, 1.460769, 2.869126
   ]
   assert np.allclose(charge, refcharge)
   assert np.allclose(vol, refvol)

   # atoms = pybel2ase(mol)
   # from last cell, after charg comp
   # atoms.set_calculator(calc)
   # opt = ase.optimize.BFGS(atoms, trajectory='smd_opt.traj')
   # opt.run(0.02)
   # traj = ase.io.Trajectory('smd_opt.traj', 'r')
   # nglview.show_asetraj(traj)

if __name__ == '__main__':
   test_aimnet_charges()
