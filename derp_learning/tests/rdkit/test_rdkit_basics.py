import pytest
import rdkit.Chem.Draw
import rdkit.Chem.AllChem

rdkit_atom_mnames = [
   'GetSymbol',
   'GetIdx',
   'GetAtomicNum',
   'GetAtomMapNum',
   'GetChiralTag',
   'GetDegree',
   'GetExplicitValence',
   'GetFormalCharge',
   'GetHybridization',
   'GetImplicitValence',
   'GetIsAromatic',
   'GetMass',
   'GetNumExplicitHs',
   'GetNumImplicitHs',
   'GetNumRadicalElectrons',
   'GetTotalDegree',
   'GetTotalNumHs',
   'GetTotalValence',
   'IsInRing',
   # IsInRingSize(3)
   # IsInRingSize(4)
   # IsInRingSize(5)
]
rdkit_atom_mnames_hinvariant = rdkit_atom_mnames.copy()
rdkit_atom_mnames_hinvariant.remove('GetDegree')
rdkit_atom_mnames_hinvariant.remove('GetExplicitValence')
rdkit_atom_mnames_hinvariant.remove('GetImplicitValence')
rdkit_atom_mnames_hinvariant.remove('GetNumImplicitHs')
rdkit_atom_mnames_hinvariant.remove('GetTotalNumHs')

rdkit_bond_mnames = [
   # 'GetBeginAtom',
   'GetBeginAtomIdx',
   'GetBondDir',
   'GetBondType',
   'GetBondTypeAsDouble',
   # 'GetEndAtom',
   'GetEndAtomIdx',
   'GetIdx',
   'GetIsAromatic',
   'GetIsConjugated',
   # 'GetOtherAtom',
   # 'GetOtherAtomIdx',
   # 'GetOwningMol',
   # 'GetQuery',
   'GetStereo',
   # 'GetStereoAtoms',
   # 'GetStereoAtoms',
   # 'GetValenceContrib',
   'HasOwningMol',
   'HasQuery',
]
rdkit_bond_mnames_hinvariant = rdkit_bond_mnames.copy()
rdkit_bond_mnames_hinvariant.remove('GetBeginAtomIdx')
rdkit_bond_mnames_hinvariant.remove('GetEndAtomIdx')
rdkit_bond_mnames_hinvariant.remove('GetIdx')

testmolpdb = '''
HETATM 2639 ZN    ZN A   1     -22.708  36.013  70.692  1.00 30.07          ZN  
HETATM 2640 MG    MG A   2     -23.387  33.184  73.010  1.00 24.43          MG  
HETATM 2641  C1  VDN A 201     -17.311  28.769  59.425  1.00 41.57           C  
HETATM 2642  C2  VDN A 201     -18.424  27.750  59.552  1.00 40.75           C  
HETATM 2643  O3  VDN A 201     -19.499  28.317  60.289  1.00 39.69           O  
HETATM 2644  C4  VDN A 201     -20.679  27.647  60.532  1.00 39.78           C  
HETATM 2645  C5  VDN A 201     -20.872  26.322  60.132  1.00 41.92           C  
HETATM 2646  C6  VDN A 201     -22.088  25.700  60.407  1.00 43.01           C  
HETATM 2647  C7  VDN A 201     -23.112  26.387  61.075  1.00 45.33           C  
HETATM 2648  C8  VDN A 201     -22.903  27.707  61.474  1.00 40.50           C  
HETATM 2649  C9  VDN A 201     -21.710  28.336  61.166  1.00 37.34           C  
HETATM 2650  S10 VDN A 201     -24.513  25.683  61.421  1.00 49.64           S  
HETATM 2651  O11 VDN A 201     -25.628  26.501  61.051  1.00 51.30           O  
HETATM 2652  O12 VDN A 201     -24.559  24.448  60.700  1.00 51.54           O  
HETATM 2653  C13 VDN A 201     -23.926  24.969  67.853  1.00 60.07           C  
HETATM 2654  N14 VDN A 201     -24.652  25.389  62.919  1.00 53.30           N  
HETATM 2655  C15 VDN A 201     -23.589  24.530  63.465  1.00 55.49           C  
HETATM 2656  C16 VDN A 201     -23.701  24.352  64.968  1.00 56.95           C  
HETATM 2657  N17 VDN A 201     -23.954  25.673  65.507  1.00 57.84           N  
HETATM 2658  C18 VDN A 201     -25.303  26.139  65.236  1.00 56.81           C  
HETATM 2659  C19 VDN A 201     -25.477  26.304  63.720  1.00 55.55           C  
HETATM 2660  C20 VDN A 201     -23.380  25.948  66.820  1.00 59.33           C  
HETATM 2661  C21 VDN A 201     -21.389  29.716  61.614  1.00 34.93           C  
HETATM 2662  N22 VDN A 201     -21.038  30.726  60.643  1.00 34.65           N  
HETATM 2663  C23 VDN A 201     -20.995  32.029  60.921  1.00 34.40           C  
HETATM 2664  C24 VDN A 201     -21.536  32.397  62.243  1.00 32.84           C  
HETATM 2665  N25 VDN A 201     -21.991  31.533  63.124  1.00 34.15           N  
HETATM 2666  N26 VDN A 201     -22.038  30.181  62.807  1.00 33.69           N  
HETATM 2667  O27 VDN A 201     -20.573  32.827  60.098  1.00 37.12           O  
HETATM 2668  C28 VDN A 201     -21.710  33.636  62.850  1.00 33.10           C  
HETATM 2669  N29 VDN A 201     -22.267  33.389  64.067  1.00 32.90           N  
HETATM 2670  C30 VDN A 201     -22.461  32.057  64.225  1.00 33.58           C  
HETATM 2671  C31 VDN A 201     -21.384  35.007  62.325  1.00 34.00           C  
HETATM 2672  C32 VDN A 201     -23.018  31.324  65.410  1.00 38.61           C  
HETATM 2673  C33 VDN A 201     -21.860  30.626  66.124  1.00 40.80           C  
HETATM 2674  C34 VDN A 201     -22.410  29.892  67.324  1.00 44.40           C  
   '''

def test_read_from_pdb():
   m = rdkit.Chem.rdmolfiles.MolFromPDBBlock(testmolpdb)
   assert m is not None

@pytest.mark.skip
def test_opt_mmff94():
   m = rdkit.Chem.rdmolfiles.MolFromPDBBlock(testmolpdb)
   m2 = rdkit.Chem.AddHs(m)
   rdkit.Chem.AllChem.EmbedMolecule(m2)
   rdkit.Chem.AllChem.MMFFOptimizeMolecule(m2)
   assert m2 is not None

def test_rdkit_atoms(printme=False):
   m = rdkit.Chem.rdmolfiles.MolFromPDBBlock(testmolpdb)
   assert len(m.GetAtoms()) == 35
   vals, valsh = list(), list()
   for i, atom in enumerate(m.GetAtoms()):
      assert atom.GetIdx() == i
      if printme: print('============', i, '=============')
      for mname in rdkit_atom_mnames:
         mval = getattr(atom, mname)()
         if printme: print(f'    {mname:.<25} {mval}')
      vals.append([getattr(atom, mname)() for mname in rdkit_atom_mnames_hinvariant])

   m = rdkit.Chem.AddHs(m)
   assert len(m.GetAtoms()) == 85
   for i, atom in enumerate(m.GetAtoms()):
      if atom.GetSymbol() != 'H':
         valsh.append([getattr(atom, mname)() for mname in rdkit_atom_mnames_hinvariant])
   assert vals == valsh

def test_rdkit_bonds(printme=False):
   m = rdkit.Chem.rdmolfiles.MolFromPDBBlock(testmolpdb)
   vals, valsh = set(), set()  # order changes in this case
   bonds = m.GetBonds()
   assert len(bonds) == 37
   for i, bond in enumerate(bonds):
      assert bond.GetIdx() == i
      if printme: print('============', i, '=============')
      for mname in rdkit_bond_mnames:
         mval = getattr(bond, mname)()
         if printme: print(f'    {mname:.<25} {mval}')
      vals.add(tuple([getattr(bond, mname)() for mname in rdkit_bond_mnames_hinvariant]))

   m = rdkit.Chem.AddHs(m)
   bonds = m.GetBonds()
   assert len(bonds) == 87
   for i, bond in enumerate(bonds):
      assert bond.GetIdx() == i
      if printme: print('============', i, '=============')
      for mname in rdkit_bond_mnames:
         mval = getattr(bond, mname)()
         if printme: print(f'    {mname:.<25} {mval}')
      if all([
            bond.GetBeginAtom().GetSymbol() != 'H',
            bond.GetBeginAtom().GetSymbol() != 'H',
      ]):
         valsh.add(tuple([getattr(bond, mname)() for mname in rdkit_bond_mnames_hinvariant]))

   assert vals == valsh

# Chem.GetSymmSSSR(m)

if __name__ == '__main__':
   test_read_from_pdb()
   # test_opt_mmff94()
   test_rdkit_atoms(printme=False)
   test_rdkit_bonds(printme=False)
