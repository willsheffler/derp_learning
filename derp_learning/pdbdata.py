import numpy as np
import pyrosetta


score_types = {
    "fa_atr": pyrosetta.rosetta.core.scoring.ScoreType.fa_atr,
    "fa_rep": pyrosetta.rosetta.core.scoring.ScoreType.fa_rep,
    "fa_sol": pyrosetta.rosetta.core.scoring.ScoreType.fa_sol,
    #    # "fa_intra_atr_xover4": pyrosetta.rosetta.core.scoring.ScoreType.fa_intra_atr_xover4,
    #    # "fa_intra_rep_xover4": pyrosetta.rosetta.core.scoring.ScoreType.fa_intra_rep_xover4,
    #    # "fa_intra_sol_xover4": pyrosetta.rosetta.core.scoring.ScoreType.fa_intra_sol_xover4,
    "lk_ball": pyrosetta.rosetta.core.scoring.ScoreType.lk_ball,
    #    # "lk_ball_iso": pyrosetta.rosetta.core.scoring.ScoreType.lk_ball_iso,
    #    # "lk_ball_bridge": pyrosetta.rosetta.core.scoring.ScoreType.lk_ball_bridge,
    #    # "lk_ball_bridge_uncpl": pyrosetta.rosetta.core.scoring.ScoreType.lk_ball_bridge_uncpl,
    "fa_elec": pyrosetta.rosetta.core.scoring.ScoreType.fa_elec,
    #    # "fa_intra_elec": pyrosetta.rosetta.core.scoring.ScoreType.fa_intra_elec,
    #    # "pro_close": pyrosetta.rosetta.core.scoring.ScoreType.pro_close,
    "hbond_sr_bb": pyrosetta.rosetta.core.scoring.ScoreType.hbond_sr_bb,
    "hbond_lr_bb": pyrosetta.rosetta.core.scoring.ScoreType.hbond_lr_bb,
    "hbond_bb_sc": pyrosetta.rosetta.core.scoring.ScoreType.hbond_bb_sc,
    "hbond_sc": pyrosetta.rosetta.core.scoring.ScoreType.hbond_sc,
    #    # "dslf_fa13": pyrosetta.rosetta.core.scoring.ScoreType.dslf_fa13,
    #    # "rama_prepro": pyrosetta.rosetta.core.scoring.ScoreType.rama_prepro,
    #    # "omega": pyrosetta.rosetta.core.scoring.ScoreType.omega,
    #    # "p_aa_pp": pyrosetta.rosetta.core.scoring.ScoreType.p_aa_pp,
    #    # "fa_dun_rot": pyrosetta.rosetta.core.scoring.ScoreType.fa_dun_rot,
    #    # "fa_dun_dev": pyrosetta.rosetta.core.scoring.ScoreType.fa_dun_dev,
    #    # "fa_dun_semi": pyrosetta.rosetta.core.scoring.ScoreType.fa_dun_semi,
    #    # "hxl_tors": pyrosetta.rosetta.core.scoring.ScoreType.hxl_tors,
    #    # "ref": pyrosetta.rosetta.core.scoring.ScoreType.ref,
}


def pdbdata(pose, fname):

    sf = pyrosetta.rosetta.core.scoring.get_score_function()
    sfopt = sf.energy_method_options()
    sfopt.hbond_options().decompose_bb_hb_into_pair_energies(True)
    sf.set_energy_method_options(sfopt)
    sf(pose)

    # one-body stuff

    ncac = get_bb_coords(pose)
    stubs = ncac_to_stubs(ncac)

    cb = get_cb_coords(pose)
    com = np.mean(cb, axis=0)
    rg = np.sqrt(np.sum((cb - com) ** 2) / len(cb))
    coords = dict(ncac=ncac, cb=cb, stubs=stubs, com=com, rg=rg)
    chains = get_chain_bounds(pose)

    seq = pose.sequence()
    ss = pyrosetta.rosetta.core.scoring.dssp.Dssp(pose).get_dssp_secstruct()
    phi = np.array([pose.phi(i) for i in range(1, len(pose) + 1)])
    psi = np.array([pose.psi(i) for i in range(1, len(pose) + 1)])
    omega = np.array([pose.omega(i) for i in range(1, len(pose) + 1)])
    resdata = dict(seq=seq, ss=ss, phi=phi, psi=psi, omega=omega)

    sasa_probe_vals = np.array([2, 3, 4])
    # print(fname, "compute sasa")
    sasa = polya_sasa(pose, sasa_probe_vals)
    assert len(pose) == sasa.shape[0]
    assert sasa.shape[1] == len(sasa_probe_vals)
    for i, v in enumerate(sasa_probe_vals):
        resdata["sasa" + str(v)] = sasa[:, i]

    assert len(pose) == len(ncac)
    assert len(pose) == len(cb)
    assert len(pose) == len(stubs)
    assert len(pose) == len(ss)
    assert len(pose) == len(seq)
    assert len(pose) == len(phi)
    assert len(pose) == len(psi)
    assert len(pose) == len(omega)

    if ncac.shape[-1] is 4:
        ncac = ncac.astype(np.float64)
    elif ncac.shape[-1] is 3:
        tmp = np.ones((ncac.shape[0], 3, 4), dtype=np.float64)
        tmp[..., :3] = ncac
        ncac = tmp
    else:
        assert 0, "bad ncac"

    # two-body stuff
    chainseqs = [seq[lb:ub] for lb, ub in chains]
    sym_chain_follows = [chainseqs.index(x) for x in chainseqs]

    pairdata = extract_pair_terms(**vars())
    # print(fname, "npairterms", pairdata["dist"].shape, len(pose))

    hbonds = extract_hbond_terms(**vars())

    # print("coords", coords.keys())
    # print("resdata", resdata.keys())
    # print("pairdata", pairdata.keys())
    # print("hbonds", hbonds.keys())

    return dict(
        fname=fname,
        coords=coords,
        chains=chains,
        resdata=resdata,
        pairdata=pairdata,
        hbonds=hbonds,
    )


def extract_hbond_terms(pose, fname, **kw):
    hbset = pyrosetta.rosetta.core.scoring.hbonds.HBondSet()
    pyrosetta.rosetta.core.scoring.hbonds.fill_hbond_set(pose, False, hbset)
    result = [set(), set(), set(), set()]
    for ihb in range(hbset.nhbonds()):
        hbond = hbset.hbond(ihb + 1)
        ir = hbond.don_res()
        jr = hbond.acc_res()
        irbb = hbond.don_hatm_is_protein_backbone()
        jrbb = hbond.acc_atm_is_protein_backbone()
        if jr < ir:
            ir, irbb, jr, bb = jr, jrbb, ir, irbb
        idx = 2 * irbb + jrbb
        result[idx].add((ir, jr, hbond.energy()))
        # cute indexing sanity check...
        # print(irbb, jrbb, ["sc_sc", "sc_bb", "bb_sc", "bb_bb"][idx])
    # result = [np.array(list(x)) for x in result]
    labels = ["sc_sc", "sc_bb", "bb_sc", "bb_bb"]
    return {k: v for k, v in zip(labels, result)}


def extract_pair_terms(pose, sym_chain_follows, chains, fname, **kw):
    eweights = pose.energies().weights()
    energy_graph = pose.energies().energy_graph()

    # print(fname, "extract pair energies")
    lbls = ["dist", "etot", "resi", "resj"] + list(score_types.keys())
    pairterms = {k: list() for k in lbls}
    for ichain, chain in enumerate(chains):
        if ichain != sym_chain_follows[ichain]:
            # print(fname, "skip sym redundant chain", ichain, "of", len(chains))
            continue
        for ir in range(*chain):
            assert pose.residue(ir + 1).is_protein()
            for jr in range(ir + 1, len(pose)):
                edge = energy_graph.find_edge(ir + 1, jr + 1)
                if not edge:
                    continue
                etot = edge.dot(eweights)
                if etot == 0.0:
                    continue
                pairterms["resi"].append(ir)
                pairterms["resj"].append(jr)
                pairterms["dist"].append(np.sqrt(edge.square_distance()))
                pairterms["etot"].append(etot)
                for lbl, st in score_types.items():
                    pairterms[lbl].append(edge[st])
    for k in pairterms.keys():
        pairterms[k] = np.array(pairterms[k], np.float32)
    return pairterms


def polya_sasa(pose, sasa_probe_vals):
    M = pyrosetta.rosetta.protocols.simple_moves.MakePolyXMover
    m = M("ALA", keep_pro=False, keep_gly=True, keep_disulfide_cys=True)
    polya_pose = pose.clone()
    m.apply(polya_pose)
    sasacalc = pyrosetta.rosetta.core.scoring.sasa.SasaCalc()

    rsdsasa = np.zeros((len(pose), len(sasa_probe_vals)))
    for i, r in enumerate(sasa_probe_vals):
        sasacalc.set_probe_radius(r)
        sasacalc.calculate(polya_pose)
        rsdsasa[:, i] = np.array(sasacalc.get_residue_sasa())
        # print(
        # i,
        # r,
        # len(rsdsasa),
        # np.min(rsdsasa),
        # np.max(rsdsasa),
        # np.sum(rsdsasa[:, i] == 0),
        # )

    return rsdsasa


def get_bb_stubs(pose, which_resi=None):
    if which_resi is None:
        which_resi = list(range(1, pose.size() + 1))
    npstubs, n_ca_c = [], []
    for ir in which_resi:
        r = pose.residue(ir)
        if not r.is_protein():
            raise ValueError("non-protein residue %s at position %i" % (r.name(), ir))
        n, ca, c = r.xyz("N"), r.xyz("CA"), r.xyz("C")
        ros_stub = ros.core.kinematics.Stub(ca, n, ca, c)
        npstubs.append(numpy_stub_from_rosetta_stub(ros_stub))
        n_ca_c.append(np.array([[n.x, n.y, n.z], [ca.x, ca.y, ca.z], [c.x, c.y, c.z]]))
    return np.stack(npstubs).astype("f8"), np.stack(n_ca_c).astype("f8")


def get_bb_coords(pose, which_resi=None):
    if which_resi is None:
        which_resi = list(range(1, pose.size() + 1))
    n_ca_c = []
    for ir in which_resi:
        r = pose.residue(ir)
        if not r.is_protein():
            raise ValueError("non-protein residue %s at position %i" % (r.name(), ir))
        n, ca, c = r.xyz("N"), r.xyz("CA"), r.xyz("C")
        n_ca_c.append(
            np.array([[n.x, n.y, n.z, 1], [ca.x, ca.y, ca.z, 1], [c.x, c.y, c.z, 1]])
        )
    return np.stack(n_ca_c).astype("f8")


def get_cb_coords(pose, which_resi=None):
    if which_resi is None:
        which_resi = list(range(1, pose.size() + 1))
    cbs = []
    for ir in which_resi:
        r = pose.residue(ir)
        if not r.is_protein():
            raise ValueError("non-protein residue %s at position %i" % (r.name(), ir))
        if r.has("CB"):
            cb = r.xyz("CB")
        else:
            cb = r.xyz("CA")
        cbs.append(np.array([cb.x, cb.y, cb.z, 1]))
    return np.stack(cbs).astype("f8")


def get_chain_bounds(pose):
    ch = np.array([pose.chain(i + 1) for i in range(len(pose))])
    chains = list()
    for i in range(ch[-1]):
        chains.append((np.sum(ch <= i), np.sum(ch <= i + 1)))
    assert chains[0][0] == 0
    assert chains[-1][-1] == len(pose)
    return chains


def ncac_to_stubs(ncac):
    """
        Vector const & center,
        Vector const & a,
        Vector const & b,
        Vector const & c
    )
    {
        Vector e1( a - b);
        e1.normalize();

        Vector e3( cross( e1, c - b ) );
        e3.normalize();

        Vector e2( cross( e3,e1) );
        M.col_x( e1 ).col_y( e2 ).col_z( e3 );
        v = center;
    """
    assert ncac.shape[1:] == (3, 4)
    stubs = np.zeros((len(ncac), 4, 4), dtype=np.float64)
    ca2n = (ncac[:, 0] - ncac[:, 1])[..., :3]
    ca2c = (ncac[:, 2] - ncac[:, 1])[..., :3]
    # tgt1 = ca2n + ca2c  # thought this might make
    # tgt2 = ca2n - ca2c  # n/c coords match better
    tgt1 = ca2n  # rosetta style
    tgt2 = ca2c  # seems better
    a = tgt1
    a /= np.linalg.norm(a, axis=-1)[:, None]
    c = np.cross(a, tgt2)
    c /= np.linalg.norm(c, axis=-1)[:, None]
    b = np.cross(c, a)
    assert np.allclose(np.sum(a * b, axis=-1), 0)
    assert np.allclose(np.sum(b * c, axis=-1), 0)
    assert np.allclose(np.sum(c * a, axis=-1), 0)
    assert np.allclose(np.linalg.norm(a, axis=-1), 1)
    assert np.allclose(np.linalg.norm(b, axis=-1), 1)
    assert np.allclose(np.linalg.norm(c, axis=-1), 1)
    stubs[:, :3, 0] = a
    stubs[:, :3, 1] = b
    stubs[:, :3, 2] = c
    stubs[:, :3, 3] = ncac[:, 1, :3]
    stubs[:, 3, 3] = 1
    return stubs
