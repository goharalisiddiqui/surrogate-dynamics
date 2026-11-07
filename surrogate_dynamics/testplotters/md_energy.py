from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import openmm
import openmm.app as app
import openmm.unit as unit
from openmm.app.forcefield import _createResidueSignature
from openmm.app.internal import compiled

from surrogate_dynamics.testplotters.simple import TimeseriesSimplePlotter

_FORCEFIELD_ALIASES: Dict[str, str] = {
    "amber03": "amber03.xml",
    "amber99sb": "amber99sb.xml",
    "amber99sbildn": "amber99sbildn.xml",
    "amber14": "amber14-all.xml",
    "amber14-all": "amber14-all.xml",
    "charmm": "charmm36.xml",
    "charmm36": "charmm36.xml",
    "amoeba2018": "amoeba2018.xml",
}


class InternalEnergyPlotter(TimeseriesSimplePlotter):
    """
    Test plotter that computes molecular internal potential energies (harmonic bond,
    harmonic angle, and periodic torsion energies) from decoded internal coordinate
    predictions and plots the energy trajectory and distribution.

    Compatibility Requirements
    --------------------------
    1. **DataModule**: Only compatible with
       :class:`~surrogate_dynamics.datamodules.atomistic_trajectory.AtomisticTrajectoryDataModule`
       (``_COMPATIBLE_DATAMODULES = ["ATOMISTIC_TRAJECTORY"]``).
    2. **Embedding Model**: Only compatible with
       :class:`~collective_encoder.models.neural_nets.bge.BondGraphEncoderDecoder`
       (``_COMPATIBLE_EMBEDDINGS = ["BGE", "BGE_V2"]``), which provides the molecular
       graph topology (``bond_index``, ``angle_index``, ``torsion_index``) and decodes
       predictions into ``{'bond_dist', 'angle', 'dihedral_cos', 'dihedral_sin'}``.

    Force-Field Parameterization
    ----------------------------
    Uses :class:`openmm.app.ForceField` at initialization to parameterize the
    molecular topology from ``datamodule.datareader`` using the configured force
    field (``"amber03"`` / ``"amber03.xml"`` by default, or user overrides such as
    ``"charmm36"``, ``"amber99sb"``, ``"amber14"``) and extracts exact harmonic bond
    $(r_0, k_b)$, harmonic angle $(\theta_0, k_\theta)$, and periodic torsion
    $(n, \delta, k_\phi)$ parameters for each internal coordinate in the decoder.

    Parameters
    ----------
    args : dict, optional
        Configuration dictionary supporting all :class:`TimeseriesSimplePlotter`
        keys plus:
        - ``datamodule`` (AtomisticTrajectoryDataModule): Active atomistic trajectory
          datamodule (injected automatically during test evaluation).
        - ``forcefield`` (str or list of str, default "amber03"): Force field name
          or OpenMM XML filename(s), e.g., ``"amber03"``, ``"charmm36"``,
          ``"amber99sb"``, ``"amber14"``.
        - ``include_dihedrals`` (bool, default False): Whether to include periodic
          torsion energies in the total internal energy sum.
        - ``distance_unit`` (str, default "angstrom"): Unit of decoded ``bond_dist``
          values (``"angstrom"`` or ``"nm"``). Converted internally to nanometers to
          match OpenMM force constants in $\\mathrm{kJ/(mol \\cdot nm^2)}$.
    kwargs : dict
        Additional keyword arguments forwarded to :class:`TimeseriesSimplePlotter`.
    """

    _IDENTIFIER = "INTERNAL_ENERGY"
    _COMPATIBLE_DATAMODULES: List[str] = ["ATOMISTIC_TRAJECTORY"]
    _COMPATIBLE_EMBEDDINGS: List[str] = ["BGE", "BGE_V2"]
    _REQUIRED_ARGS = TimeseriesSimplePlotter._REQUIRED_ARGS + ["datamodule"]
    _OPTIONAL_ARGS = TimeseriesSimplePlotter._OPTIONAL_ARGS.copy()
    _OPTIONAL_ARGS.update({
        "forcefield": "amber03",
        "include_dihedrals": False,
        "distance_unit": "angstrom",
    })

    def __init__(self, args: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """
        Initialize ``InternalEnergyPlotter``, validate datamodule and embedding
        compatibility, and extract force-field parameters via OpenMM.

        Parameters
        ----------
        args : dict, optional
            Plotter configuration dictionary.
        **kwargs
            Runtime keyword arguments forwarded to :class:`TimeseriesSimplePlotter`.
        """
        super().__init__(args=args, **kwargs)
        self._extract_ff_parameters()

    def check_datamodule_compatibility(self, datamodule: Any) -> None:
        """
        Verify that ``datamodule`` is an ``AtomisticTrajectoryDataModule``.

        Parameters
        ----------
        datamodule : Any
            DataModule instance or identifier string to validate.

        Raises
        ------
        ValueError
            If ``datamodule`` is missing or not in ``_COMPATIBLE_DATAMODULES``.
        """
        compatible_dmods = self.get_compatible_datamodules()
        if datamodule is None:
            self.raise_error(
                f"{type(self).__name__} requires a compatible datamodule "
                f"{compatible_dmods}, but 'datamodule' was None. Calculating "
                "internal energy for non-atomistic trajectory data is not supported."
            )
        dm_id = (
            datamodule
            if isinstance(datamodule, str)
            else getattr(datamodule, "_IDENTIFIER", None)
        )
        dm_cls_name = (
            datamodule if isinstance(datamodule, str) else type(datamodule).__name__
        )
        if dm_id not in compatible_dmods and dm_cls_name not in compatible_dmods:
            self.raise_error(
                f"Datamodule type '{dm_id or dm_cls_name}' is not compatible with "
                f"{type(self).__name__}, which only supports {compatible_dmods}. "
                "Calculating internal energy for any other type of predicted data is "
                "non-sensical."
            )

    def check_embedding_compatibility(self, embedding_model: Any) -> None:
        """
        Verify that ``embedding_model`` is a ``BondGraphEncoderDecoder`` exposing
        molecular topology indices on ``decoder_net``.

        Parameters
        ----------
        embedding_model : Any
            Embedding model instance to validate.

        Raises
        ------
        ValueError
            If ``embedding_model`` is missing, not in ``_COMPATIBLE_EMBEDDINGS``, or
            does not expose ``decoder_net`` with ``bond_index``, ``angle_index``,
            and ``torsion_index``.
        """
        compatible_embs = self.get_compatible_embeddings()
        if embedding_model is None:
            self.raise_error(
                f"{type(self).__name__} requires a compatible embedding model "
                f"{compatible_embs}, but 'embedding_model' was None."
            )
        emb_id = (
            embedding_model
            if isinstance(embedding_model, str)
            else getattr(embedding_model, "_IDENTIFIER", None)
        )
        emb_cls_name = (
            embedding_model
            if isinstance(embedding_model, str)
            else type(embedding_model).__name__
        )
        if (
            emb_id not in compatible_embs
            and emb_cls_name not in compatible_embs
            and emb_cls_name != "BondGraphEncoderDecoder"
        ):
            self.raise_error(
                f"Embedding model '{emb_id or emb_cls_name}' is not compatible with "
                f"{type(self).__name__}, which only supports {compatible_embs} "
                "(BondGraphEncoderDecoder) since no other embedding returns the "
                "molecular topology required for energy calculations."
            )

        decoder_net = getattr(embedding_model, "decoder_net", None)
        for req_attr in ("bond_index", "angle_index", "torsion_index"):
            if decoder_net is None or not hasattr(decoder_net, req_attr):
                self.raise_error(
                    f"Embedding model '{emb_id or emb_cls_name}' is missing "
                    f"'decoder_net.{req_attr}' required by {type(self).__name__}."
                )

    @staticmethod
    def _resolve_forcefield_files(
        forcefield: Union[str, List[str]]
    ) -> List[str]:
        """
        Convert a force-field name or list of names into OpenMM XML filenames.

        Parameters
        ----------
        forcefield : str or list of str
            Force-field identifier (e.g., ``"amber03"``, ``"charmm36"``) or XML file
            path(s).

        Returns
        -------
        list of str
            Resolved OpenMM XML force-field filenames.
        """
        names = [forcefield] if isinstance(forcefield, str) else list(forcefield)
        resolved: List[str] = []
        for name in names:
            key = name.strip().lower()
            if key in _FORCEFIELD_ALIASES:
                resolved.append(_FORCEFIELD_ALIASES[key])
            elif name.endswith(".xml"):
                resolved.append(name)
            else:
                resolved.append(f"{key}.xml")
        return resolved

    @staticmethod
    def _disambiguate_template_matcher(
        ff_obj: app.ForceField,
        res: app.Topology,
        bonded_to_atom: Any,
        ignore_external_bonds: bool,
        ignore_extra_particles: bool,
    ) -> Any:
        """
        Template matcher registered on :class:`openmm.app.ForceField` to disambiguate
        merged residues when multiple templates share the same graph connectivity
        (such as ``ALAD`` and ``AANM`` in ``charmm36.xml``).

        Parameters
        ----------
        ff_obj : openmm.app.ForceField
            Active OpenMM ForceField object.
        res : openmm.app.Residue
            Residue being matched.
        bonded_to_atom : list
            Adjacency list of bonded atoms.
        ignore_external_bonds : bool
            Whether external bonds are ignored.
        ignore_extra_particles : bool
            Whether extra particles are ignored.

        Returns
        -------
        Any or None
            Matched force-field template or ``None``.
        """
        sig = _createResidueSignature([a.element for a in res.atoms()])
        if sig in ff_obj._templateSignatures:
            matches = [
                t
                for t in ff_obj._templateSignatures[sig]
                if compiled.matchResidueToTemplate(
                    res,
                    t,
                    bonded_to_atom,
                    ignore_external_bonds,
                    ignore_extra_particles,
                )
                is not None
            ]
            if matches:
                for t in matches:
                    if t.name == res.name:
                        return t
                return matches[0]
        return None

    def _build_openmm_topology(self) -> Tuple[app.Topology, List[int]]:
        """
        Construct an :class:`openmm.app.Topology` for the solute molecule from
        ``self.datamodule.datareader`` and return the mapping from selected atom
        indices ``0..N_sel-1`` to OpenMM particle indices.

        Returns
        -------
        tuple of (openmm.app.Topology, list of int)
            - ``topology``: OpenMM Topology containing the full solute residues.
            - ``sel_to_omm``: Mapping from selected atom index to OpenMM atom index.
        """
        datareader = getattr(self.datamodule, "datareader", None)
        if datareader is None:
            self.raise_error(
                f"{type(self).__name__} requires 'datamodule.datareader' to access "
                "the molecular topology."
            )

        u = getattr(datareader, "u", None)
        mol = getattr(datareader, "mol", None)
        if u is None or mol is None:
            import MDAnalysis as mda

            top_file = getattr(datareader, "topology_file", None)
            selection = getattr(datareader, "selection", "all")
            if top_file is None:
                self.raise_error(
                    "Could not find 'u'/'mol' or 'topology_file' on datamodule.datareader."
                )
            u = mda.Universe(top_file)
            mol = u.select_atoms(selection)

        solute = mol.residues.atoms
        top = app.Topology()
        chain = top.addChain()
        res_map: Dict[int, Any] = {}
        atom_map: Dict[int, Any] = {}

        for res in mol.residues:
            omm_res = top.addResidue(res.resname, chain, id=str(res.resid))
            res_map[int(res.resid)] = omm_res

        for atom in solute:
            el_sym = getattr(atom, "element", "") or ""
            if not el_sym:
                alpha_chars = "".join([c for c in atom.name if c.isalpha()])
                el_sym = alpha_chars[0].upper() if alpha_chars else "C"
            elem = app.Element.getBySymbol(el_sym)
            omm_atom = top.addAtom(atom.name, elem, res_map[int(atom.resid)])
            atom_map[int(atom.index)] = omm_atom

        solute_indices = {int(idx) for idx in solute.indices}
        if hasattr(u, "bonds") and len(u.bonds) > 0:
            for bond in u.bonds:
                i, j = int(bond.indices[0]), int(bond.indices[1])
                if i in solute_indices and j in solute_indices:
                    top.addBond(atom_map[i], atom_map[j])
        else:
            decoder_bonds = np.asarray(self.embedding_model.decoder_net.bond_index)
            for bi, bj in decoder_bonds:
                ui, uj = int(mol.indices[int(bi)]), int(mol.indices[int(bj)])
                top.addBond(atom_map[ui], atom_map[uj])

        sel_to_omm = [atom_map[int(idx)].index for idx in mol.indices]
        return top, sel_to_omm

    def _extract_ff_parameters(self) -> None:
        """
        Parameterize the molecular topology with :class:`openmm.app.ForceField` and
        populate ``self.bond_r0``, ``self.bond_k``, ``self.angle_theta0``,
        ``self.angle_k``, and ``self.torsion_terms`` aligned with
        ``self.embedding_model.decoder_net``.
        """
        ff_files = self._resolve_forcefield_files(self.forcefield)
        self.log_info(
            f"Parameterizing molecular topology using OpenMM ForceField {ff_files}..."
        )
        try:
            ff = app.ForceField(*ff_files)
        except Exception as e:
            self.raise_error(
                f"Failed to load force field '{self.forcefield}' ({ff_files}) in "
                f"OpenMM: {e}"
            )
        ff.registerTemplateMatcher(self._disambiguate_template_matcher)

        topology, sel_to_omm = self._build_openmm_topology()
        try:
            system = ff.createSystem(topology, constraints=None)
        except Exception as e:
            self.raise_error(
                f"OpenMM ForceField {ff_files} failed to parameterize the molecular "
                f"topology: {e}"
            )

        bond_lookup: Dict[Tuple[int, int], Tuple[float, float]] = {}
        angle_lookup: Dict[Tuple[int, int, int], Tuple[float, float]] = {}
        torsion_lookup: Dict[Tuple[int, int, int, int], List[Tuple[int, float, float]]] = {}

        for force in system.getForces():
            if isinstance(force, openmm.HarmonicBondForce):
                for idx in range(force.getNumBonds()):
                    p1, p2, r0, k = force.getBondParameters(idx)
                    r0_nm = float(r0.value_in_unit(unit.nanometer))
                    k_val = float(
                        k.value_in_unit(
                            unit.kilojoule_per_mole / (unit.nanometer**2)
                        )
                    )
                    bond_lookup[(min(p1, p2), max(p1, p2))] = (r0_nm, k_val)
            elif isinstance(force, openmm.HarmonicAngleForce):
                for idx in range(force.getNumAngles()):
                    p1, p2, p3, theta0, k = force.getAngleParameters(idx)
                    theta0_rad = float(theta0.value_in_unit(unit.radian))
                    k_val = float(
                        k.value_in_unit(unit.kilojoule_per_mole / (unit.radian**2))
                    )
                    angle_lookup[(min(p1, p3), p2, max(p1, p3))] = (
                        theta0_rad,
                        k_val,
                    )
            elif isinstance(force, openmm.PeriodicTorsionForce):
                for idx in range(force.getNumTorsions()):
                    p1, p2, p3, p4, per, phase, k = force.getTorsionParameters(idx)
                    phase_rad = float(phase.value_in_unit(unit.radian))
                    k_val = float(k.value_in_unit(unit.kilojoule_per_mole))
                    key_fwd = (p1, p2, p3, p4)
                    key_rev = (p4, p3, p2, p1)
                    torsion_lookup.setdefault(key_fwd, []).append(
                        (int(per), phase_rad, k_val)
                    )
                    if key_rev != key_fwd:
                        torsion_lookup.setdefault(key_rev, []).append(
                            (int(per), phase_rad, k_val)
                        )

        decoder_net = self.embedding_model.decoder_net
        bond_index = np.asarray(decoder_net.bond_index, dtype=int)
        angle_index = np.asarray(decoder_net.angle_index, dtype=int)
        torsion_index = np.asarray(decoder_net.torsion_index, dtype=int)

        bond_r0_list: List[float] = []
        bond_k_list: List[float] = []
        for i, j in bond_index:
            oi, oj = sel_to_omm[int(i)], sel_to_omm[int(j)]
            key = (min(oi, oj), max(oi, oj))
            if key not in bond_lookup:
                self.raise_error(
                    f"Bond ({i}, {j}) [OpenMM atoms {key}] not found in "
                    f"HarmonicBondForce for force field {ff_files}."
                )
            r0_nm, k_val = bond_lookup[key]
            bond_r0_list.append(r0_nm)
            bond_k_list.append(k_val)

        angle_theta0_list: List[float] = []
        angle_k_list: List[float] = []
        for i, j, k in angle_index:
            oi, oj, ok = sel_to_omm[int(i)], sel_to_omm[int(j)], sel_to_omm[int(k)]
            key = (min(oi, ok), oj, max(oi, ok))
            if key not in angle_lookup:
                self.raise_error(
                    f"Angle ({i}, {j}, {k}) [OpenMM atoms {key}] not found in "
                    f"HarmonicAngleForce for force field {ff_files}."
                )
            theta0_rad, k_val = angle_lookup[key]
            angle_theta0_list.append(theta0_rad)
            angle_k_list.append(k_val)

        torsion_terms_list: List[List[Tuple[int, float, float]]] = []
        for i, j, k, l in torsion_index:
            oi, oj, ok, ol = (
                sel_to_omm[int(i)],
                sel_to_omm[int(j)],
                sel_to_omm[int(k)],
                sel_to_omm[int(l)],
            )
            terms = torsion_lookup.get((oi, oj, ok, ol), [])
            torsion_terms_list.append(terms)

        self.bond_r0 = np.array(bond_r0_list, dtype=np.float64)
        self.bond_k = np.array(bond_k_list, dtype=np.float64)
        self.angle_theta0 = np.array(angle_theta0_list, dtype=np.float64)
        self.angle_k = np.array(angle_k_list, dtype=np.float64)
        self.torsion_terms = torsion_terms_list

    def calculate_bond_energies(self, decoded: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute per-frame harmonic bond potential energy
        $E_{\\mathrm{bond}} = \\sum_b \\frac{1}{2} k_b (r_b - r_{0,b})^2$.

        Parameters
        ----------
        decoded : dict of str to numpy.ndarray
            Decoded internal coordinates containing ``'bond_dist'`` of shape
            ``(n_frames, n_bonds)``.

        Returns
        -------
        numpy.ndarray
            1-D array of bond energies in kJ/mol of shape ``(n_frames,)``.
        """
        bond_distances = np.asarray(decoded["bond_dist"], dtype=np.float64)
        if str(self.distance_unit).lower() in ("angstrom", "angstroms", "a"):
            bond_distances = bond_distances * 0.1
        diff = bond_distances - self.bond_r0[np.newaxis, :]
        energies = 0.5 * self.bond_k[np.newaxis, :] * (diff**2)
        return np.sum(energies, axis=1)

    def calculate_angle_energies(self, decoded: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute per-frame harmonic angle potential energy
        $E_{\\mathrm{angle}} = \\sum_a \\frac{1}{2} k_{\\theta,a} (\\theta_a - \\theta_{0,a})^2$.

        Parameters
        ----------
        decoded : dict of str to numpy.ndarray
            Decoded internal coordinates containing ``'angle'`` of shape
            ``(n_frames, n_angles)``.

        Returns
        -------
        numpy.ndarray
            1-D array of angle energies in kJ/mol of shape ``(n_frames,)``.
        """
        angle_values = np.asarray(decoded["angle"], dtype=np.float64)
        diff = angle_values - self.angle_theta0[np.newaxis, :]
        energies = 0.5 * self.angle_k[np.newaxis, :] * (diff**2)
        return np.sum(energies, axis=1)

    def calculate_dihedral_energies(
        self, decoded: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Compute per-frame periodic torsion potential energy
        $E_{\\mathrm{torsion}} = \\sum_d \\sum_m k_{\\phi,d,m} (1 + \\cos(n_{d,m} \\phi_d - \\delta_{d,m}))$.

        Parameters
        ----------
        decoded : dict of str to numpy.ndarray
            Decoded internal coordinates containing ``'dihedral_cos'`` and
            ``'dihedral_sin'`` of shape ``(n_frames, n_torsions)``.

        Returns
        -------
        numpy.ndarray
            1-D array of torsion energies in kJ/mol of shape ``(n_frames,)``.
        """
        dihedral_cos = np.asarray(decoded["dihedral_cos"], dtype=np.float64)
        dihedral_sin = np.asarray(decoded["dihedral_sin"], dtype=np.float64)
        dihedral_angles = np.arctan2(dihedral_sin, dihedral_cos)
        n_frames = dihedral_angles.shape[0]
        total_torsion = np.zeros(n_frames, dtype=np.float64)

        for d_idx, terms in enumerate(self.torsion_terms):
            phi = dihedral_angles[:, d_idx]
            for periodicity, phase_rad, k_val in terms:
                total_torsion += k_val * (
                    1.0 + np.cos(periodicity * phi - phase_rad)
                )
        return total_torsion

    def calculate_total_energies(
        self, decoded: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute bond, angle, dihedral, and total internal energies for all frames.

        Parameters
        ----------
        decoded : dict of str to numpy.ndarray
            Decoded internal coordinates dictionary.

        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)
            ``(total_energies, bond_energies, angle_energies, dihedral_energies)``.
        """
        bond_energies = self.calculate_bond_energies(decoded)
        angle_energies = self.calculate_angle_energies(decoded)
        dihedral_energies = self.calculate_dihedral_energies(decoded)
        total_energies = bond_energies + angle_energies
        if self.include_dihedrals:
            total_energies = total_energies + dihedral_energies
        return total_energies, bond_energies, angle_energies, dihedral_energies

    def _plot_energy_trajectories(
        self,
        total_energies: np.ndarray,
        bond_energies: np.ndarray,
        angle_energies: np.ndarray,
        dihedral_energies: np.ndarray,
    ) -> None:
        """
        Plot total internal energy over time, energy component breakdown, and
        the energy distribution histogram.

        Parameters
        ----------
        total_energies : numpy.ndarray
            1-D array of total internal energies.
        bond_energies : numpy.ndarray
            1-D array of harmonic bond energies.
        angle_energies : numpy.ndarray
            1-D array of harmonic angle energies.
        dihedral_energies : numpy.ndarray
            1-D array of periodic torsion energies.
        """
        steps = np.arange(len(total_energies))

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(steps, total_energies, label="Total Internal Energy")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Total Energy (kJ/mol)")
        ax.set_title(f"Total Internal Energy over Time ({self.forcefield})")
        ax.legend()
        plt.tight_layout()
        self.log_image(fig, "predicted_energies")
        plt.close(fig)

        fig_comp, ax_comp = plt.subplots(figsize=(8, 5))
        ax_comp.plot(steps, bond_energies, label="Bond Energy", alpha=0.85)
        ax_comp.plot(steps, angle_energies, label="Angle Energy", alpha=0.85)
        if self.include_dihedrals:
            ax_comp.plot(
                steps, dihedral_energies, label="Dihedral Energy", alpha=0.85
            )
        ax_comp.set_xlabel("Time Step")
        ax_comp.set_ylabel("Energy (kJ/mol)")
        ax_comp.set_title(f"Internal Energy Components ({self.forcefield})")
        ax_comp.legend()
        plt.tight_layout()
        self.log_image(fig_comp, "energy_components")
        plt.close(fig_comp)

        finite_vals = total_energies[np.isfinite(total_energies)]
        if len(finite_vals) > 0:
            fig_hist, ax_hist = plt.subplots(figsize=(7, 5))
            ax_hist.hist(
                finite_vals, bins=50, color="steelblue", edgecolor="black", alpha=0.8
            )
            ax_hist.set_xlabel("Total Internal Energy (kJ/mol)")
            ax_hist.set_ylabel("Count")
            ax_hist.set_title(f"Internal Energy Distribution ({self.forcefield})")
            plt.tight_layout()
            self.log_image(fig_hist, "energy_distribution")
            plt.close(fig_hist)

    def plot(
        self,
        data: Any,
        latent: Any,
        pred: Any,
        labels: Any,
        meta: Any,
    ) -> None:
        """
        Decode accumulated predictions into internal coordinates, evaluate molecular
        internal energies, save energy arrays and scalar metrics, and generate plots.

        Parameters
        ----------
        data : Any
            Accumulated inputs.
        latent : Any
            Accumulated latent representations.
        pred : Any
            Accumulated latent predictions.
        labels : Any
            Accumulated ground-truth labels.
        meta : Any
            Accumulated metadata.
        """
        decoded_pred = self._decode_predictions(pred)
        if not isinstance(decoded_pred, dict):
            self.raise_error(
                f"{type(self).__name__} expected decoded predictions to be a "
                "dictionary of internal coordinates ('bond_dist', 'angle', "
                f"'dihedral_cos', 'dihedral_sin'), got {type(decoded_pred)}."
            )

        total_e, bond_e, angle_e, dihedral_e = self.calculate_total_energies(
            decoded_pred
        )

        self._save_data(total_e, "total_energies")
        self._save_data(bond_e, "bond_energies")
        self._save_data(angle_e, "angle_energies")
        self._save_data(dihedral_e, "dihedral_energies")

        self.set_metric("mean_total_energy", float(np.mean(total_e)))
        self.set_metric("std_total_energy", float(np.std(total_e)))
        self.set_metric("median_total_energy", float(np.median(total_e)))
        self.set_metric("mean_bond_energy", float(np.mean(bond_e)))
        self.set_metric("mean_angle_energy", float(np.mean(angle_e)))
        self.set_metric("mean_dihedral_energy", float(np.mean(dihedral_e)))

        self._plot_energy_trajectories(total_e, bond_e, angle_e, dihedral_e)
