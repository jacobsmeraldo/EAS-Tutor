"""Reaction engine for an aromatic synthesis practice playground.

The engine models undergraduate-level aromatic chemistry with RDKit reaction
SMARTS and textbook directing rules. It supports single-step EAS practice,
multi-step synthesis from benzene, functional-group interconversions, route
history, and target challenge generation.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Optional

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.rdChemReactions import ChemicalReaction
except ImportError as exc:  # pragma: no cover - depends on local environment.
    raise ImportError(
        "eas_generator.py requires RDKit. Install it with "
        "`conda install -c conda-forge rdkit` or `pip install rdkit`."
    ) from exc


BENZENE_SMILES = "c1ccccc1"


@dataclass(frozen=True)
class StartingMaterial:
    """Named aromatic starting material for single-step practice."""

    name: str
    smiles: str


@dataclass(frozen=True)
class Reagent:
    """A textbook reagent set represented as one or more RDKit transforms."""

    key: str
    label: str
    short_name: str
    category: str
    reaction_smarts: tuple[str, ...]
    description: str
    is_eas: bool = False
    is_friedel_crafts: bool = False
    exhaustive: bool = False


@dataclass(frozen=True)
class RingSubstituent:
    """A substituent attached to the active benzene ring."""

    attachment_idx: int
    label: str
    director: str
    strength_label: str
    directing_weight: float
    control_priority: float
    fc_blocking: bool
    note: str


@dataclass(frozen=True)
class MoleculeAnalysis:
    """Textbook directing analysis for the current aromatic intermediate."""

    smiles: str
    canonical_smiles: str
    substituents: tuple[RingSubstituent, ...]
    available_ring_positions: tuple[int, ...]
    prevailing_director: str
    ring_activation: str
    friedel_crafts_allowed: bool
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return a JSON/session-state friendly representation."""

        return {
            "smiles": self.smiles,
            "canonical_smiles": self.canonical_smiles,
            "substituents": [asdict(sub) for sub in self.substituents],
            "available_ring_positions": list(self.available_ring_positions),
            "prevailing_director": self.prevailing_director,
            "ring_activation": self.ring_activation,
            "friedel_crafts_allowed": self.friedel_crafts_allowed,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class ProductCandidate:
    """One enumerated product from an EAS reagent."""

    smiles: str
    attack_atom_idx: Optional[int]
    score: float
    orientation_summary: str
    explanation: str


@dataclass(frozen=True)
class ReactionOutcome:
    """Result of applying one reagent to one aromatic intermediate."""

    success: bool
    reagent_key: str
    reagent_label: str
    substrate_smiles: str
    product_smiles: Optional[str]
    message: str
    analysis_before: MoleculeAnalysis
    analysis_after: Optional[MoleculeAnalysis] = None
    candidates: tuple[ProductCandidate, ...] = ()
    selected_candidate: Optional[ProductCandidate] = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON/session-state friendly representation."""

        return {
            "success": self.success,
            "reagent_key": self.reagent_key,
            "reagent_label": self.reagent_label,
            "substrate_smiles": self.substrate_smiles,
            "product_smiles": self.product_smiles,
            "message": self.message,
            "analysis_before": self.analysis_before.to_dict(),
            "analysis_after": (
                self.analysis_after.to_dict() if self.analysis_after else None
            ),
            "candidates": [asdict(candidate) for candidate in self.candidates],
            "selected_candidate": (
                asdict(self.selected_candidate) if self.selected_candidate else None
            ),
        }


@dataclass(frozen=True)
class SynthesisStep:
    """A recorded step in a student's route."""

    step_number: int
    reagent_key: str
    reagent_label: str
    substrate_smiles: str
    product_smiles: str
    message: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON/session-state friendly representation."""

        return asdict(self)


@dataclass(frozen=True)
class SynthesisChallenge:
    """A target molecule challenge generated from a known route."""

    name: str
    target_smiles: str
    starting_smiles: str
    recommended_route: tuple[str, ...]
    learning_goal: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON/session-state friendly representation."""

        return asdict(self)


@dataclass(frozen=True)
class SubstituentPattern:
    """SMARTS pattern used to classify a ring substituent."""

    label: str
    smarts: str
    director: str
    strength_label: str
    directing_weight: float
    control_priority: float
    fc_blocking: bool
    note: str


@dataclass(frozen=True)
class TargetTemplate:
    """Challenge template whose target is computed from a reagent sequence."""

    name: str
    route: tuple[str, ...]
    learning_goal: str


class EASGenerator:
    """Generate and execute aromatic synthesis practice problems."""

    def __init__(self, random_seed: Optional[int] = None) -> None:
        self.rng = random.Random(random_seed)
        self.starting_materials = self._build_starting_materials()
        self.reagents = self._build_reagent_library()
        self._reaction_cache = self._compile_reactions()
        self._substituent_patterns = self._build_substituent_patterns()
        self._compiled_substituent_patterns = [
            (pattern, Chem.MolFromSmarts(pattern.smarts))
            for pattern in self._substituent_patterns
        ]
        self.target_templates = self._build_target_templates()

    def list_reagents(self) -> list[Reagent]:
        """Return all supported reagents."""

        return list(self.reagents.values())

    def get_reagent(self, reagent_key: str) -> Reagent:
        """Return one reagent by key."""

        try:
            return self.reagents[reagent_key]
        except KeyError as exc:
            raise ValueError(f"Unknown reagent key: {reagent_key}") from exc

    def generate_single_step_problem(self) -> dict[str, object]:
        """Generate a random single-step EAS prediction problem."""

        eas_reagents = [reagent for reagent in self.reagents.values() if reagent.is_eas]

        for _ in range(100):
            substrate = self.rng.choice(self.starting_materials)
            reagent = self.rng.choice(eas_reagents)
            outcome = self.apply_reagent(substrate.smiles, reagent.key)
            if outcome.success:
                return {
                    "mode": "single_step",
                    "substrate_name": substrate.name,
                    "substrate_smiles": substrate.smiles,
                    "reagent_key": reagent.key,
                    "reagent": reagent.label,
                    "major_product_smiles": outcome.product_smiles,
                    "director_type": outcome.analysis_before.prevailing_director,
                    "analysis": outcome.analysis_before.to_dict(),
                    "outcome": outcome.to_dict(),
                }

        raise RuntimeError("Could not generate a compatible single-step problem.")

    def generate_synthesis_challenge(self) -> SynthesisChallenge:
        """Return a random target challenge from the route bank."""

        templates = list(self.target_templates)
        self.rng.shuffle(templates)

        for template in templates:
            product_smiles = self.apply_reagent_sequence(BENZENE_SMILES, template.route)
            if product_smiles:
                return SynthesisChallenge(
                    name=template.name,
                    target_smiles=product_smiles,
                    starting_smiles=BENZENE_SMILES,
                    recommended_route=template.route,
                    learning_goal=template.learning_goal,
                )

        raise RuntimeError("No synthesis challenge route produced a valid target.")

    def apply_reagent_sequence(
        self, starting_smiles: str, reagent_keys: tuple[str, ...] | list[str]
    ) -> Optional[str]:
        """Apply a complete reagent sequence and return the final SMILES."""

        current_smiles = self.canonicalize_smiles(starting_smiles)
        for reagent_key in reagent_keys:
            outcome = self.apply_reagent(current_smiles, reagent_key)
            if not outcome.success or outcome.product_smiles is None:
                return None
            current_smiles = outcome.product_smiles
        return current_smiles

    def apply_reagent(self, substrate_smiles: str, reagent_key: str) -> ReactionOutcome:
        """Apply one reagent to the current intermediate."""

        reagent = self.get_reagent(reagent_key)
        substrate_smiles = self.canonicalize_smiles(substrate_smiles)
        analysis_before = self.analyze_molecule(substrate_smiles)

        if reagent.is_friedel_crafts and not analysis_before.friedel_crafts_allowed:
            return ReactionOutcome(
                success=False,
                reagent_key=reagent.key,
                reagent_label=reagent.label,
                substrate_smiles=substrate_smiles,
                product_smiles=None,
                message=(
                    "Friedel-Crafts chemistry is not expected to work well here: "
                    "the ring is too deactivated or contains a Lewis-basic group "
                    "that complexes with AlCl3."
                ),
                analysis_before=analysis_before,
            )

        if reagent.is_eas:
            return self._apply_eas_reagent(substrate_smiles, reagent, analysis_before)

        return self._apply_fgi_reagent(substrate_smiles, reagent, analysis_before)

    def analyze_molecule(self, smiles: str) -> MoleculeAnalysis:
        """Identify ring substituents and summarize directing effects."""

        mol = self._mol_from_smiles(smiles)
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        ring_atoms = self._benzene_ring_atoms(mol)
        available_positions = tuple(
            sorted(
                atom_idx
                for atom_idx in ring_atoms
                if self._is_aromatic_carbon_with_hydrogen(mol.GetAtomWithIdx(atom_idx))
            )
        )
        substituents = tuple(
            self._classify_ring_substituent(mol, atom_idx)
            for atom_idx in self._substituted_ring_atoms(mol, ring_atoms)
        )

        prevailing_director = self._prevailing_director(substituents)
        ring_activation = self._ring_activation(substituents)
        fc_allowed, fc_warnings = self._friedel_crafts_allowed(substituents)

        warnings = list(fc_warnings)
        if not available_positions:
            warnings.append("No aromatic C-H positions remain for another EAS step.")

        return MoleculeAnalysis(
            smiles=smiles,
            canonical_smiles=canonical_smiles,
            substituents=substituents,
            available_ring_positions=available_positions,
            prevailing_director=prevailing_director,
            ring_activation=ring_activation,
            friedel_crafts_allowed=fc_allowed,
            warnings=tuple(warnings),
        )

    def canonicalize_smiles(self, smiles: str) -> str:
        """Return RDKit canonical SMILES."""

        mol = self._mol_from_smiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True)

    def _apply_eas_reagent(
        self,
        substrate_smiles: str,
        reagent: Reagent,
        analysis_before: MoleculeAnalysis,
    ) -> ReactionOutcome:
        """Enumerate possible EAS products and choose by directing rules."""

        if not analysis_before.available_ring_positions:
            return ReactionOutcome(
                success=False,
                reagent_key=reagent.key,
                reagent_label=reagent.label,
                substrate_smiles=substrate_smiles,
                product_smiles=None,
                message="No aromatic C-H positions are available for substitution.",
                analysis_before=analysis_before,
            )

        substrate = self._mol_from_smiles(substrate_smiles)
        reaction = self._reaction_cache[reagent.key][0]
        product_sets = reaction.RunReactants((substrate,))
        attack_atoms = [
            match[0] for match in substrate.GetSubstructMatches(Chem.MolFromSmarts("[cH]"))
        ]

        candidates_by_smiles: dict[str, ProductCandidate] = {}
        for index, product_tuple in enumerate(product_sets):
            if not product_tuple:
                continue

            product = self._sanitize_product(product_tuple[0])
            if product is None:
                continue

            product_smiles = Chem.MolToSmiles(product, canonical=True)
            attack_atom_idx = attack_atoms[index] if index < len(attack_atoms) else None
            score, orientation, explanation = self._score_eas_attack(
                analysis_before, attack_atom_idx
            )

            existing = candidates_by_smiles.get(product_smiles)
            if existing is None or score > existing.score:
                candidates_by_smiles[product_smiles] = ProductCandidate(
                    smiles=product_smiles,
                    attack_atom_idx=attack_atom_idx,
                    score=score,
                    orientation_summary=orientation,
                    explanation=explanation,
                )

        candidates = tuple(
            sorted(
                candidates_by_smiles.values(),
                key=lambda candidate: (-candidate.score, candidate.smiles),
            )
        )

        if not candidates:
            return ReactionOutcome(
                success=False,
                reagent_key=reagent.key,
                reagent_label=reagent.label,
                substrate_smiles=substrate_smiles,
                product_smiles=None,
                message=f"No valid product was generated for {reagent.label}.",
                analysis_before=analysis_before,
            )

        selected = candidates[0]
        analysis_after = self.analyze_molecule(selected.smiles)
        return ReactionOutcome(
            success=True,
            reagent_key=reagent.key,
            reagent_label=reagent.label,
            substrate_smiles=substrate_smiles,
            product_smiles=selected.smiles,
            message=(
                f"{reagent.short_name} gives the major product predicted by "
                f"{analysis_before.prevailing_director} directing effects."
            ),
            analysis_before=analysis_before,
            analysis_after=analysis_after,
            candidates=candidates,
            selected_candidate=selected,
        )

    def _apply_fgi_reagent(
        self,
        substrate_smiles: str,
        reagent: Reagent,
        analysis_before: MoleculeAnalysis,
    ) -> ReactionOutcome:
        """Apply a functional-group interconversion reagent."""

        products = self._run_transforms(
            substrate_smiles,
            self._reaction_cache[reagent.key],
            exhaustive=reagent.exhaustive,
        )

        if not products:
            return ReactionOutcome(
                success=False,
                reagent_key=reagent.key,
                reagent_label=reagent.label,
                substrate_smiles=substrate_smiles,
                product_smiles=None,
                message=(
                    f"{reagent.label} has no compatible functional group on this "
                    "intermediate."
                ),
                analysis_before=analysis_before,
            )

        product_smiles = sorted(products)[0]
        analysis_after = self.analyze_molecule(product_smiles)
        return ReactionOutcome(
            success=True,
            reagent_key=reagent.key,
            reagent_label=reagent.label,
            substrate_smiles=substrate_smiles,
            product_smiles=product_smiles,
            message=f"{reagent.short_name} converted the matching functional group.",
            analysis_before=analysis_before,
            analysis_after=analysis_after,
        )

    def _run_transforms(
        self,
        starting_smiles: str,
        reactions: tuple[ChemicalReaction, ...],
        exhaustive: bool,
    ) -> set[str]:
        """Run RDKit reactions once or repeatedly until no new products appear."""

        frontier = {self.canonicalize_smiles(starting_smiles)}
        products: set[str] = set()
        terminal_products: set[str] = set()
        max_rounds = 6 if exhaustive else 1

        for _ in range(max_rounds):
            next_frontier: set[str] = set()
            for smiles in frontier:
                reactant = self._mol_from_smiles(smiles)
                for reaction in reactions:
                    for product_tuple in reaction.RunReactants((reactant,)):
                        if not product_tuple:
                            continue
                        product = self._sanitize_product(product_tuple[0])
                        if product is None:
                            continue
                        product_smiles = Chem.MolToSmiles(product, canonical=True)
                        if product_smiles not in products and product_smiles != smiles:
                            products.add(product_smiles)
                            next_frontier.add(product_smiles)

            if not next_frontier:
                terminal_products = frontier if exhaustive else products
                break

            terminal_products = next_frontier
            if not exhaustive:
                break
            frontier = next_frontier

        if exhaustive and products and terminal_products:
            return terminal_products
        return products

    def _score_eas_attack(
        self, analysis: MoleculeAnalysis, attack_atom_idx: Optional[int]
    ) -> tuple[float, str, str]:
        """Score an EAS attack atom by all existing directing groups."""

        if attack_atom_idx is None or not analysis.substituents:
            return 1.0, "benzene-equivalent", "Unsubstituted benzene has one product."

        mol = self._mol_from_smiles(analysis.canonical_smiles)
        score = 0.0
        orientation_parts: list[str] = []
        explanation_parts: list[str] = []

        for substituent in analysis.substituents:
            distance = self._ring_distance(mol, substituent.attachment_idx, attack_atom_idx)
            orientation = self._orientation_from_distance(distance)
            orientation_parts.append(f"{orientation} to {substituent.label}")

            preferred = self._is_preferred_orientation(substituent.director, distance)
            if preferred:
                score += substituent.directing_weight
                explanation_parts.append(
                    f"{substituent.label} favors this {orientation} position"
                )
            else:
                score -= max(1.0, substituent.directing_weight * 0.35)

            if distance == 1 and substituent.strength_label in {
                "moderately activating",
                "weakly activating",
                "weakly deactivating",
                "moderately deactivating",
                "strongly deactivating",
            }:
                score -= 0.6
            if distance == 3 and substituent.director == "ortho/para":
                score += 0.25

        return (
            score,
            ", ".join(orientation_parts),
            "; ".join(explanation_parts) or "This site is not strongly directed.",
        )

    def _is_preferred_orientation(self, director: str, distance: int) -> bool:
        """Return whether a ring distance matches a director's preferred sites."""

        if director == "ortho/para":
            return distance in {1, 3}
        if director == "meta":
            return distance == 2
        return True

    def _orientation_from_distance(self, distance: int) -> str:
        """Convert ring distance to common disubstitution language."""

        return {1: "ortho", 2: "meta", 3: "para"}.get(distance, "unknown")

    def _ring_distance(self, mol: Chem.Mol, atom_a: int, atom_b: int) -> int:
        """Return the shortest path length between two ring atoms."""

        path = Chem.GetShortestPath(mol, atom_a, atom_b)
        return max(0, len(path) - 1)

    def _classify_ring_substituent(
        self, mol: Chem.Mol, ring_atom_idx: int
    ) -> RingSubstituent:
        """Classify the substituent attached at one ring atom."""

        for pattern, query in self._compiled_substituent_patterns:
            if query is None:
                continue
            for match in mol.GetSubstructMatches(query):
                if match and match[0] == ring_atom_idx:
                    return RingSubstituent(
                        attachment_idx=ring_atom_idx,
                        label=pattern.label,
                        director=pattern.director,
                        strength_label=pattern.strength_label,
                        directing_weight=pattern.directing_weight,
                        control_priority=pattern.control_priority,
                        fc_blocking=pattern.fc_blocking,
                        note=pattern.note,
                    )

        atom = mol.GetAtomWithIdx(ring_atom_idx)
        outside_neighbors = [
            neighbor
            for neighbor in atom.GetNeighbors()
            if not (neighbor.GetIsAromatic() and neighbor.IsInRing())
        ]
        neighbor_symbol = outside_neighbors[0].GetSymbol() if outside_neighbors else "R"
        return RingSubstituent(
            attachment_idx=ring_atom_idx,
            label=f"{neighbor_symbol} substituent",
            director="ortho/para",
            strength_label="weakly activating",
            directing_weight=2.0,
            control_priority=20.0,
            fc_blocking=False,
            note="Defaulted to a weak ortho/para director.",
        )

    def _prevailing_director(self, substituents: tuple[RingSubstituent, ...]) -> str:
        """Summarize the strongest directing effect on the ring."""

        if not substituents:
            return "none"

        dominant = max(substituents, key=lambda sub: sub.control_priority)
        return dominant.director

    def _ring_activation(self, substituents: tuple[RingSubstituent, ...]) -> str:
        """Summarize activation/deactivation of the ring."""

        if not substituents:
            return "neutral"

        if any("strongly activating" == sub.strength_label for sub in substituents):
            return "strongly activated"
        if any("moderately activating" == sub.strength_label for sub in substituents):
            return "activated"
        if any(sub.fc_blocking for sub in substituents):
            return "strongly deactivated"
        if any("deactivating" in sub.strength_label for sub in substituents):
            return "deactivated"
        return "weakly activated"

    def _friedel_crafts_allowed(
        self, substituents: tuple[RingSubstituent, ...]
    ) -> tuple[bool, tuple[str, ...]]:
        """Return whether Friedel-Crafts conditions are reasonable."""

        blockers = [sub for sub in substituents if sub.fc_blocking]
        if not blockers:
            return True, ()

        labels = ", ".join(sub.label for sub in blockers)
        return (
            False,
            (
                "Friedel-Crafts reactions fail or become impractical on rings "
                f"containing {labels}.",
            ),
        )

    def _benzene_ring_atoms(self, mol: Chem.Mol) -> set[int]:
        """Find the first aromatic six-carbon ring."""

        for ring in mol.GetRingInfo().AtomRings():
            if len(ring) != 6:
                continue
            if all(
                mol.GetAtomWithIdx(atom_idx).GetAtomicNum() == 6
                and mol.GetAtomWithIdx(atom_idx).GetIsAromatic()
                for atom_idx in ring
            ):
                return set(ring)
        raise ValueError("Expected an aromatic six-carbon ring.")

    def _substituted_ring_atoms(self, mol: Chem.Mol, ring_atoms: set[int]) -> list[int]:
        """Return ring atoms bearing at least one non-ring substituent."""

        substituted: list[int] = []
        for atom_idx in sorted(ring_atoms):
            atom = mol.GetAtomWithIdx(atom_idx)
            if any(neighbor.GetIdx() not in ring_atoms for neighbor in atom.GetNeighbors()):
                substituted.append(atom_idx)
        return substituted

    def _is_aromatic_carbon_with_hydrogen(self, atom: Chem.Atom) -> bool:
        """Return whether an atom can undergo an EAS C-H replacement."""

        return (
            atom.GetAtomicNum() == 6
            and atom.GetIsAromatic()
            and atom.GetTotalNumHs() > 0
        )

    def _mol_from_smiles(self, smiles: str) -> Chem.Mol:
        """Parse and sanitize a SMILES string."""

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Could not parse SMILES: {smiles}")
        Chem.SanitizeMol(mol)
        return mol

    def _sanitize_product(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Sanitize a product molecule, returning None for invalid products."""

        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None

        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        return mol

    def _compile_reactions(self) -> dict[str, tuple[ChemicalReaction, ...]]:
        """Compile all reagent SMARTS into RDKit reactions."""

        compiled: dict[str, tuple[ChemicalReaction, ...]] = {}
        for key, reagent in self.reagents.items():
            reactions: list[ChemicalReaction] = []
            for smarts in reagent.reaction_smarts:
                reaction = AllChem.ReactionFromSmarts(smarts)
                if reaction is None:
                    raise ValueError(f"Could not parse SMARTS for {reagent.label}")
                reaction.Initialize()
                reactions.append(reaction)
            compiled[key] = tuple(reactions)
        return compiled

    def _build_starting_materials(self) -> list[StartingMaterial]:
        """Return common undergraduate aromatic substrates."""

        return [
            StartingMaterial("benzene", BENZENE_SMILES),
            StartingMaterial("toluene", "Cc1ccccc1"),
            StartingMaterial("ethylbenzene", "CCc1ccccc1"),
            StartingMaterial("isopropylbenzene", "CC(C)c1ccccc1"),
            StartingMaterial("phenol", "Oc1ccccc1"),
            StartingMaterial("anisole", "COc1ccccc1"),
            StartingMaterial("aniline", "Nc1ccccc1"),
            StartingMaterial("chlorobenzene", "Clc1ccccc1"),
            StartingMaterial("bromobenzene", "Brc1ccccc1"),
            StartingMaterial("nitrobenzene", "O=[N+]([O-])c1ccccc1"),
            StartingMaterial("acetophenone", "CC(=O)c1ccccc1"),
            StartingMaterial("benzoic acid", "O=C(O)c1ccccc1"),
            StartingMaterial("benzonitrile", "N#Cc1ccccc1"),
            StartingMaterial("benzenesulfonic acid", "O=S(=O)(O)c1ccccc1"),
        ]

    def _build_reagent_library(self) -> dict[str, Reagent]:
        """Return the supported textbook reagent library."""

        reagents = [
            Reagent(
                key="nitration",
                label="HNO3/H2SO4",
                short_name="Nitration",
                category="Standard EAS",
                reaction_smarts=("[cH:1]>>[c:1][N+](=O)[O-]",),
                description="Adds NO2 through the nitronium ion.",
                is_eas=True,
            ),
            Reagent(
                key="bromination",
                label="Br2/FeBr3",
                short_name="Bromination",
                category="Standard EAS",
                reaction_smarts=("[cH:1]>>[c:1]Br",),
                description="Adds Br through Lewis-acid activation.",
                is_eas=True,
            ),
            Reagent(
                key="chlorination",
                label="Cl2/FeCl3",
                short_name="Chlorination",
                category="Standard EAS",
                reaction_smarts=("[cH:1]>>[c:1]Cl",),
                description="Adds Cl through Lewis-acid activation.",
                is_eas=True,
            ),
            Reagent(
                key="sulfonation",
                label="SO3/H2SO4",
                short_name="Sulfonation",
                category="Standard EAS",
                reaction_smarts=("[cH:1]>>[c:1]S(=O)(=O)O",),
                description="Adds SO3H under fuming sulfuric acid conditions.",
                is_eas=True,
            ),
            Reagent(
                key="fc_methylation",
                label="CH3Cl/AlCl3",
                short_name="Friedel-Crafts methylation",
                category="Friedel-Crafts",
                reaction_smarts=("[cH:1]>>[c:1]C",),
                description="Adds a methyl group to rings that tolerate AlCl3.",
                is_eas=True,
                is_friedel_crafts=True,
            ),
            Reagent(
                key="fc_ethylation",
                label="CH3CH2Cl/AlCl3",
                short_name="Friedel-Crafts ethylation",
                category="Friedel-Crafts",
                reaction_smarts=("[cH:1]>>[c:1]CC",),
                description="Adds an ethyl group under Friedel-Crafts conditions.",
                is_eas=True,
                is_friedel_crafts=True,
            ),
            Reagent(
                key="fc_isopropylation",
                label="(CH3)2CHCl/AlCl3",
                short_name="Friedel-Crafts isopropylation",
                category="Friedel-Crafts",
                reaction_smarts=("[cH:1]>>[c:1]C(C)C",),
                description="Adds an isopropyl group.",
                is_eas=True,
                is_friedel_crafts=True,
            ),
            Reagent(
                key="fc_acylation",
                label="CH3COCl/AlCl3",
                short_name="Friedel-Crafts acylation",
                category="Friedel-Crafts",
                reaction_smarts=("[cH:1]>>[c:1]C(C)=O",),
                description="Adds an acetyl group without carbocation rearrangement.",
                is_eas=True,
                is_friedel_crafts=True,
            ),
            Reagent(
                key="clemmensen",
                label="Zn(Hg), HCl",
                short_name="Clemmensen reduction",
                category="Functional Group Interconversion",
                reaction_smarts=("[c:1][C:2](=[O:3])[CH3:4]>>[c:1][CH2:2][CH3:4]",),
                description="Converts aryl methyl ketones into ethyl groups.",
            ),
            Reagent(
                key="nitro_reduction",
                label="H2, Pd/C or Fe/HCl",
                short_name="Nitro reduction",
                category="Functional Group Interconversion",
                reaction_smarts=("[c:1][N+:2](=[O:3])[O-:4]>>[c:1][NH2:2]",),
                description="Converts aryl nitro groups into anilines.",
            ),
            Reagent(
                key="side_chain_oxidation",
                label="KMnO4, heat",
                short_name="Side-chain oxidation",
                category="Functional Group Interconversion",
                reaction_smarts=(
                    "[c:1][CH3:2]>>[c:1]C(=O)O",
                    "[c:1][CH2:2][#6:3]>>[c:1]C(=O)O",
                    "[c:1][CH:2]([#6:3])[#6:4]>>[c:1]C(=O)O",
                ),
                description="Oxidizes alkyl substituents with benzylic H to CO2H.",
                exhaustive=True,
            ),
            Reagent(
                key="sandmeyer_cl",
                label="1. NaNO2/HCl  2. CuCl",
                short_name="Sandmeyer chlorination",
                category="Diazotization/Sandmeyer",
                reaction_smarts=("[c:1][NH2:2]>>[c:1]Cl",),
                description="Converts aniline into an aryl chloride.",
            ),
            Reagent(
                key="sandmeyer_br",
                label="1. NaNO2/HCl  2. CuBr",
                short_name="Sandmeyer bromination",
                category="Diazotization/Sandmeyer",
                reaction_smarts=("[c:1][NH2:2]>>[c:1]Br",),
                description="Converts aniline into an aryl bromide.",
            ),
            Reagent(
                key="sandmeyer_cn",
                label="1. NaNO2/HCl  2. CuCN",
                short_name="Sandmeyer cyanation",
                category="Diazotization/Sandmeyer",
                reaction_smarts=("[c:1][NH2:2]>>[c:1]C#N",),
                description="Converts aniline into an aryl nitrile.",
            ),
            Reagent(
                key="sandmeyer_i",
                label="1. NaNO2/HCl  2. KI",
                short_name="Sandmeyer iodination",
                category="Diazotization/Sandmeyer",
                reaction_smarts=("[c:1][NH2:2]>>[c:1]I",),
                description="Converts aniline into an aryl iodide.",
            ),
            Reagent(
                key="diazonium_h",
                label="1. NaNO2/HCl  2. H3PO2",
                short_name="Diazonium replacement by H",
                category="Diazotization/Sandmeyer",
                reaction_smarts=("[c:1][NH2:2]>>[cH:1]",),
                description="Replaces an aniline nitrogen with hydrogen.",
            ),
        ]
        return {reagent.key: reagent for reagent in reagents}

    def _build_substituent_patterns(self) -> tuple[SubstituentPattern, ...]:
        """Return ordered substituent classifiers, prioritized by activating strength."""

        return (
            SubstituentPattern("amino", "[c][NH2]", "ortho/para", "strongly activating", 6.0, 100.0, True, "Strong activator; complexes with AlCl3."),
            SubstituentPattern("hydroxy", "[c][OH]", "ortho/para", "strongly activating", 6.0, 95.0, True, "Strong ortho/para director."),
            SubstituentPattern("alkoxy", "[c]O[#6]", "ortho/para", "moderately activating", 5.0, 90.0, False, "Oxygen lone-pair donation directs ortho/para."),
            SubstituentPattern("acylamino", "[c]N([H])C(=O)[#6]", "ortho/para", "moderately activating", 4.5, 85.0, True, "Amide lone-pair donation directs ortho/para."),
            SubstituentPattern("alkyl", "[c][CX4;H1,H2,H3]", "ortho/para", "weakly activating", 3.0, 70.0, False, "Alkyl groups donate by hyperconjugation."),
            
            # CRITICAL FIX: Halogens are deactivating, but they direct o/p and override meta-directors!
            SubstituentPattern("fluoro", "[c]F", "ortho/para", "weakly deactivating", 2.0, 60.0, False, "Halogens deactivate but direct ortho/para."),
            SubstituentPattern("chloro", "[c]Cl", "ortho/para", "weakly deactivating", 2.0, 60.0, False, "Halogens deactivate but direct ortho/para."),
            SubstituentPattern("bromo", "[c]Br", "ortho/para", "weakly deactivating", 2.0, 60.0, False, "Halogens deactivate but direct ortho/para."),
            SubstituentPattern("iodo", "[c]I", "ortho/para", "weakly deactivating", 2.0, 60.0, False, "Halogens deactivate but direct ortho/para."),
            
            # Meta-directing deactivators
            SubstituentPattern("carboxylic acid", "[c]C(=O)O", "meta", "moderately deactivating", 4.5, 40.0, True, "Carbonyl withdraws electron density and directs meta."),
            SubstituentPattern("acyl", "[c]C(=O)[#6]", "meta", "moderately deactivating", 4.5, 40.0, True, "Aryl ketones are meta-directing deactivators."),
            SubstituentPattern("cyano", "[c]C#N", "meta", "moderately deactivating", 4.5, 40.0, True, "Nitriles withdraw inductively and by resonance."),
            SubstituentPattern("sulfonic acid", "[c]S(=O)(=O)O", "meta", "strongly deactivating", 5.0, 30.0, True, "SO3H is a strongly deactivating meta director."),
            SubstituentPattern("nitro", "[c][N+](=O)[O-]", "meta", "strongly deactivating", 5.5, 20.0, True, "Strong meta director; strongly deactivates the ring."),
        )

    def _build_target_templates(self) -> tuple[TargetTemplate, ...]:
        """Dynamically generate 120+ synthesis targets across all difficulty levels."""
        
        templates = []

        # Helper lists for combinatorics
        halogens = [("bromo", "bromination"), ("chloro", "chlorination")]
        alkyls = [("methyl", "fc_methylation"), ("ethyl", "fc_ethylation"), ("isopropyl", "fc_isopropylation")]
        sandmeyers = [("bromo", "sandmeyer_br"), ("chloro", "sandmeyer_cl"), ("cyano", "sandmeyer_cn"), ("iodo", "sandmeyer_i")]

        # ==========================================
        # LEVEL 1: 2-STEP WARM-UPS (30+ Targets)
        # ==========================================
        for hal, eas in halogens:
            templates.append(TargetTemplate(f"m-{hal}nitrobenzene", ("nitration", eas), "Use a meta director first."))
            templates.append(TargetTemplate(f"p-{hal}nitrobenzene", (eas, "nitration"), "Use an ortho/para director first."))
            templates.append(TargetTemplate(f"m-{hal}benzenesulfonic acid", ("sulfonation", eas), "Sulfonate first to direct meta."))
            templates.append(TargetTemplate(f"p-{hal}benzenesulfonic acid", (eas, "sulfonation"), "Halogenate first to direct para."))
            templates.append(TargetTemplate(f"m-{hal}acetophenone", ("fc_acylation", eas), "Acylate first to direct meta."))
            
            for alk, alk_eas in alkyls:
                templates.append(TargetTemplate(f"p-{hal}{alk}benzene", (alk_eas, eas), "Install the alkyl group to direct para."))
                
        templates.append(TargetTemplate("m-dinitrobenzene", ("nitration", "nitration"), "Double nitration."))
        templates.append(TargetTemplate("m-nitrobenzenesulfonic acid", ("nitration", "sulfonation"), "Deactivator followed by deactivator."))
        templates.append(TargetTemplate("benzoic acid", ("fc_methylation", "side_chain_oxidation"), "Alkyl installation followed by benzylic oxidation."))
        templates.append(TargetTemplate("ethylbenzene", ("fc_acylation", "clemmensen"), "Acylation followed by carbonyl reduction."))

        # ==========================================
        # LEVEL 2: 3-STEP INTERMEDIATE (40+ Targets)
        # ==========================================
        for hal, eas in halogens:
            # Benzoic Acid derivatives
            templates.append(TargetTemplate(f"p-{hal}benzoic acid", ("fc_methylation", eas, "side_chain_oxidation"), "Halogenate while the methyl group is still an o/p director."))
            templates.append(TargetTemplate(f"m-{hal}benzoic acid", ("fc_methylation", "side_chain_oxidation", eas), "Oxidize the methyl group to a meta-director before halogenating."))
            
            # Aniline derivatives
            templates.append(TargetTemplate(f"m-{hal}aniline", ("nitration", eas, "nitro_reduction"), "Use the nitro group to direct meta, then reduce it."))
            templates.append(TargetTemplate(f"p-{hal}aniline", (eas, "nitration", "nitro_reduction"), "Halogenate first, nitrate para, then reduce."))
            
            # Clemmensen chains
            templates.append(TargetTemplate(f"m-{hal}ethylbenzene", ("fc_acylation", eas, "clemmensen"), "Use the acyl group to direct meta, then reduce it to an ethyl group."))
            templates.append(TargetTemplate(f"1-{hal}-3-propylbenzene", ("fc_acylation", eas, "clemmensen"), "Acyl directs meta, then reduce.")) # Matches acylation logic

        templates.append(TargetTemplate("p-nitrobenzoic acid", ("fc_methylation", "nitration", "side_chain_oxidation"), "Nitrate para to the methyl, then oxidize."))
        templates.append(TargetTemplate("m-nitrobenzoic acid", ("fc_methylation", "side_chain_oxidation", "nitration"), "Oxidize to acid first, then nitrate meta."))
        
        # Base Sandmeyers
        for group, sand in sandmeyers:
            templates.append(TargetTemplate(f"{group}benzene", ("nitration", "nitro_reduction", sand), "Install nitrogen, reduce to aniline, then Sandmeyer."))

        # ==========================================
        # LEVEL 3: 4-STEP ADVANCED (30+ Targets)
        # ==========================================
        # Sandmeyer combinations (1,3-disubstituted where one is a tricky group)
        for hal1, eas1 in halogens:
            for group2, sand2 in sandmeyers:
                templates.append(TargetTemplate(f"1-{hal1}-3-{group2}benzene", ("nitration", eas1, "nitro_reduction", sand2), "Nitrate, halogenate meta, reduce to aniline, then Sandmeyer."))
                templates.append(TargetTemplate(f"1-{group2}-4-methylbenzene", ("fc_methylation", "nitration", "nitro_reduction", sand2), "Alkyl directs nitro para, reduce, then Sandmeyer."))

        templates.append(TargetTemplate("p-aminobenzoic acid", ("fc_methylation", "nitration", "side_chain_oxidation", "nitro_reduction"), "Methyl directs NO2 para, oxidize methyl, reduce NO2."))
        templates.append(TargetTemplate("m-aminobenzoic acid", ("fc_methylation", "side_chain_oxidation", "nitration", "nitro_reduction"), "Oxidize methyl to acid, nitrate meta, reduce NO2."))
        
        for hal, eas in halogens:
            templates.append(TargetTemplate(f"3-{hal}-5-nitrobenzoic acid", ("fc_methylation", "side_chain_oxidation", "nitration", eas), "Oxidize to acid, nitrate meta, halogenate meta."))

        # ==========================================
        # LEVEL 4: 5-STEP EXPERT (20+ Targets)
        # ==========================================
        for group, sand in sandmeyers:
            templates.append(TargetTemplate(f"3-{group}benzoic acid", ("fc_methylation", "side_chain_oxidation", "nitration", "nitro_reduction", sand), "Acid directs NO2 meta, reduce to aniline, Sandmeyer."))
            templates.append(TargetTemplate(f"4-{group}benzoic acid", ("fc_methylation", "nitration", "side_chain_oxidation", "nitro_reduction", sand), "Methyl directs NO2 para, oxidize methyl, reduce to aniline, Sandmeyer."))

        return tuple(templates)


if __name__ == "__main__":
    generator = EASGenerator(random_seed=4)
    challenge = generator.generate_synthesis_challenge()
    print(challenge)
