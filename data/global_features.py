"""
Global feature extraction functions for crystal structures
"""

import numpy as np
from pymatgen.core import Structure
from scipy.spatial import KDTree
from pymatgen.core.periodic_table import Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from typing import Dict, List

from config.constants import COVALENT_RADII, CRYSTAL_SYSTEMS, POINT_GROUPS


def atomic_volume(element_symbol: str) -> float:
    """Calculate atomic volume based on covalent radius"""
    r = COVALENT_RADII.get(element_symbol, 1.0)
    return (4/3) * np.pi * (r ** 3)


def compute_packing_density(structure: Structure) -> float:
    """Compute packing density of a crystal structure"""
    total_volume = 0.0
    for site in structure.sites:
        elem = site.specie.symbol
        total_volume += atomic_volume(elem)
    return total_volume / structure.lattice.volume


def compute_avg_bond_length(structure: Structure, cutoff: float = 3.0) -> float:
    """Compute average bond length in the structure"""
    positions = np.array([site.coords for site in structure.sites])
    kdtree = KDTree(positions)
    bond_lengths = []
    for i, pos in enumerate(positions):
        neighbors = [j for j in kdtree.query_ball_point(pos, cutoff) if j != i]
        for j in neighbors:
            bond_lengths.append(np.linalg.norm(pos - positions[j]))
    return np.mean(bond_lengths) if bond_lengths else 0.0


def compute_avg_coordination_number(structure: Structure, cutoff: float = 3.0) -> float:
    """Compute average coordination number"""
    positions = np.array([site.coords for site in structure.sites])
    kdtree = KDTree(positions)
    coordination_numbers = []
    for i, pos in enumerate(positions):
        indices = kdtree.query_ball_point(pos, cutoff)
        coordination_numbers.append(len(indices) - 1)
    return np.mean(coordination_numbers) if coordination_numbers else 0.0


def compute_avg_bond_angle(structure: Structure, cutoff: float = 3.0) -> float:
    """Compute average bond angle"""
    positions = np.array([site.coords for site in structure.sites])
    kdtree = KDTree(positions)
    angles = []
    for i, pos in enumerate(positions):
        neighbors = [positions[j] for j in kdtree.query_ball_point(pos, cutoff) if j != i]
        for k in range(len(neighbors)):
            for l in range(k + 1, len(neighbors)):
                vec1 = neighbors[k] - pos
                vec2 = neighbors[l] - pos
                cosine = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                cosine = np.clip(cosine, -1.0, 1.0)
                angles.append(np.degrees(np.arccos(cosine)))
    return np.mean(angles) if angles else 0.0


def compute_polyhedra_shape_index(structure: Structure, cutoff: float = 3.0) -> float:
    """Compute polyhedra shape index"""
    positions = np.array([site.coords for site in structure.sites])
    kdtree = KDTree(positions)
    shape_indices = []
    for i, pos in enumerate(positions):
        neighbors = [positions[j] for j in kdtree.query_ball_point(pos, cutoff) if j != i]
        angles = []
        for k in range(len(neighbors)):
            for l in range(k + 1, len(neighbors)):
                vec1 = neighbors[k] - pos
                vec2 = neighbors[l] - pos
                cosine = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                cosine = np.clip(cosine, -1.0, 1.0)
                angles.append(np.degrees(np.arccos(cosine)))
        if angles:
            shape_indices.append(np.std(angles))
    return np.mean(shape_indices) if shape_indices else 0.0


def compute_avg_en_diff(structure: Structure, cutoff: float = 3.0) -> float:
    """Compute average electronegativity difference"""
    positions = np.array([site.coords for site in structure.sites])
    kdtree = KDTree(positions)
    en_diffs = []
    for i, site in enumerate(structure.sites):
        en_center = Element(site.specie.symbol).X or 0.0
        neighbors = [j for j in kdtree.query_ball_point(site.coords, cutoff) if j != i]
        if neighbors:
            en_neighbors = [Element(structure.sites[j].specie.symbol).X or 0.0 for j in neighbors]
            en_diffs.append(abs(en_center - np.mean(en_neighbors)))
    return np.mean(en_diffs) if en_diffs else 0.0


def compute_num_distinct_wyckoffs(structure: Structure) -> float:
    """Compute number of distinct Wyckoff positions"""
    try:
        sga = SpacegroupAnalyzer(structure)
        sym = sga.get_symmetry_dataset()
        return float(len(set(sym.get('wyckoffs', [])))) if sym else 0.0
    except:
        return 0.0


def extract_global_features(cif_file: str) -> Dict[str, any]:
    """Extract all global features from a CIF file"""
    try:
        structure = Structure.from_file(cif_file)
    except Exception as e:
        print(f"Could not read CIF file {cif_file}: {e}")
        return {}
    
    lat = structure.lattice
    a, b, c = lat.a, lat.b, lat.c
    alpha, beta, gamma = lat.alpha, lat.beta, lat.gamma
    volume = lat.volume
    packing_density = compute_packing_density(structure)
    ratio_c_a = c / a
    ratio_b_a = b / a
    num_atoms = structure.num_sites

    # Local geometry
    avg_bond_length = compute_avg_bond_length(structure)
    avg_coordination_number = compute_avg_coordination_number(structure)
    avg_bond_angle = compute_avg_bond_angle(structure)
    polyhedra_shape_index = compute_polyhedra_shape_index(structure)

    # Symmetry
    num_distinct_wyckoffs = compute_num_distinct_wyckoffs(structure)
    try:
        sga = SpacegroupAnalyzer(structure)
        pg = sga.get_point_group_symbol() or ''
        cs = sga.get_crystal_system() or ''
    except:
        pg, cs = '', ''

    # Chemistry
    avg_en_diff = compute_avg_en_diff(structure)

    return {
        'a': a, 'b': b, 'c': c,
        'alpha': alpha, 'beta': beta, 'gamma': gamma,
        'volume': volume,
        'packing_density': packing_density,
        'ratio_c_a': ratio_c_a, 'ratio_b_a': ratio_b_a,
        'num_atoms': num_atoms,
        'avg_bond_length': avg_bond_length,
        'avg_coordination_number': avg_coordination_number,
        'avg_bond_angle': avg_bond_angle,
        'num_distinct_wyckoffs': num_distinct_wyckoffs,
        'avg_en_diff': avg_en_diff,
        'polyhedra_shape_index': polyhedra_shape_index,
        'point_group': pg,
        'crystal_system': cs
    }


def one_hot_encode(value: str, categories: List[str]) -> np.ndarray:
    """One-hot encode categorical values"""
    vec = np.zeros(len(categories))
    try:
        vec[categories.index(value.lower())] = 1.0
    except ValueError:
        pass
    return vec


def build_global_feature_vector(features: Dict[str, any]) -> np.ndarray:
    """Build global feature vector from extracted features"""
    continuous = np.array([
        features['a'], features['b'], features['c'],
        features['alpha'], features['beta'], features['gamma'],
        features['volume'], features['packing_density'],
        features['ratio_c_a'], features['ratio_b_a'],
        features['num_atoms'],
        features['avg_bond_length'], features['avg_coordination_number'],
        features['avg_bond_angle'],
        features['num_distinct_wyckoffs'],
        features['avg_en_diff'], features['polyhedra_shape_index']
    ], dtype=np.float32)

    cs_vec = one_hot_encode(features['crystal_system'], CRYSTAL_SYSTEMS)
    pg_vec = one_hot_encode(features['point_group'], POINT_GROUPS)
    return np.concatenate([continuous, cs_vec, pg_vec])