"""
Grain Registry - Centralized Grain Storage

Manages grain storage, lookup, and lifecycle.
Provides state indexing and L1/L2 breakdown capabilities.

Author: Kitbash Team
Date: February 2026
Phase: 2C Consolidation → Modularized (May 2026)
"""

import json
import statistics
from pathlib import Path
from typing import Dict, List, Set, Any

from .data_structures import GrainMetadata, GrainState


class GrainRegistry:
    """
    Centralized registry of crystallized grains.
    Manages grain storage, lookup, and lifecycle.
    
    From shannon_grain.py
    """
    
    def __init__(self, cartridge_id: str, storage_path: str = "./grains"):
        """Initialize grain registry"""
        self.cartridge_id = cartridge_id
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory grain index
        self.grains: Dict[str, GrainMetadata] = {}
        self.grain_state_index: Dict[GrainState, Set[str]] = {
            state: set() for state in GrainState
        }
    
    def add_grain(self, grain: GrainMetadata) -> None:
        """Add a grain to the registry"""
        self.grains[grain.grain_id] = grain
        self.grain_state_index[grain.state].add(grain.grain_id)
    
    def update_grain_state(self, grain_id: str, new_state: GrainState) -> None:
        """Update grain state (e.g., CANDIDATE → CRYSTALLIZED)"""
        if grain_id not in self.grains:
            raise KeyError(f"Grain not found: {grain_id}")
        
        grain = self.grains[grain_id]
        old_state = grain.state
        
        # Remove from old state index
        self.grain_state_index[old_state].discard(grain_id)
        
        # Update and re-index
        grain.state = new_state
        self.grain_state_index[new_state].add(grain_id)
    
    def get_grains_by_state(self, state: GrainState) -> List[GrainMetadata]:
        """Get all grains in a specific state"""
        grain_ids = self.grain_state_index[state]
        return [self.grains[gid] for gid in grain_ids]
    
    def get_active_grains(self) -> List[GrainMetadata]:
        """Get all active grains (in L3 cache)"""
        return self.get_grains_by_state(GrainState.ACTIVE)
    
    def get_crystallized_grains(self) -> List[GrainMetadata]:
        """Get all crystallized grains"""
        active = set(g.grain_id for g in self.get_grains_by_state(GrainState.ACTIVE))
        crystallized = self.get_grains_by_state(GrainState.CRYSTALLIZED)
        return crystallized + [self.grains[gid] for gid in active if gid not in active]
    
    def get_axioms(self) -> List[GrainMetadata]:
        """Return only axiom-type grains (grain_type='axiom')"""
        return [g for g in self.grains.values() if hasattr(g, 'grain_type') and g.grain_type == "axiom"]
    
    def get_observations(self) -> List[GrainMetadata]:
        """Return only observation-type grains (grain_type='observation')"""
        return [g for g in self.grains.values() if hasattr(g, 'grain_type') and g.grain_type == "observation"]
    
    def save_grain(self, grain: GrainMetadata) -> Path:
        """Save a grain to disk"""
        grain_path = self.storage_path / f"{grain.grain_id}.json"
        
        # Save metadata (bits stored separately)
        metadata = grain.to_dict()
        with open(grain_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save bit arrays if present
        if grain.bit_array_plus:
            bits_plus_path = self.storage_path / f"{grain.grain_id}.plus.bin"
            with open(bits_plus_path, 'wb') as f:
                f.write(grain.bit_array_plus)
        
        if grain.bit_array_minus:
            bits_minus_path = self.storage_path / f"{grain.grain_id}.minus.bin"
            with open(bits_minus_path, 'wb') as f:
                f.write(grain.bit_array_minus)
        
        return grain_path
    
    def get_stats(self) -> Dict:
        """Get registry statistics with L1/L2 breakdown"""
        axioms = self.get_axioms()
        observations = self.get_observations()
        
        return {
            "cartridge_id": self.cartridge_id,
            "total_grains": len(self.grains),
            # ===== L1/L2 BREAKDOWN (NEW) =====
            "axioms": len(axioms),
            "observations": len(observations),
            "avg_confidence_axiom": statistics.mean([g.confidence for g in axioms]) if axioms else 0.0,
            "avg_confidence_observation": statistics.mean([g.confidence for g in observations]) if observations else 0.0,
            # ===== EXISTING STATS =====
            "by_state": {
                state.value: len(grain_ids)
                for state, grain_ids in self.grain_state_index.items()
            },
            "total_storage_mb": sum(g.size_mb() for g in self.grains.values()),
            "avg_grain_size_bytes": statistics.mean([len(g.bit_array_plus) + len(g.bit_array_minus) 
                                                      for g in self.grains.values()]) if self.grains else 0,
        }
