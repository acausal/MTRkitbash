"""
Phantom Tracker - Query Pattern Detection

Tracks persistent query patterns and detects harmonic lock.
Integrates with DeltaRegistry to identify crystallization candidates.

Author: Kitbash Team
Date: February 2026
Phase: 2C Consolidation → Modularized (May 2026)
"""

import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Set, Any

from .data_structures import PhantomCandidate, EpistemicLevel


class PhantomTracker:
    """
    Tracks persistent query patterns and detects harmonic lock.
    Integrates with DeltaRegistry to identify crystallization candidates.
    
    From shannon_grain.py
    """
    
    def __init__(self, cartridge_id: str, 
                 persistence_threshold: int = 5,
                 confidence_threshold: float = 0.75,
                 harmonic_lock_cycles: int = 50):
        """Initialize phantom tracker."""
        self.cartridge_id = cartridge_id
        self.persistence_threshold = persistence_threshold
        self.confidence_threshold = confidence_threshold
        self.harmonic_lock_cycles = harmonic_lock_cycles
        
        # Phantom storage
        self.phantoms: Dict[str, PhantomCandidate] = {}
        self.cycle_count = 0
        
        # Stats
        self.total_hits = 0
    
    def record_phantom_hit(self, fact_ids: Set[int], concepts: List[str],
                          confidence: float, epistemic_level: EpistemicLevel = EpistemicLevel.L2_AXIOMATIC) -> None:
        """Record a phantom hit (called during query execution)"""
        
        # Create phantom ID from fact IDs
        phantom_key = "_".join(str(fid) for fid in sorted(fact_ids))
        
        if phantom_key not in self.phantoms:
            self.phantoms[phantom_key] = PhantomCandidate(
                phantom_id=phantom_key,
                fact_ids=fact_ids,
                cartridge_id=self.cartridge_id,
                epistemic_level=epistemic_level,
            )
        
        phantom = self.phantoms[phantom_key]
        phantom.hit_count += 1
        phantom.confidence_scores.append(confidence)
        phantom.query_concepts.extend(concepts)
        phantom.last_cycle_seen = self.cycle_count
        
        if phantom.first_cycle_seen == 0:
            phantom.first_cycle_seen = self.cycle_count
        
        self.total_hits += 1
        
        # Check persistence criteria
        if phantom.is_persistent(self.persistence_threshold, self.confidence_threshold):
            phantom.status = "persistent"
    
    def advance_cycle(self) -> None:
        """Advance cycle and check for harmonic lock"""
        
        self.cycle_count += 1
        
        # Record hits in this cycle for all phantoms
        for phantom in self.phantoms.values():
            phantom.hit_history.append(phantom.hit_count)
            phantom.hit_count = 0  # Reset for next cycle
            
            # Check for locked status
            if len(phantom.hit_history) >= self.harmonic_lock_cycles:
                self._calculate_consistency(phantom)
    
    def _calculate_consistency(self, phantom: PhantomCandidate) -> None:
        """Calculate consistency and detect harmonic lock"""
        
        # Calculate hit consistency (variance in hit counts)
        if len(phantom.hit_history) > 1:
            try:
                hit_variance = statistics.variance(phantom.hit_history[-50:])
                hit_consistency = 1.0 - min(hit_variance / 10.0, 1.0)
            except:
                hit_consistency = 0.0
        else:
            hit_consistency = 0.0
        
        # Calculate confidence consistency
        confidence_consistency = 1.0 - min(statistics.variance(phantom.confidence_scores) / 0.25, 1.0) \
            if len(phantom.confidence_scores) > 1 else 0.0
        
        # Harmonic lock achieved if both are consistent
        overall_consistency = (hit_consistency + confidence_consistency) / 2.0
        phantom.cycle_consistency = overall_consistency
        
        if hit_consistency > 0.8 and confidence_consistency > 0.8:
            phantom.status = "locked"
    
    def get_persistent_phantoms(self) -> List[PhantomCandidate]:
        """Get all persistent phantoms (5+ hits, high confidence)"""
        return [p for p in self.phantoms.values() if p.status == "persistent"]
    
    def get_locked_phantoms(self) -> List[PhantomCandidate]:
        """Get all locked phantoms (ready for crystallization)"""
        return [p for p in self.phantoms.values() if p.status == "locked"]
    
    def get_stats(self) -> Dict:
        """Get tracker statistics"""
        persistent = self.get_persistent_phantoms()
        locked = self.get_locked_phantoms()
        
        return {
            "cartridge_id": self.cartridge_id,
            "cycle_count": self.cycle_count,
            "total_hits": self.total_hits,
            "total_phantoms": len(self.phantoms),
            "persistent_count": len(persistent),
            "locked_count": len(locked),
            "crystallization_ready": len(locked),
            "avg_phantom_hits": statistics.mean([p.hit_count for p in self.phantoms.values()]) if self.phantoms else 0,
        }
    
    def save(self, filepath: str) -> None:
        """Save phantom tracker to JSON"""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "cartridge_id": self.cartridge_id,
            "cycle_count": self.cycle_count,
            "total_hits": self.total_hits,
            "phantoms": {
                key: phantom.to_dict()
                for key, phantom in self.phantoms.items()
            },
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> "PhantomTracker":
        """Load phantom tracker from JSON"""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(path) as f:
            data = json.load(f)
        
        tracker = cls(data["cartridge_id"])
        tracker.cycle_count = data.get("cycle_count", 0)
        tracker.total_hits = data.get("total_hits", 0)
        
        return tracker
