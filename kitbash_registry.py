"""
Kitbash DeltaRegistry
Learning infrastructure - tracks patterns, identifies phantoms, monitors cycles

The Delta Registry is the perception system for the Cartridge:
- Records every query hit with confidence
- Identifies persistent patterns (phantoms)
- Tracks cycle consistency for harmonic lock detection
- Monitors which facts are ready for crystallization
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from collections import defaultdict
import statistics


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class QueryHit:
    """Single query hit record."""
    fact_id: int
    cartridge_id: str
    query_concepts: List[str]
    confidence: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class PhantomPattern:
    """Query pattern that appears consistently."""
    concepts: Tuple[str, ...]  # Sorted tuple of query concepts
    count: int = 0  # How many times this pattern appeared
    first_seen: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_seen: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class PhantomCandidate:
    """A fact that shows phantom behavior."""
    fact_id: int
    cartridge_id: str
    hit_count: int = 0
    confidence_history: List[float] = field(default_factory=list)
    query_patterns: Dict[str, int] = field(default_factory=dict)  # pattern_key -> count
    first_cycle_seen: int = 0
    last_cycle_seen: int = 0
    cycle_consistency: float = 0.0
    status: str = "none"  # none, transient, persistent, locked
    
    def to_dict(self) -> dict:
        return {
            "fact_id": self.fact_id,
            "cartridge_id": self.cartridge_id,
            "hit_count": self.hit_count,
            "confidence_history": self.confidence_history,
            "query_patterns": self.query_patterns,
            "first_cycle_seen": self.first_cycle_seen,
            "last_cycle_seen": self.last_cycle_seen,
            "cycle_consistency": self.cycle_consistency,
            "status": self.status,
            "avg_confidence": self._avg_confidence(),
        }
    
    def _avg_confidence(self) -> float:
        """Average confidence across all hits."""
        if not self.confidence_history:
            return 0.0
        return statistics.mean(self.confidence_history)
    
    def _consistency_score(self) -> float:
        """
        Consistency score based on confidence variance.
        Lower variance = higher consistency.
        Returns 0-1 where 1 = perfectly consistent.
        """
        if len(self.confidence_history) < 2:
            return 0.0
        
        try:
            variance = statistics.variance(self.confidence_history)
            # Normalize variance to 0-1 scale (lower variance → higher consistency)
            # Variance typically ranges 0-0.25, so clamp at 0.25
            consistency = 1.0 - min(variance / 0.25, 1.0)
            return max(0.0, consistency)
        except:
            return 0.0


# ============================================================================
# DELTA REGISTRY
# ============================================================================

class DeltaRegistry:
    """
    Tracks query patterns and identifies learning targets (phantoms).
    
    The Delta Registry is the "sensory cortex" - it:
    1. Records every query hit (raw signal)
    2. Identifies persistent patterns (phantoms)
    3. Detects when patterns stabilize (harmonic lock)
    4. Flags facts ready for grain crystallization
    
    Behavior:
    - Transient: 1-4 hits (noise)
    - Persistent: 5+ hits with avg confidence > 0.75
    - Locked: Persistent for 50+ cycles (ready for crystallization)
    """
    
    def __init__(self, cartridge_id: str, persistence_threshold: int = 5,
                 confidence_threshold: float = 0.75,
                 harmonic_lock_cycles: int = 50):
        """
        Initialize Delta Registry.
        
        Args:
            cartridge_id: Which cartridge this registry tracks
            persistence_threshold: Hits needed to become "persistent" (default 5)
            confidence_threshold: Avg confidence needed for persistence (default 0.75)
            harmonic_lock_cycles: Cycles needed for lock (default 50)
        """
        self.cartridge_id = cartridge_id
        self.persistence_threshold = persistence_threshold
        self.confidence_threshold = confidence_threshold
        self.harmonic_lock_cycles = harmonic_lock_cycles
        
        # Raw data
        self.hits: List[QueryHit] = []
        
        # Phantom tracking
        self.phantoms: Dict[int, PhantomCandidate] = {}  # fact_id -> phantom
        
        # Cycle tracking (metabolism)
        self.cycle_count = 0
        self.cycle_history: Dict[int, List[int]] = defaultdict(list)  # fact_id -> [hit_counts per cycle]
        
        # Statistics
        self.total_queries = 0
        self.last_update = datetime.now(timezone.utc).isoformat()
    
    # ========================================================================
    # HIT RECORDING (Immediate)
    # ========================================================================
    
    def record_hit(self, fact_id: int, query_concepts: List[str],
                   confidence: float = 0.8) -> None:
        """
        Record a single query hit (called during query execution).
        
        Updates:
        - Raw hit log
        - Phantom tracking
        - Query pattern tracking
        
        Args:
            fact_id: Which fact was accessed
            query_concepts: Keywords from the query
            confidence: Confidence in the match
        """
        # Record raw hit
        hit = QueryHit(
            fact_id=fact_id,
            cartridge_id=self.cartridge_id,
            query_concepts=query_concepts,
            confidence=confidence,
        )
        self.hits.append(hit)
        self.total_queries += 1
        
        # Update phantom
        if fact_id not in self.phantoms:
            self.phantoms[fact_id] = PhantomCandidate(
                fact_id=fact_id,
                cartridge_id=self.cartridge_id,
                first_cycle_seen=self.cycle_count,
            )
        
        phantom = self.phantoms[fact_id]
        phantom.hit_count += 1
        phantom.confidence_history.append(confidence)
        phantom.last_cycle_seen = self.cycle_count
        
        # Track query pattern (normalized)
        pattern_key = tuple(sorted(query_concepts))
        pattern_str = "|".join(pattern_key)
        phantom.query_patterns[pattern_str] = phantom.query_patterns.get(pattern_str, 0) + 1
        
        # Update status
        self._update_phantom_status(fact_id)
        
        self.last_update = datetime.now(timezone.utc).isoformat()
    
    def _update_phantom_status(self, fact_id: int) -> None:
        """Update phantom status based on current hits and consistency."""
        phantom = self.phantoms[fact_id]
        
        # Check for persistent status
        if len(phantom.confidence_history) >= self.persistence_threshold:
            avg_conf = statistics.mean(phantom.confidence_history)
            if avg_conf >= self.confidence_threshold:
                phantom.status = "persistent"
            else:
                phantom.status = "transient"
        else:
            phantom.status = "transient"
    
    # ========================================================================
    # CYCLE METABOLISM (Per ~100 queries)
    # ========================================================================
    
    def advance_cycle(self) -> None:
        """
        Advance to next metabolic cycle (call every ~100 queries).
        
        Updates:
        - Cycle counter
        - Cycle history (hit counts per phantom)
        - Harmonic lock detection
        """
        self.cycle_count += 1
        
        # STEP 1: Update status for all phantoms BEFORE recording history
        for fact_id, phantom in self.phantoms.items():
            self._update_phantom_status(fact_id)
        
        # STEP 2: Record hit counts for each phantom in this cycle BEFORE resetting
        for fact_id, phantom in self.phantoms.items():
            # Always record to history if the phantom has been hit
            if phantom.hit_count > 0 or phantom.status == "persistent":
                self.cycle_history[fact_id].append(phantom.hit_count)
                
                # Check for harmonic lock (only if persistent)
                if phantom.status == "persistent":
                    self._check_harmonic_lock(fact_id)
        
        # STEP 3: Reset hit counts for next cycle
        for phantom in self.phantoms.values():
            phantom.hit_count = 0
    
    def _check_harmonic_lock(self, fact_id: int) -> None:
        """
        Check if a phantom has achieved harmonic lock.
        
        Harmonic lock = pattern is stable over multiple cycles
        Stability criteria:
        - Been persistent for 50+ cycles
        - Hit counts are consistent (low variance)
        - Confidence is stable (low variance)
        """
        phantom = self.phantoms[fact_id]
        history = self.cycle_history[fact_id]
        
        if len(history) < self.harmonic_lock_cycles:
            return  # Not enough cycles yet
        
        # Check last 50 cycles for consistency
        recent_history = history[-self.harmonic_lock_cycles:]
        
        # Calculate variance in hit counts
        try:
            hit_variance = statistics.variance(recent_history)
            # Normalize: low variance (< 10) = good consistency
            hit_consistency = 1.0 - min(hit_variance / 10.0, 1.0)
        except:
            hit_consistency = 0.0
        
        # Calculate confidence consistency
        confidence_consistency = phantom._consistency_score()
        
        # Harmonic lock = both are consistent
        overall_consistency = (hit_consistency + confidence_consistency) / 2.0
        phantom.cycle_consistency = overall_consistency
        
        # If both are very consistent, lock is achieved
        if hit_consistency > 0.8 and confidence_consistency > 0.8:
            phantom.status = "locked"
    
    # ========================================================================
    # PHANTOM QUERIES
    # ========================================================================
    
    def get_phantom_candidates(self, status: Optional[str] = None) -> List[PhantomCandidate]:
        """
        Get phantom candidates (facts ready for processing).
        
        Args:
            status: Filter by status ("persistent", "locked", or None for all)
            
        Returns:
            List of phantoms, sorted by hit count (highest first)
        """
        candidates = [
            p for p in self.phantoms.values()
            if status is None or p.status == status
        ]
        return sorted(candidates, key=lambda p: p.hit_count, reverse=True)
    
    def get_persistent_phantoms(self) -> List[PhantomCandidate]:
        """Get all persistent phantoms (5+ hits with high confidence)."""
        return self.get_phantom_candidates(status="persistent")
    
    def get_locked_phantoms(self) -> List[PhantomCandidate]:
        """Get all locked phantoms (ready for crystallization)."""
        return self.get_phantom_candidates(status="locked")
    
    def get_phantom(self, fact_id: int) -> Optional[PhantomCandidate]:
        """Get specific phantom by fact_id."""
        return self.phantoms.get(fact_id)
    
    # ========================================================================
    # STATISTICS & ANALYSIS
    # ========================================================================
    
    def get_stats(self) -> Dict:
        """Get registry statistics."""
        persistent = self.get_persistent_phantoms()
        locked = self.get_locked_phantoms()
        
        return {
            "cartridge": self.cartridge_id,
            "cycle_count": self.cycle_count,
            "total_queries": self.total_queries,
            "total_phantoms": len(self.phantoms),
            "persistent_count": len(persistent),
            "locked_count": len(locked),
            "avg_phantom_hits": self._avg_phantom_hits(),
            "avg_phantom_confidence": self._avg_phantom_confidence(),
            "top_phantoms": [p.to_dict() for p in persistent[:5]],
        }
    
    def _avg_phantom_hits(self) -> float:
        """Average hit count across all phantoms."""
        if not self.phantoms:
            return 0.0
        return statistics.mean(p.hit_count for p in self.phantoms.values())
    
    def _avg_phantom_confidence(self) -> float:
        """Average confidence across all phantom hits."""
        if not self.phantoms:
            return 0.0
        
        all_confidences = []
        for p in self.phantoms.values():
            all_confidences.extend(p.confidence_history)
        
        if not all_confidences:
            return 0.0
        return statistics.mean(all_confidences)
    
    # ========================================================================
    # PERSISTENCE
    # ========================================================================
    
    def save(self, filepath: str) -> None:
        """Save registry to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "cartridge_id": self.cartridge_id,
            "cycle_count": self.cycle_count,
            "total_queries": self.total_queries,
            "last_update": self.last_update,
            "phantoms": {
                str(fid): p.to_dict()
                for fid, p in self.phantoms.items()
            },
            "cycle_history": {
                str(fid): history
                for fid, history in self.cycle_history.items()
            },
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> "DeltaRegistry":
        """Load registry from JSON file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Registry file not found: {filepath}")
        
        with open(path) as f:
            data = json.load(f)
        
        # Create new registry
        registry = cls(data["cartridge_id"])
        registry.cycle_count = data.get("cycle_count", 0)
        registry.total_queries = data.get("total_queries", 0)
        registry.last_update = data.get("last_update", "")
        
        # Restore phantoms
        for fact_id_str, phantom_data in data.get("phantoms", {}).items():
            fact_id = int(fact_id_str)
            phantom = PhantomCandidate(
                fact_id=fact_id,
                cartridge_id=phantom_data["cartridge_id"],
                hit_count=phantom_data["hit_count"],
                confidence_history=phantom_data["confidence_history"],
                query_patterns=phantom_data["query_patterns"],
                first_cycle_seen=phantom_data["first_cycle_seen"],
                last_cycle_seen=phantom_data["last_cycle_seen"],
                cycle_consistency=phantom_data["cycle_consistency"],
                status=phantom_data["status"],
            )
            registry.phantoms[fact_id] = phantom
        
        # Restore cycle history
        for fact_id_str, history in data.get("cycle_history", {}).items():
            fact_id = int(fact_id_str)
            registry.cycle_history[fact_id] = history
        
        return registry


# ============================================================================
# METABOLISM COORDINATOR
# ============================================================================

class MetabolismCoordinator:
    """
    Coordinates metabolism across cartridges.
    
    Manages:
    - Cycle advancement across all registries
    - Identification of facts ready for crystallization
    - Statistics aggregation
    """
    
    def __init__(self):
        """Initialize metabolism coordinator."""
        self.registries: Dict[str, DeltaRegistry] = {}
        self.cycle_count = 0
    
    def register(self, registry: DeltaRegistry) -> None:
        """Register a cartridge's Delta Registry."""
        self.registries[registry.cartridge_id] = registry
    
    def advance_cycle(self) -> None:
        """Advance all registries to next cycle."""
        self.cycle_count += 1
        for registry in self.registries.values():
            registry.advance_cycle()
    
    def get_crystallization_candidates(self) -> Dict[str, List[PhantomCandidate]]:
        """
        Get all facts ready for crystallization across all cartridges.
        
        Returns:
            Dict mapping cartridge_id -> list of locked phantoms
        """
        result = {}
        for cart_id, registry in self.registries.items():
            locked = registry.get_locked_phantoms()
            if locked:
                result[cart_id] = locked
        return result
    
    def get_all_stats(self) -> Dict:
        """Get aggregated statistics across all cartridges."""
        stats = {
            "global_cycle_count": self.cycle_count,
            "cartridges": {},
        }
        
        total_phantoms = 0
        total_locked = 0
        
        for cart_id, registry in self.registries.items():
            cart_stats = registry.get_stats()
            stats["cartridges"][cart_id] = cart_stats
            
            total_phantoms += cart_stats["total_phantoms"]
            total_locked += cart_stats["locked_count"]
        
        stats["summary"] = {
            "total_phantoms": total_phantoms,
            "total_locked_and_ready": total_locked,
            "avg_lock_rate": self._avg_lock_rate(),
        }
        
        return stats
    
    def _avg_lock_rate(self) -> float:
        """Percentage of phantoms that are locked."""
        total = sum(r.get_stats()["total_phantoms"] for r in self.registries.values())
        locked = sum(r.get_stats()["locked_count"] for r in self.registries.values())
        
        if total == 0:
            return 0.0
        return locked / total


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("Delta Registry Examples\n")
    
    # Create registry for a cartridge
    registry = DeltaRegistry("bioplastics")
    
    # Simulate some queries
    print("Simulating 100 queries...")
    queries = [
        (1, ["temperature", "pla"], 0.92),
        (2, ["gelling", "pla"], 0.90),
        (3, ["polymer", "synthetic"], 0.85),
        (1, ["temperature", "gelling", "pla"], 0.95),
        (2, ["temperature", "gelling"], 0.92),
    ]
    
    # Run 100 queries (20 cycles of 5 patterns each)
    for cycle in range(20):
        for fact_id, concepts, confidence in queries:
            registry.record_hit(fact_id, concepts, confidence)
        
        registry.advance_cycle()
        print(f"  Cycle {cycle + 1}: {len(registry.get_persistent_phantoms())} persistent")
    
    # Check results
    print("\nResults after 20 cycles:\n")
    
    stats = registry.get_stats()
    print(f"Total queries: {stats['total_queries']}")
    print(f"Phantoms: {stats['total_phantoms']}")
    print(f"Persistent: {stats['persistent_count']}")
    print(f"Locked: {stats['locked_count']}")
    print(f"Avg phantom hits: {stats['avg_phantom_hits']:.1f}")
    print(f"Avg confidence: {stats['avg_phantom_confidence']:.2f}\n")
    
    # Show top phantoms
    print("Top persistent phantoms:")
    for phantom in stats['top_phantoms'][:3]:
        print(f"  Fact {phantom['fact_id']}: "
              f"{phantom['hit_count']} hits, "
              f"{phantom['avg_confidence']:.2f} avg confidence, "
              f"status={phantom['status']}")
    
    # Show locked (crystallization-ready)
    print("\nLocked phantoms (ready for crystallization):")
    for phantom in registry.get_locked_phantoms():
        print(f"  Fact {phantom.fact_id}: "
              f"{phantom.hit_count} recent hits, "
              f"{phantom.cycle_consistency:.2f} consistency")
    
    if not registry.get_locked_phantoms():
        print("  (None yet - need more cycles)")
    
    # Test persistence
    print("\nTesting persistence...")
    registry.save("/tmp/test_registry.json")
    registry2 = DeltaRegistry.load("/tmp/test_registry.json")
    print(f"✓ Loaded: {len(registry2.phantoms)} phantoms")
    print(f"✓ Cycle count: {registry2.cycle_count}")
