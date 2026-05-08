"""
Data Structures - Grain System Enums & Dataclasses

Defines the core data types used throughout the grain crystallization pipeline:
- EpistemicLevel: Knowledge hierarchy (L0-L3)
- GrainState: Grain lifecycle states
- PhantomCandidate: Query pattern data structure
- GrainMetadata: Crystallized grain representation
- TernaryDelta: Ternary relationship representation

Author: Kitbash Team
Date: February 2026
Phase: 2C Consolidation → Modularized (May 2026)
"""

import statistics
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class EpistemicLevel(Enum):
    """Knowledge hierarchy levels from spec"""
    L0_EMPIRICAL = 0    # Physics, logic, math (immutable)
    L1_NARRATIVE = 1    # World facts, history (append-only)
    L2_AXIOMATIC = 2    # Behavioral rules, identity (metabolic)
    L3_PERSONA = 3      # Beliefs, dialogue, noise (ephemeral)


class GrainState(Enum):
    """Grain lifecycle state"""
    CANDIDATE = "candidate"           # Phantom considered for crystallization
    LOCKED = "locked"                 # Harmonic lock detected, ready to crystallize
    CRYSTALLIZED = "crystallized"     # Formally converted to grain
    ACTIVE = "active"                 # Loaded in L3 cache
    ARCHIVED = "archived"             # Stored but not active


# ============================================================================
# DATACLASSES
# ============================================================================

@dataclass
class PhantomCandidate:
    """
    A query pattern that appears persistently.
    Becomes a grain candidate when locked (50+ cycles, high consistency).
    
    From shannon_grain.py
    """
    phantom_id: str                   # Unique identifier
    fact_ids: Set[int]                # Which facts are queried together
    cartridge_id: str                 # Source cartridge
    hit_count: int = 0                # Total hits this cycle
    hit_history: List[int] = field(default_factory=list)  # Hits per cycle
    confidence_scores: List[float] = field(default_factory=list)
    query_concepts: List[str] = field(default_factory=list)
    
    # Cycle tracking
    first_cycle_seen: int = 0
    last_cycle_seen: int = 0
    cycle_consistency: float = 0.0    # Stability metric (0-1)
    
    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Status
    status: str = "transient"         # transient, persistent, locked
    epistemic_level: EpistemicLevel = EpistemicLevel.L2_AXIOMATIC
    
    def avg_confidence(self) -> float:
        """Average confidence across all hits"""
        if not self.confidence_scores:
            return 0.0
        return statistics.mean(self.confidence_scores)
    
    def is_persistent(self, min_hits: int = 5, min_confidence: float = 0.75) -> bool:
        """Check if phantom meets persistence criteria"""
        return (self.hit_count >= min_hits and 
                self.avg_confidence() >= min_confidence)
    
    def is_locked(self, min_cycles: int = 50, min_consistency: float = 0.8) -> bool:
        """Check if phantom has achieved harmonic lock"""
        cycles_stable = len(self.hit_history) >= min_cycles
        consistency_high = self.cycle_consistency >= min_consistency
        return cycles_stable and consistency_high
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "phantom_id": self.phantom_id,
            "fact_ids": list(self.fact_ids),
            "cartridge_id": self.cartridge_id,
            "hit_count": self.hit_count,
            "hit_history": self.hit_history[-50:],  # Last 50 cycles only
            "avg_confidence": self.avg_confidence(),
            "query_concepts": self.query_concepts,
            "cycles_stable": len(self.hit_history),
            "cycle_consistency": round(self.cycle_consistency, 4),
            "status": self.status,
            "epistemic_level": self.epistemic_level.name,
        }


@dataclass
class GrainMetadata:
    """
    A crystallized grain: compressed representation of a persistent pattern.
    Stores ternary weights instead of full embeddings (90% size reduction).
    
    PHASE 5A UPDATE (May 2026):
    - grain_type: Distinguishes axioms (≥95%, immutable) from observations (70-95%, mutable)
    - confidence: Primary confidence metric (replaces avg_confidence conceptually)
    - confidence_mutable: Whether confidence can be updated by Dream Bucket
    
    From shannon_grain.py
    """
    grain_id: str                     # Unique identifier
    source_phantom_id: str            # Which phantom crystallized into this
    cartridge_id: str                 # Source cartridge
    
    # ===== L1/L2 DISTINCTION (NEW - Phase 5A) =====
    grain_type: str = "observation"   # "axiom" (≥95%, immutable) or "observation" (70-95%, mutable)
    confidence: float = 0.0           # 0.0-1.0, primary confidence metric
    confidence_mutable: bool = True   # False for axioms; True for observations
    
    # Ternary representation
    num_bits: int = 256               # Bit-sliced representation size
    bits_positive: int = 0            # Count of +1 weights
    bits_negative: int = 0            # Count of -1 weights
    bits_void: int = 0                # Count of 0 (unset) weights
    
    # Axiom linkage
    axiom_ids: List[str] = field(default_factory=list)
    evidence_hash: str = ""           # SHA-256 of supporting observations
    
    # Quality metrics
    internal_hamming: float = 0.0     # Avg distance between cluster members
    weight_skew: float = 0.0          # Std dev / mean of weights
    avg_confidence: float = 0.0       # DEPRECATED: use confidence instead
    observation_count: int = 0        # How many observations formed this
    
    # Temporal tracking
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    crystallized_at: str = ""
    last_hit: Optional[str] = None    # Last time this grain was activated
    hit_count: int = 0                # Total activations
    
    # Lifecycle
    state: GrainState = GrainState.CANDIDATE
    epistemic_level: EpistemicLevel = EpistemicLevel.L2_AXIOMATIC
    
    # Cache
    bit_array_plus: bytes = b""       # Actual bit array for +1 weights
    bit_array_minus: bytes = b""      # Actual bit array for -1 weights
    
    def __post_init__(self):
        """Validate grain type and confidence relationship"""
        if self.grain_type not in ("axiom", "observation"):
            raise ValueError(f"Invalid grain_type: {self.grain_type}. Must be 'axiom' or 'observation'.")
        
        if self.grain_type == "axiom" and self.confidence < 0.95:
            raise ValueError(f"Axiom grain must have confidence >= 0.95, got {self.confidence}")
        
        if self.grain_type == "observation" and not (0.70 <= self.confidence < 0.95):
            raise ValueError(f"Observation grain must have confidence in [0.70, 0.95), got {self.confidence}")
        
        if self.grain_type == "axiom" and self.confidence_mutable:
            raise ValueError("Axiom grains must have confidence_mutable=False")
        
        # Sync legacy avg_confidence with new confidence field
        if self.avg_confidence == 0.0 and self.confidence > 0.0:
            self.avg_confidence = self.confidence
    
    def to_dict(self) -> dict:
        """Convert to dictionary with L1/L2 stratification info"""
        return {
            "grain_id": self.grain_id,
            "source_phantom_id": self.source_phantom_id,
            "cartridge_id": self.cartridge_id,
            # ===== L1/L2 INFO (NEW) =====
            "grain_type": self.grain_type,
            "confidence": round(self.confidence, 3),
            "confidence_mutable": self.confidence_mutable,
            # ===== TERNARY WEIGHTS =====
            "num_bits": self.num_bits,
            "weight_distribution": {
                "positive": self.bits_positive,
                "negative": self.bits_negative,
                "void": self.bits_void,
            },
            "axiom_ids": self.axiom_ids,
            "quality_metrics": {
                "internal_hamming": round(self.internal_hamming, 3),
                "weight_skew": round(self.weight_skew, 3),
                "avg_confidence": round(self.avg_confidence, 3),
                "observation_count": self.observation_count,
            },
            "state": self.state.value,
            "epistemic_level": self.epistemic_level.name,
            "size_bytes": len(self.bit_array_plus) + len(self.bit_array_minus),
        }
    
    def size_mb(self) -> float:
        """Size in megabytes"""
        return (len(self.bit_array_plus) + len(self.bit_array_minus)) / (1024 * 1024)


@dataclass
class TernaryDelta:
    """
    Ternary relationship representation.
    
    From ternary_crush.py
    """
    positive: List[str]  # Dependencies (+1): what this depends on
    negative: List[str]  # Negations (-1): what contradicts this
    void: List[str]      # Independence (0): what's orthogonal
