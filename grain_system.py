"""
Grain System - Unified Crystallization Pipeline
Consolidates all Phase 2C grain functionality into single module

Sections:
1. Enums & Data Structures (100 lines) - PhantomCandidate, GrainMetadata, TernaryDelta
2. Phantom Tracking (150 lines) - PhantomTracker, cycle management, lock detection
3. Grain Registry (100 lines) - Grain storage, indexing, lifecycle
4. Compression (TernaryCrush) (200 lines) - Ternary representation, weight encoding
5. Persistence (GrainCrystallizer) (150 lines) - Grain saving/loading, manifest updates
6. Orchestration (ShannonGrainOrchestrator) (200+ lines) - Full pipeline coordination

Author: Kitbash Team
Date: February 2026
Phase: 2C Consolidation
"""

import json
import time
import hashlib
import statistics
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from enum import Enum

# External dependencies
from kitbash_cartridge import Cartridge
from axiom_validator import AxiomValidator, ValidationRule


# ============================================================================
# SECTION 1: ENUMS & DATA STRUCTURES
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
    
    From shannon_grain.py
    """
    grain_id: str                     # Unique identifier
    source_phantom_id: str            # Which phantom crystallized into this
    cartridge_id: str                 # Source cartridge
    
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
    avg_confidence: float = 0.0       # Avg confidence of source observations
    observation_count: int = 0        # How many observations formed this
    
    # Lifecycle
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    crystallized_at: str = ""
    state: GrainState = GrainState.CANDIDATE
    epistemic_level: EpistemicLevel = EpistemicLevel.L2_AXIOMATIC
    
    # Cache
    bit_array_plus: bytes = b""       # Actual bit array for +1 weights
    bit_array_minus: bytes = b""      # Actual bit array for -1 weights
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "grain_id": self.grain_id,
            "source_phantom_id": self.source_phantom_id,
            "cartridge_id": self.cartridge_id,
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


# ============================================================================
# SECTION 2: PHANTOM TRACKING
# ============================================================================

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


# ============================================================================
# SECTION 3: GRAIN REGISTRY
# ============================================================================

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
        """Get registry statistics"""
        return {
            "cartridge_id": self.cartridge_id,
            "total_grains": len(self.grains),
            "by_state": {
                state.value: len(grain_ids)
                for state, grain_ids in self.grain_state_index.items()
            },
            "total_storage_mb": sum(g.size_mb() for g in self.grains.values()),
            "avg_grain_size_bytes": statistics.mean([len(g.bit_array_plus) + len(g.bit_array_minus) 
                                                      for g in self.grains.values()]) if self.grains else 0,
        }


# ============================================================================
# SECTION 4: COMPRESSION (TernaryCrush)
# ============================================================================

class TernaryCrush:
    """
    Compress validated phantoms to ternary grain representation.
    
    Strategy:
    - Extract top 5 derivations from fact annotations
    - Map to ternary relationships {-1, 0, 1}
    - Build pointer map for O(1) lookup
    - Calculate 1.58-bit weight encoding
    
    From ternary_crush.py
    """
    
    def __init__(self, cartridge: Cartridge):
        """Initialize crusher for a specific cartridge."""
        self.cartridge = cartridge
        self.cartridge_id = cartridge.name
    
    def crush_phantom(self, phantom: PhantomCandidate,
                     validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Crush a validated phantom into ternary grain.
        
        Args:
            phantom: PhantomCandidate from registry
            validation_result: Result from AxiomValidator
        
        Returns:
            Ternary grain structure ready for crystallization
        """
        
        if not validation_result.get('locked'):
            raise ValueError(f"Cannot crush unvalidated phantom {phantom.phantom_id}")
        
        # Get fact content
        fact_text = self.cartridge.get_fact(list(phantom.fact_ids)[0])
        if not fact_text:
            raise ValueError(f"Fact not found in cartridge")
        
        # Extract derivations
        try:
            fact_obj = self.cartridge.get_fact_object(list(phantom.fact_ids)[0])
            derivations = fact_obj.derivations if fact_obj else []
        except:
            derivations = []
        
        # Map to ternary
        ternary_delta = self._extract_ternary_delta(fact_text, derivations)
        
        # Build pointer map (fast lookup structure)
        pointer_map = self._build_pointer_map(phantom, ternary_delta)
        
        # Calculate weight (1.58-bit equivalent)
        weight = self._calculate_weight(ternary_delta)
        
        # Generate grain ID (deterministic from fact)
        grain_id = self._generate_grain_id(list(phantom.fact_ids)[0], self.cartridge_id)
        
        return {
            'grain_id': grain_id,
            'fact_ids': list(phantom.fact_ids),
            'cartridge_id': self.cartridge_id,
            'delta': {
                'positive': ternary_delta.positive,
                'negative': ternary_delta.negative,
                'void': ternary_delta.void,
            },
            'weight': weight,
            'pointer_map': pointer_map,
            'confidence': validation_result['confidence'],
            'cycles_locked': validation_result['cycles_locked'],
            'fact_snippet': fact_text[:100],
        }
    
    def _extract_ternary_delta(self, fact_text: str,
                              derivations: List[Any]) -> TernaryDelta:
        """Extract ternary relationships from fact and derivations."""
        
        positive = []
        negative = []
        void = []
        
        # Extract from derivations (structured)
        for deriv in derivations:
            if not deriv:
                continue
            
            deriv_str = str(deriv).lower()
            deriv_type = deriv.get('type', '') if isinstance(deriv, dict) else ''
            target = deriv.get('target', '') if isinstance(deriv, dict) else ''
            
            # Classify by type
            if 'dependency' in deriv_type or 'requires' in deriv_type:
                if target:
                    positive.append(target)
            elif 'negation' in deriv_type or 'inverse' in deriv_type:
                if target:
                    negative.append(target)
            elif 'independent' in deriv_type or 'orthogonal' in deriv_type:
                if target:
                    void.append(target)
            elif 'boundary' in deriv_type:
                if target:
                    negative.append(f"constrained_by:{target}")
        
        # Extract from fact text (unstructured) - keyword heuristics
        fact_lower = fact_text.lower()
        
        dep_keywords = ['requires', 'depends on', 'needs', 'causes', 'leads to', 
                       'enables', 'triggers', 'necessary for', 'sufficient for']
        for kw in dep_keywords:
            if kw in fact_lower:
                idx = fact_lower.find(kw)
                snippet = fact_text[idx:idx+50].strip()
                if snippet not in positive:
                    positive.append(f"inferred:{snippet[:30]}")
        
        neg_keywords = ['not', 'cannot', 'opposite', 'contradicts', 'conflicts',
                       'incompatible', 'prevents', 'blocks', 'inhibits', 'never']
        for kw in neg_keywords:
            if kw in fact_lower:
                idx = fact_lower.find(kw)
                snippet = fact_text[idx:idx+50].strip()
                if snippet not in negative:
                    negative.append(f"inferred:{snippet[:30]}")
        
        indep_keywords = ['independent', 'orthogonal', 'unrelated', 'separate',
                         'parallel', 'distinct', 'isolated']
        for kw in indep_keywords:
            if kw in fact_lower:
                idx = fact_lower.find(kw)
                snippet = fact_text[idx:idx+50].strip()
                if snippet not in void:
                    void.append(f"inferred:{snippet[:30]}")
        
        # Limit to top N per category
        positive = self._rank_and_limit(positive, 3)
        negative = self._rank_and_limit(negative, 2)
        void = self._rank_and_limit(void, 2)
        
        return TernaryDelta(
            positive=positive,
            negative=negative,
            void=void
        )
    
    def _rank_and_limit(self, items: List[str], limit: int) -> List[str]:
        """Rank by specificity and limit to top N."""
        if not items:
            return []
        
        unique = list(dict.fromkeys(items))
        unique = sorted(unique, key=len, reverse=True)
        return unique[:limit]
    
    def _build_pointer_map(self, phantom: PhantomCandidate,
                          ternary: TernaryDelta) -> Dict[str, Any]:
        """Build pointer map for O(1) relationship lookup."""
        
        pointer_map = {
            'positive_ptrs': {},
            'negative_ptrs': {},
            'void_ptrs': {},
            'access_pattern': {
                'hit_count': phantom.hit_count,
                'confidence': phantom.avg_confidence(),
                'first_seen': phantom.first_cycle_seen,
                'last_seen': phantom.last_cycle_seen,
            }
        }
        
        bit_pos = 0
        
        for concept in ternary.positive:
            pointer_map['positive_ptrs'][concept] = {
                'bit_position': bit_pos,
                'value': 1,
            }
            bit_pos += 1
        
        for concept in ternary.negative:
            pointer_map['negative_ptrs'][concept] = {
                'bit_position': bit_pos,
                'value': -1,
            }
            bit_pos += 1
        
        for concept in ternary.void:
            pointer_map['void_ptrs'][concept] = {
                'bit_position': bit_pos,
                'value': 0,
            }
            bit_pos += 1
        
        pointer_map['total_bits'] = bit_pos
        
        return pointer_map
    
    def _calculate_weight(self, ternary: TernaryDelta) -> float:
        """Calculate 1.58-bit weight encoding."""
        
        total_concepts = (len(ternary.positive) + 
                         len(ternary.negative) + 
                         len(ternary.void))
        
        # 1 ternary position = log2(3) ≈ 1.585 bits
        weight = total_concepts * 1.585
        
        return round(weight, 2)
    
    def _generate_grain_id(self, fact_id: int, cartridge_id: str) -> str:
        """Generate deterministic grain ID from fact identity."""
        
        hash_input = f"{cartridge_id}:{fact_id}".encode()
        hash_obj = hashlib.sha256(hash_input)
        hex_hash = hash_obj.hexdigest()[:8]
        
        return f"sg_{hex_hash.upper()}"
    
    def crush_all_phantoms(self, 
                          phantoms: Dict[str, PhantomCandidate],
                          validation_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Crush all validated phantoms to ternary grains."""
        
        grains = []
        
        for phantom_id, phantom in phantoms.items():
            val_result = validation_results.get(phantom_id)
            
            if val_result and val_result.get('locked'):
                try:
                    grain = self.crush_phantom(phantom, val_result)
                    grains.append(grain)
                except Exception as e:
                    print(f"Warning: Could not crush phantom {phantom_id}: {e}")
        
        return grains


# ============================================================================
# SECTION 5: PERSISTENCE (GrainCrystallizer)
# ============================================================================

class GrainCrystallizer:
    """
    Persists crushed ternary grains to cartridge storage.
    
    Responsibilities:
    - Save grain JSON files to cartridges/{name}/grains/
    - Update cartridge manifest with grain_inventory
    - Track crystallization metadata
    - Enable grain loading for activation layer
    
    From grain_crystallizer.py
    """
    
    def __init__(self, cartridges_dir: str = "./cartridges"):
        """Initialize crystallizer."""
        self.cartridges_dir = Path(cartridges_dir)
    
    def crystallize_grains(self, crushed_grains: List[Dict[str, Any]],
                          cartridge_id: str) -> Dict[str, Any]:
        """
        Save crushed grains for a specific cartridge.
        
        Args:
            crushed_grains: List of ternary grain dicts from TernaryCrush
            cartridge_id: Name of cartridge these grains belong to
        
        Returns:
            Crystallization result with file paths and statistics
        """
        
        cartridge_path = self.cartridges_dir / f"{cartridge_id}.kbc"
        grains_dir = cartridge_path / "grains"
        
        grains_dir.mkdir(parents=True, exist_ok=True)
        
        result = {
            'cartridge_id': cartridge_id,
            'grain_count': 0,
            'grain_files': [],
            'manifest_updated': False,
            'errors': [],
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }
        
        # Save each grain
        for grain in crushed_grains:
            try:
                grain_file = self._save_grain_file(grain, grains_dir)
                result['grain_files'].append(grain_file)
                result['grain_count'] += 1
            except Exception as e:
                result['errors'].append(f"Grain {grain.get('grain_id')}: {str(e)}")
        
        # Update manifest
        try:
            self._update_manifest(grains_dir.parent, crushed_grains, result)
            result['manifest_updated'] = True
        except Exception as e:
            result['errors'].append(f"Manifest update: {str(e)}")
        
        return result
    
    def _save_grain_file(self, grain: Dict[str, Any], grains_dir: Path) -> str:
        """Save individual grain to JSON file."""
        
        grain_id = grain.get('grain_id')
        if not grain_id:
            raise ValueError("Grain missing grain_id")
        
        grain_json = {
            'grain_id': grain['grain_id'],
            'fact_id': grain['fact_ids'][0] if grain['fact_ids'] else None,
            'fact_ids': grain['fact_ids'],
            'cartridge_source': grain['cartridge_id'],
            'axiom_link': 'domain_concept',
            'lock_state': 'Sicherman_Validated',
            'weight': grain['weight'],
            'delta': grain['delta'],
            'confidence': grain['confidence'],
            'cycles_locked': grain['cycles_locked'],
            'validation_timestamp': datetime.now(timezone.utc).isoformat(),
            'pointer_map': grain['pointer_map'],
        }
        
        grain_file = grains_dir / f"{grain_id}.json"
        with open(grain_file, 'w') as f:
            json.dump(grain_json, f, indent=2)
        
        return str(grain_file.relative_to(self.cartridges_dir))
    
    def _update_manifest(self, cartridge_path: Path, 
                        crushed_grains: List[Dict], 
                        result: Dict) -> None:
        """Update cartridge manifest with grain inventory."""
        
        manifest_file = cartridge_path / "manifest.json"
        
        if manifest_file.exists():
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
        else:
            manifest = {
                'cartridge_id': cartridge_path.name.replace('.kbc', ''),
                'created': datetime.now(timezone.utc).isoformat(),
            }
        
        if 'grain_inventory' not in manifest:
            manifest['grain_inventory'] = {}
        
        for grain in crushed_grains:
            grain_id = grain['grain_id']
            manifest['grain_inventory'][grain_id] = {
                'fact_ids': grain['fact_ids'],
                'axiom_link': 'domain_concept',
                'lock_state': 'Sicherman_Validated',
                'confidence': grain['confidence'],
                'weight': grain['weight'],
                'crystallization_timestamp': datetime.now(timezone.utc).isoformat(),
            }
        
        manifest['grain_crystallization'] = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'grain_count': len(manifest['grain_inventory']),
            'phase': '2C_consolidated',
        }
        
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def load_grain(self, cartridge_id: str, grain_id: str) -> Optional[Dict[str, Any]]:
        """Load a crystallized grain from disk."""
        
        grain_file = self.cartridges_dir / f"{cartridge_id}.kbc" / "grains" / f"{grain_id}.json"
        
        if not grain_file.exists():
            return None
        
        with open(grain_file, 'r') as f:
            return json.load(f)
    
    def load_all_grains(self, cartridge_id: str) -> Dict[str, Dict[str, Any]]:
        """Load all crystallized grains for a cartridge."""
        
        grains_dir = self.cartridges_dir / f"{cartridge_id}.kbc" / "grains"
        
        if not grains_dir.exists():
            return {}
        
        grains = {}
        for grain_file in grains_dir.glob("*.json"):
            try:
                with open(grain_file, 'r') as f:
                    grain = json.load(f)
                    grain_id = grain.get('grain_id')
                    if grain_id:
                        grains[grain_id] = grain
            except Exception as e:
                print(f"Warning: Could not load grain {grain_file}: {e}")
        
        return grains
    
    def get_manifests(self) -> Dict[str, Dict[str, Any]]:
        """Load all cartridge manifests with grain inventory."""
        
        manifests = {}
        
        for cartridge_dir in self.cartridges_dir.glob("*.kbc"):
            manifest_file = cartridge_dir / "manifest.json"
            if manifest_file.exists():
                try:
                    with open(manifest_file, 'r') as f:
                        manifest = json.load(f)
                        cartridge_id = cartridge_dir.name.replace('.kbc', '')
                        manifests[cartridge_id] = manifest
                except Exception as e:
                    print(f"Warning: Could not load manifest {manifest_file}: {e}")
        
        return manifests


class GrainCrystallizationReport:
    """Generate and save crystallization summary report."""
    
    def __init__(self, cartridges_dir: str = "./cartridges"):
        self.cartridges_dir = Path(cartridges_dir)
        self.report = {
            'phase': '2C_consolidated',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'cartridges': {},
            'summary': {
                'total_grains': 0,
                'total_cartridges': 0,
                'total_size': 0,
            }
        }
    
    def add_crystallization_result(self, result: Dict[str, Any]) -> None:
        """Add a crystallization result to the report."""
        
        cartridge_id = result['cartridge_id']
        self.report['cartridges'][cartridge_id] = {
            'grain_count': result['grain_count'],
            'grain_files': result['grain_files'],
            'manifest_updated': result['manifest_updated'],
            'errors': result['errors'],
            'timestamp': result['timestamp'],
        }
        
        self.report['summary']['total_grains'] += result['grain_count']
        self.report['summary']['total_cartridges'] += 1
    
    def calculate_sizes(self) -> None:
        """Calculate total crystallized grain size."""
        
        total_size = 0
        
        for cartridge_id in self.report['cartridges']:
            grain_files = self.report['cartridges'][cartridge_id]['grain_files']
            for grain_file in grain_files:
                filepath = self.cartridges_dir / grain_file
                if filepath.exists():
                    total_size += filepath.stat().st_size
        
        self.report['summary']['total_size'] = total_size
    
    def save(self, output_file: str = "./phase2c_crystallization_report.json") -> None:
        """Save report to JSON file."""
        
        self.calculate_sizes()
        
        with open(output_file, 'w') as f:
            json.dump(self.report, f, indent=2)
    
    def print_summary(self) -> None:
        """Print report summary."""
        
        self.calculate_sizes()
        
        print("\n" + "="*70)
        print("GRAIN CRYSTALLIZATION REPORT")
        print("="*70)
        print(f"Timestamp: {self.report['timestamp']}")
        print(f"Cartridges processed: {self.report['summary']['total_cartridges']}")
        print(f"Total grains crystallized: {self.report['summary']['total_grains']}")
        print(f"Total grain storage: {self.report['summary']['total_size']:,} bytes")
        
        print("\nPer-cartridge breakdown:")
        for cart_id, cart_report in self.report['cartridges'].items():
            print(f"  {cart_id}:")
            print(f"    - Grains: {cart_report['grain_count']}")
            print(f"    - Manifest updated: {cart_report['manifest_updated']}")
            if cart_report['errors']:
                print(f"    - Errors: {len(cart_report['errors'])}")
        
        print("="*70 + "\n")


# ============================================================================
# SECTION 6: ORCHESTRATION (ShannonGrainOrchestrator)
# ============================================================================

class ShannonGrainOrchestrator:
    """
    End-to-end orchestration of grain crystallization.
    
    Pipeline:
    1. Track phantoms (persistent patterns)
    2. Detect harmonic lock (50+ cycles stable)
    3. Validate with axiom rules
    4. Crystallize to grains
    5. Activate in L3 cache
    
    From shannon_grain_orchestrator.py
    """
    
    def __init__(self, cartridge_id: str, storage_path: str = "./grains"):
        """Initialize orchestrator"""
        self.cartridge_id = cartridge_id
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Core components
        self.phantom_tracker = PhantomTracker(cartridge_id)
        self.axiom_validator = AxiomValidator()
        self.grain_registry = GrainRegistry(cartridge_id, str(self.storage_path))
        
        # Note: TernaryCrush requires Cartridge, which will be provided at crystallize time
        self.ternary_crusher = None
        
        # Statistics
        self.crystallization_stats = {
            "total_phantoms_tracked": 0,
            "total_phantoms_locked": 0,
            "total_validated": 0,
            "total_crystallized": 0,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
    
    def record_phantom_hit(self, fact_ids: Set[int], concepts: List[str],
                          confidence: float) -> None:
        """
        Record a phantom hit (called during query execution).
        
        This integrates with DeltaRegistry - when facts are accessed together
        with high confidence, they form phantoms.
        """
        self.phantom_tracker.record_phantom_hit(
            fact_ids=fact_ids,
            concepts=concepts,
            confidence=confidence
        )
        self.crystallization_stats["total_phantoms_tracked"] += 1
    
    def advance_cycle(self) -> None:
        """Advance metabolic cycle (call every ~100 queries)."""
        self.phantom_tracker.advance_cycle()
    
    def get_crystallization_candidates(self) -> List[PhantomCandidate]:
        """
        Get phantoms ready for crystallization.
        
        Returns:
            Locked phantoms (50+ cycles, high consistency)
        """
        locked = self.phantom_tracker.get_locked_phantoms()
        self.crystallization_stats["total_phantoms_locked"] += len(locked)
        return locked
    
    def validate_candidates(self, phantoms: List[PhantomCandidate]) -> Dict:
        """
        Validate phantoms against axiom rules.
        
        Returns:
            Validation report with ready-to-crystallize list
        """
        existing_grains = list(self.grain_registry.grains.values())
        
        validation_report = self.axiom_validator.validate_batch(
            phantoms,
            existing_grains=existing_grains
        )
        
        self.crystallization_stats["total_validated"] += len(phantoms)
        
        return validation_report
    
    def crystallize_grains(self, validated_phantoms: List[Dict], 
                          cartridge: Cartridge) -> List[GrainMetadata]:
        """
        Crystallize validated phantoms into grains.
        
        Args:
            validated_phantoms: List of {phantom_id, fact_ids, confidence}
            cartridge: Cartridge instance for fact extraction
        
        Returns:
            List of created GrainMetadata objects
        """
        # Initialize crusher with provided cartridge
        self.ternary_crusher = TernaryCrush(cartridge)
        
        grains = []
        
        for phantom_info in validated_phantoms:
            # Retrieve phantom
            phantom_key = phantom_info.get("phantom_id")
            phantom = self.phantom_tracker.phantoms.get(phantom_key)
            
            if not phantom:
                continue
            
            # Crush to ternary grain
            try:
                grain_dict = self.ternary_crusher.crush_phantom(phantom, phantom_info)
                
                # Convert to GrainMetadata
                grain = GrainMetadata(
                    grain_id=grain_dict['grain_id'],
                    source_phantom_id=phantom.phantom_id,
                    cartridge_id=self.cartridge_id,
                    state=GrainState.CRYSTALLIZED,
                    crystallized_at=datetime.now(timezone.utc).isoformat(),
                    avg_confidence=phantom_info.get('confidence', 0.0),
                    observation_count=phantom.hit_count,
                )
                
                self.grain_registry.add_grain(grain)
                self.grain_registry.save_grain(grain)
                grains.append(grain)
                self.crystallization_stats["total_crystallized"] += 1
            
            except Exception as e:
                print(f"Warning: Could not crystallize phantom {phantom_key}: {e}")
        
        return grains
    
    def run_crystallization_cycle(self, cartridge: Cartridge) -> Dict:
        """
        Run complete crystallization cycle.
        
        Steps:
        1. Get locked phantoms
        2. Validate with axiom rules
        3. Crystallize to grains
        4. (Activation handled by caller)
        
        Args:
            cartridge: Cartridge instance for crushing
        
        Returns:
            Complete report
        """
        cycle_start = time.time()
        
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phase_1_phantoms": {},
            "phase_2_validation": {},
            "phase_3_crystallization": {},
            "total_latency_ms": 0,
        }
        
        # Phase 1: Get candidates
        t1_start = time.time()
        candidates = self.get_crystallization_candidates()
        report["phase_1_phantoms"] = {
            "locked_count": len(candidates),
            "latency_ms": round((time.time() - t1_start) * 1000, 2),
        }
        
        if not candidates:
            report["verdict"] = "No phantoms ready for crystallization"
            return report
        
        # Phase 2: Validate
        t2_start = time.time()
        validation_report = self.validate_candidates(candidates)
        ready_to_crystallize = validation_report.get("crystallization_ready", [])
        report["phase_2_validation"] = {
            "total_validated": len(candidates),
            "passed_all_rules": len(ready_to_crystallize),
            "latency_ms": round((time.time() - t2_start) * 1000, 2),
        }
        
        if not ready_to_crystallize:
            report["phase_3_crystallization"] = {"grains_created": 0}
            report["verdict"] = "All candidates rejected by validation"
            return report
        
        # Phase 3: Crystallize
        t3_start = time.time()
        crystallized_grains = self.crystallize_grains(ready_to_crystallize, cartridge)
        report["phase_3_crystallization"] = {
            "grains_created": len(crystallized_grains),
            "total_size_mb": sum(g.size_mb() for g in crystallized_grains),
            "latency_ms": round((time.time() - t3_start) * 1000, 2),
        }
        
        total_latency = time.time() - cycle_start
        report["total_latency_ms"] = round(total_latency * 1000, 2)
        
        if crystallized_grains:
            report["verdict"] = f"✅ CRYSTALLIZATION SUCCESS - {len(crystallized_grains)} grains created"
        else:
            report["verdict"] = "❌ CRYSTALLIZATION FAILED"
        
        return report
    
    def get_stats(self) -> Dict:
        """Get orchestrator statistics"""
        stats = self.crystallization_stats.copy()
        stats.update({
            "phantom_tracker": self.phantom_tracker.get_stats(),
            "grain_registry": self.grain_registry.get_stats(),
        })
        return stats


# ============================================================================
# EXPORTS & MAIN
# ============================================================================

if __name__ == "__main__":
    print("Grain System - Consolidated Phase 2C Module")
    print("=" * 70)
    print("\nAvailable classes:")
    print("  - PhantomTracker (phantom tracking, harmonic lock detection)")
    print("  - GrainRegistry (grain storage, lifecycle management)")
    print("  - TernaryCrush (phantom→grain compression)")
    print("  - GrainCrystallizer (persistence layer)")
    print("  - ShannonGrainOrchestrator (end-to-end pipeline)")
    print("\n" + "=" * 70)
