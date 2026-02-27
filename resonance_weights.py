"""
kitbash/memory/resonance_weights.py

Tier 5 Resonance Weight Service
--------------------------------
Manages ephemeral, session-scoped pattern weights using Ebbinghaus-style
decay with stability accumulation.

Decay formula:
    weight = e^(-age / S)

Where:
    age = current_turn - last_reinforced_turn  (resets to 0 on reinforcement)
    S   = stability (starts at initial_stability, grows on each reinforcement)

On reinforcement (base mode):
    S *= stability_growth
    last_reinforced_turn = current_turn
    hit_count += 1

On reinforcement (spacing-sensitive mode):
    spacing_bonus = 1.0 + (1.0 - current_weight)
    S *= stability_growth * spacing_bonus
    last_reinforced_turn = current_turn
    hit_count += 1

Turn definition: one completed query resolution (advance_turn() is called by
the QueryOrchestrator after a final answer is returned, not on every message).

Promotion signal: patterns with hit_count >= 3 are candidates for promotion
to Tier 1.5 Echo (handled by the metabolism layer, not this service).
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Tunable constants (defaults)
# ---------------------------------------------------------------------------

DEFAULT_INITIAL_STABILITY: float = 3.0
"""
How long (in turns) an unreinforced pattern survives before hitting cleanup.
With S=3.0, weight drops below 0.001 at roughly age=21 turns.

Derivation: e^(-age/3.0) < 0.001  →  age > 3.0 * ln(1000) ≈ 20.7 turns
"""

DEFAULT_STABILITY_GROWTH: float = 2.0
"""
Multiplier applied to S on each reinforcement (base mode).

Effect on survival turns (unreinforced after each reinforcement):
  0 reinforcements: ~21 turns
  1 reinforcement:  ~41 turns
  2 reinforcements: ~83 turns
  3 reinforcements: ~166 turns  ← well past typical session, promotion candidate
"""

DEFAULT_CLEANUP_THRESHOLD: float = 0.001
"""
Patterns with weight below this are removed from RAM on advance_turn().
"""

DEFAULT_SPACING_SENSITIVE: bool = False
"""
When True, reinforcement bonus scales with how much the pattern had decayed.
A pattern reinforced while nearly faded gets a larger S boost than one
reinforced while still hot. Rewards topics that resurface after absence.
"""

PROMOTION_HIT_COUNT: int = 3
"""
hit_count threshold above which a pattern is considered a promotion candidate
for Tier 1.5 Echo. Checked externally by the metabolism layer.
"""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ResonanceWeight:
    """
    Ephemeral weight record for a single query pattern.

    Fields
    ------
    pattern_hash : str
        SHA-256 hex digest of the query pattern (or equivalent identifier).
    stability : float
        Current stability S. Starts at initial_stability, grows on each
        reinforcement. Controls how slowly the pattern decays.
    created_turn : int
        Turn on which this pattern was first recorded.
    last_reinforced_turn : int
        Turn of the most recent reinforcement (or creation if never reinforced).
        This is the anchor for age calculation - age = current_turn - last_reinforced_turn.
    hit_count : int
        Number of times this pattern has been reinforced since creation.
    metadata : dict
        Arbitrary caller-supplied data (query text, entities, source cartridge, etc.).
    """
    pattern_hash: str
    stability: float
    created_turn: int
    last_reinforced_turn: int
    hit_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class ResonanceWeightService:
    """
    Manage ephemeral Tier 5 resonance weights for the current session.

    All storage is RAM-only. Weights do not persist across sessions.
    The service is session-scoped: create one instance per conversation.

    Parameters
    ----------
    initial_stability : float
        S₀ - starting stability for new patterns.
        Default: DEFAULT_INITIAL_STABILITY (3.0)
    stability_growth : float
        Multiplier applied to S on each reinforcement.
        Default: DEFAULT_STABILITY_GROWTH (2.0)
    cleanup_threshold : float
        Patterns with weight below this are pruned on advance_turn().
        Default: DEFAULT_CLEANUP_THRESHOLD (0.001)
    spacing_sensitive : bool
        If True, reinforcement bonus scales with decay at time of hit.
        Default: DEFAULT_SPACING_SENSITIVE (False)
    """

    def __init__(
        self,
        initial_stability: float = DEFAULT_INITIAL_STABILITY,
        stability_growth: float = DEFAULT_STABILITY_GROWTH,
        cleanup_threshold: float = DEFAULT_CLEANUP_THRESHOLD,
        spacing_sensitive: bool = DEFAULT_SPACING_SENSITIVE,
    ) -> None:
        self.initial_stability = initial_stability
        self.stability_growth = stability_growth
        self.cleanup_threshold = cleanup_threshold
        self.spacing_sensitive = spacing_sensitive

        self.weights: Dict[str, ResonanceWeight] = {}
        self.current_turn: int = 0

    # -----------------------------------------------------------------------
    # Core weight computation
    # -----------------------------------------------------------------------

    def compute_weight(self, pattern_hash: str) -> float:
        """
        Return the current weight for a pattern, or 0.0 if not found.

        Formula: weight = e^(-age / S)
        Where age = current_turn - last_reinforced_turn

        Age of 0 (just reinforced or just created) → weight = 1.0
        Age grows each turn until weight drops below cleanup_threshold.
        """
        if pattern_hash not in self.weights:
            return 0.0

        w = self.weights[pattern_hash]
        age = self.current_turn - w.last_reinforced_turn
        return math.exp(-age / w.stability)

    # -----------------------------------------------------------------------
    # Pattern lifecycle
    # -----------------------------------------------------------------------

    def record_pattern(
        self,
        pattern_hash: str,
        metadata: Optional[Dict[str, Any]] = None,
        initial_stability: Optional[float] = None,
    ) -> None:
        """
        Record a new pattern. Idempotent - if the hash already exists, does nothing.

        Parameters
        ----------
        pattern_hash : str
            Identifier for this pattern (e.g. SHA-256 of query text).
        metadata : dict, optional
            Arbitrary data to store alongside the weight.
        initial_stability : float, optional
            Override the service default S₀ for this specific pattern.
            Use this for variable starting confidence - e.g. an inferred
            pattern might start with lower stability than an explicit one.
        """
        if pattern_hash in self.weights:
            return

        s0 = initial_stability if initial_stability is not None else self.initial_stability

        self.weights[pattern_hash] = ResonanceWeight(
            pattern_hash=pattern_hash,
            stability=s0,
            created_turn=self.current_turn,
            last_reinforced_turn=self.current_turn,
            hit_count=0,
            metadata=metadata or {},
        )

    def reinforce_pattern(self, pattern_hash: str) -> None:
        """
        Reinforce a pattern: grow stability, reset age anchor, increment hit_count.

        Base mode:
            S *= stability_growth
            last_reinforced_turn = current_turn
            hit_count += 1

        Spacing-sensitive mode:
            Compute weight just before reinforcement.
            spacing_bonus = 1.0 + (1.0 - current_weight)
              → Pattern at weight ~0.0 gets bonus ≈ 2.0 (max)
              → Pattern at weight ~1.0 gets bonus ≈ 1.0 (no extra)
            S *= stability_growth * spacing_bonus
            last_reinforced_turn = current_turn
            hit_count += 1

        If pattern_hash is not found, does nothing (silent no-op).
        """
        if pattern_hash not in self.weights:
            return

        w = self.weights[pattern_hash]

        if self.spacing_sensitive:
            current_weight = self.compute_weight(pattern_hash)
            spacing_bonus = 1.0 + (1.0 - current_weight)
            w.stability *= self.stability_growth * spacing_bonus
        else:
            w.stability *= self.stability_growth

        w.last_reinforced_turn = self.current_turn
        w.hit_count += 1

    # -----------------------------------------------------------------------
    # Turn advancement
    # -----------------------------------------------------------------------

    def advance_turn(self) -> None:
        """
        Advance the session turn counter and prune low-weight patterns.

        Called by QueryOrchestrator after a completed query resolution
        (i.e. after a final answer is returned to the user - not on every
        message exchange during clarification).

        Cleanup removes patterns with weight < cleanup_threshold.
        """
        self.current_turn += 1

        to_remove = [
            ph for ph in self.weights
            if self.compute_weight(ph) < self.cleanup_threshold
        ]
        for ph in to_remove:
            del self.weights[ph]

    # -----------------------------------------------------------------------
    # Queries
    # -----------------------------------------------------------------------

    def get_active_patterns(self, threshold: float = 0.3) -> List[str]:
        """
        Return pattern hashes with weight >= threshold.

        Used by QueryOrchestrator and context re-injection logic to find
        patterns worth surfacing in the current turn.

        Default threshold of 0.3 roughly corresponds to:
          age = S * ln(1/0.3) ≈ S * 1.2 turns since last reinforcement
        """
        return [
            ph for ph in self.weights
            if self.compute_weight(ph) >= threshold
        ]

    def get_promotion_candidates(self) -> List[str]:
        """
        Return pattern hashes that have hit the promotion threshold.

        A promotion candidate has hit_count >= PROMOTION_HIT_COUNT (default 3).
        The metabolism layer (Week 3) uses this to decide what gets elevated
        to Tier 1.5 Echo. This service does not perform promotion itself.

        In spacing-sensitive mode, candidates promoted via spaced reinforcements
        will also have higher stability - the metabolism layer can use stability
        as a quality signal when prioritising among candidates.
        """
        return [
            ph for ph, w in self.weights.items()
            if w.hit_count >= PROMOTION_HIT_COUNT
        ]

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """
        Return a diagnostic snapshot for the REPL 'resonance' command.

        Returns a dict with:
          current_turn    : int
          total_patterns  : int
          active_patterns : int   (weight >= 0.3)
          cruft_ratio     : float (fraction of patterns with weight < 0.3)
          top_patterns    : list  (top 10 by weight, each a dict)
        """
        all_weights = [
            (ph, self.compute_weight(ph), self.weights[ph])
            for ph in self.weights
        ]
        all_weights.sort(key=lambda x: x[1], reverse=True)

        active = [x for x in all_weights if x[1] >= 0.3]
        cruft_ratio = (
            (len(all_weights) - len(active)) / len(all_weights)
            if all_weights else 0.0
        )

        top_10 = [
            {
                "pattern_hash": ph[:12] + "...",  # truncated for readability
                "weight": round(weight, 4),
                "stability": round(w.stability, 2),
                "hit_count": w.hit_count,
                "age": self.current_turn - w.last_reinforced_turn,
                "metadata_keys": list(w.metadata.keys()),
            }
            for ph, weight, w in all_weights[:10]
        ]

        return {
            "current_turn": self.current_turn,
            "total_patterns": len(self.weights),
            "active_patterns": len(active),
            "cruft_ratio": round(cruft_ratio, 3),
            "top_patterns": top_10,
        }
