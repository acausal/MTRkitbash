"""
metabolism_state.py - Shared state across Phase 4 metabolism cycles

Defines MetabolismState, a dataclass that tracks learning and coordination
state across background/daydream/sleep cycles. This is shared by all three
cycle implementations and enables cross-cycle communication.

Phase 4: Metabolism & Learning
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)


class CycleType(Enum):
    """Types of metabolism cycles."""
    BACKGROUND = "background"      # Continuous analysis (every N queries)
    DAYDREAMING = "daydreaming"    # Testing/validation (Phase 4.2)
    SLEEP = "sleep"                # Consolidation (Phase 4.4)


@dataclass
class PatternSignal:
    """Signal generated during pattern analysis."""
    pattern_id: str
    signal_type: str  # "epistemology_violation", "question_overload", "success", etc.
    confidence: float  # 0-1
    details: Dict[str, Any]
    cycle_origin: CycleType
    timestamp: float = field(default_factory=time.time)


@dataclass
class CycleSignal:
    """Signal from one cycle to another."""
    from_cycle: CycleType
    to_cycle: CycleType
    signal_type: str  # "contradiction_found", "pattern_validated", etc.
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


@dataclass
class MetabolismState:
    """
    Shared state across all Phase 4 metabolism cycles.
    
    This is the single source of truth for learning state, coordination,
    and cross-cycle communication during Phase 4.
    
    Attributes:
        current_turn: Current turn number from HeartbeatService
        cycle_type: Which cycle is currently running (for logging)
        
        learned_patterns: Dict of pattern_id → pattern details
        pattern_versions: Dict of pattern_id → version number (for rollback)
        pattern_confidence: Dict of pattern_id → confidence score
        
        epistemically_valid_patterns: Set of pattern IDs that passed L0-L5 check
        question_adjusted_scores: Dict of pattern_id → adjusted confidence
        faction_tags: Dict of pattern_id → "fiction" | "general" | "experiment"
        
        daydream_contradictions: List of pattern IDs flagged by daydream
        sleep_consolidations: List of pattern IDs consolidated by sleep
        
        background_signals: List of signals from background cycle
        daydream_signals: List of signals from daydream cycle
        sleep_signals: List of signals from sleep cycle
        
        baseline_metrics: Baseline metrics for regression detection
        learned_deltas: Total coupling deltas seen so far
    """
    
    # Cycle tracking
    current_turn: int = 0
    cycle_type: Optional[CycleType] = None
    
    # Learning state
    learned_patterns: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    pattern_versions: Dict[str, int] = field(default_factory=dict)
    pattern_confidence: Dict[str, float] = field(default_factory=dict)
    
    # Safety state
    epistemically_valid_patterns: Set[str] = field(default_factory=set)
    question_adjusted_scores: Dict[str, float] = field(default_factory=dict)
    faction_tags: Dict[str, str] = field(default_factory=dict)
    
    # Cross-cycle signals
    daydream_contradictions: List[str] = field(default_factory=list)
    sleep_consolidations: List[str] = field(default_factory=list)
    
    # Signal queues
    background_signals: List[PatternSignal] = field(default_factory=list)
    daydream_signals: List[PatternSignal] = field(default_factory=list)
    sleep_signals: List[PatternSignal] = field(default_factory=list)
    cross_cycle_signals: List[CycleSignal] = field(default_factory=list)
    
    # Metrics
    baseline_metrics: Dict[str, float] = field(default_factory=dict)
    learned_deltas: int = 0
    
    def record_pattern(
        self,
        pattern_id: str,
        pattern_details: Dict[str, Any],
        faction: str = "general"
    ) -> None:
        """
        Record a new learned pattern.
        
        Args:
            pattern_id: Unique pattern identifier
            pattern_details: Pattern data (layer_sequence, success_rate, etc.)
            faction: "fiction", "general", or "experiment"
        """
        self.learned_patterns[pattern_id] = pattern_details
        self.pattern_versions[pattern_id] = 1
        self.pattern_confidence[pattern_id] = pattern_details.get("confidence", 0.5)
        self.faction_tags[pattern_id] = faction
        
        logger.debug(f"Recorded pattern {pattern_id} (faction={faction})")
    
    def mark_epistemically_valid(self, pattern_id: str) -> None:
        """Mark pattern as passing epistemological validation."""
        if pattern_id in self.learned_patterns:
            self.epistemically_valid_patterns.add(pattern_id)
            logger.debug(f"Pattern {pattern_id} marked epistemically valid")
    
    def set_question_adjusted_score(self, pattern_id: str, score: float) -> None:
        """
        Set question-adjusted confidence for pattern.
        
        Formula: adjusted = base_success * (1 - question_rate)
        
        Args:
            pattern_id: Pattern identifier
            score: Adjusted confidence score (0-1)
        """
        self.question_adjusted_scores[pattern_id] = score
        logger.debug(f"Pattern {pattern_id} adjusted score: {score:.3f}")
    
    def add_background_signal(
        self,
        pattern_id: str,
        signal_type: str,
        confidence: float,
        details: Dict[str, Any]
    ) -> None:
        """Add signal from background cycle."""
        signal = PatternSignal(
            pattern_id=pattern_id,
            signal_type=signal_type,
            confidence=confidence,
            details=details,
            cycle_origin=CycleType.BACKGROUND
        )
        self.background_signals.append(signal)
    
    def add_daydream_signal(
        self,
        pattern_id: str,
        signal_type: str,
        confidence: float,
        details: Dict[str, Any]
    ) -> None:
        """Add signal from daydream cycle."""
        signal = PatternSignal(
            pattern_id=pattern_id,
            signal_type=signal_type,
            confidence=confidence,
            details=details,
            cycle_origin=CycleType.DAYDREAMING
        )
        self.daydream_signals.append(signal)
        
        # Track contradictions
        if signal_type == "contradiction_found":
            self.daydream_contradictions.append(pattern_id)
    
    def add_sleep_signal(
        self,
        pattern_id: str,
        signal_type: str,
        confidence: float,
        details: Dict[str, Any]
    ) -> None:
        """Add signal from sleep cycle."""
        signal = PatternSignal(
            pattern_id=pattern_id,
            signal_type=signal_type,
            confidence=confidence,
            details=details,
            cycle_origin=CycleType.SLEEP
        )
        self.sleep_signals.append(signal)
        
        # Track consolidations
        if signal_type == "consolidated":
            self.sleep_consolidations.append(pattern_id)
    
    def add_cross_cycle_signal(
        self,
        from_cycle: CycleType,
        to_cycle: CycleType,
        signal_type: str,
        payload: Dict[str, Any]
    ) -> None:
        """Add signal from one cycle to another."""
        signal = CycleSignal(
            from_cycle=from_cycle,
            to_cycle=to_cycle,
            signal_type=signal_type,
            payload=payload
        )
        self.cross_cycle_signals.append(signal)
    
    def get_signals_for_cycle(self, cycle_type: CycleType) -> List[CycleSignal]:
        """Get cross-cycle signals destined for this cycle."""
        return [s for s in self.cross_cycle_signals if s.to_cycle == cycle_type]
    
    def is_pattern_valid(self, pattern_id: str) -> bool:
        """Check if pattern passed all validation checks."""
        if pattern_id not in self.learned_patterns:
            return False
        
        # Must pass epistemology
        if pattern_id not in self.epistemically_valid_patterns:
            return False
        
        # Must not be flagged by daydream
        if pattern_id in self.daydream_contradictions:
            return False
        
        return True
    
    def get_pattern_status(self, pattern_id: str) -> Dict[str, Any]:
        """Get complete status for a pattern."""
        return {
            "pattern_id": pattern_id,
            "exists": pattern_id in self.learned_patterns,
            "version": self.pattern_versions.get(pattern_id, 0),
            "confidence": self.pattern_confidence.get(pattern_id, 0.0),
            "question_adjusted": self.question_adjusted_scores.get(pattern_id, None),
            "faction": self.faction_tags.get(pattern_id, "unknown"),
            "epistemically_valid": pattern_id in self.epistemically_valid_patterns,
            "daydream_flagged": pattern_id in self.daydream_contradictions,
            "sleep_consolidated": pattern_id in self.sleep_consolidations,
        }
    
    def clear_signals(self) -> None:
        """Clear all signal queues (called after cycle processes them)."""
        self.background_signals.clear()
        self.daydream_signals.clear()
        self.sleep_signals.clear()
        self.cross_cycle_signals.clear()
    
    def summary(self) -> Dict[str, Any]:
        """Get summary of metabolism state."""
        return {
            "current_turn": self.current_turn,
            "cycle_type": self.cycle_type.value if self.cycle_type else None,
            "patterns_learned": len(self.learned_patterns),
            "patterns_valid": len(self.epistemically_valid_patterns),
            "patterns_questioned": len(self.question_adjusted_scores),
            "daydream_flags": len(self.daydream_contradictions),
            "sleep_consolidations": len(self.sleep_consolidations),
            "pending_signals": len(self.cross_cycle_signals),
            "learned_deltas": self.learned_deltas,
        }


# ============================================================================
# Testing helpers
# ============================================================================

def create_test_state() -> MetabolismState:
    """Create a MetabolismState for testing."""
    state = MetabolismState(
        current_turn=100,
        baseline_metrics={
            "success_rate": 0.85,
            "question_rate": 0.1,
            "critical_coupling_rate": 0.02,
        }
    )
    return state


if __name__ == "__main__":
    """Quick test of MetabolismState."""
    
    logging.basicConfig(level=logging.DEBUG)
    
    state = create_test_state()
    
    # Record a pattern
    state.record_pattern(
        "pattern_001",
        {
            "layer_sequence": ["GRAIN", "CARTRIDGE"],
            "success_rate": 0.92,
            "confidence": 0.87,
        },
        faction="general"
    )
    
    # Mark as valid
    state.mark_epistemically_valid("pattern_001")
    
    # Add question-adjusted score
    state.set_question_adjusted_score("pattern_001", 0.85)
    
    # Add signals
    state.add_background_signal(
        "pattern_001",
        "success",
        0.92,
        {"queries_seen": 100}
    )
    
    # Get status
    status = state.get_pattern_status("pattern_001")
    print(f"\nPattern status: {status}")
    
    # Get summary
    summary = state.summary()
    print(f"\nState summary: {summary}")
    
    print("\n✅ MetabolismState working correctly")
