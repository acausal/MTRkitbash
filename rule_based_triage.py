"""
Rule-Based Triage Agent - Phase 3B MVP

Simple hardcoded routing rules for Phase 3B MVP.
Implements both query routing (to inference engines) and background routing
(to maintenance tasks).

Rules are based on:
- Query length
- Query patterns
- System state (resonance cruft, cartridge size, etc.)

Updated for Phase 3B MVP: GRAIN → CARTRIDGE only (BITNET deferred to Phase 4)
"""

import logging
from typing import List, Dict, Any, Optional
from interfaces.triage_agent import (
    TriageAgent,
    TriageRequest,
    TriageDecision,
    BackgroundTriageRequest,
    BackgroundTriageDecision,
    DEFAULT_CONFIDENCE_THRESHOLDS,
)

logger = logging.getLogger(__name__)


class RuleBasedTriageAgent(TriageAgent):
    """
    MVP triage agent using hardcoded rules.
    
    Rules for Query Routing (Phase 3B MVP):
    1. Very short queries (<3 words) → try GRAIN first (likely explicit)
    2. Short queries (3-10 words) → GRAIN → CARTRIDGE → ESCALATE
    3. Long queries (>10 words) → CARTRIDGE → ESCALATE (complex)
    4. Explicit fact reference ("fact 42") → GRAIN directly
    5. Default → multi-layer sieve
    
    Phase 4+ will add:
    - BITNET for learned inference
    - LLM for open-ended reasoning
    
    Rules for Background Routing:
    1. High resonance cruft (>50% low-weight) → DECAY (cleanup)
    2. Large cartridges (>5MB) → ANALYZE_SPLIT (prepare for Phase 4)
    3. Pending crystallization → TEST_CRYSTALLIZER (Phase 4+, stub for now)
    4. Default → ROUTINE (just decay)
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize rule-based triage agent.
        
        Args:
            verbose: Whether to log detailed reasoning
        """
        super().__init__()
        self.verbose = verbose
        self.query_count = 0
        self.background_count = 0
        self.current_turn = 0  # Standard attribute for metabolism tracking
    
    # ========================================================================
    # QUERY ROUTING RULES
    # ========================================================================
    
    def route(self, request: TriageRequest) -> TriageDecision:
        """
        Route a query using hardcoded rules.
        
        Args:
            request: TriageRequest with user query and context
        
        Returns:
            TriageDecision with layer sequence and confidence thresholds
        """
        self.query_count += 1
        
        if not request.user_query or not request.user_query.strip():
            raise ValueError("Query cannot be empty")
        
        # Extract query features
        query_lower = request.user_query.lower().strip()
        word_count = len(query_lower.split())
        
        # RULE 1: Explicit fact reference?
        if self._contains_explicit_fact_reference(query_lower):
            decision = self._route_explicit_fact(request)
        
        # RULE 2: Very short query (likely a lookup)
        elif word_count <= 2:
            decision = self._route_very_short_query(request)
        
        # RULE 3: Short query (balanced search)
        elif word_count <= 10:
            decision = self._route_short_query(request)
        
        # RULE 4: Long query (complex reasoning needed)
        else:
            decision = self._route_long_query(request)
        
        if self.verbose:
            logger.info(
                f"Query routing (#{self.query_count}): {request.user_query[:50]}... "
                f"→ {decision.layer_sequence}, reasoning: {decision.reasoning}"
            )
        
        return decision
    
    def _contains_explicit_fact_reference(self, query: str) -> bool:
        """
        Check if query explicitly references a fact.
        
        Examples: "fact 42", "fact_id 42", "Tell me about fact #42"
        """
        import re
        return bool(re.search(r'fact[_\s#]*(?:id)?[:\s]*\d+', query, re.IGNORECASE))
    
    def _route_explicit_fact(self, request: TriageRequest) -> TriageDecision:
        """
        Route explicit fact reference.
        
        → High confidence in GRAIN, escalate to cartridge if not found
        """
        return TriageDecision(
            layer_sequence=["GRAIN", "CARTRIDGE", "ESCALATE"],
            confidence_thresholds={
                "GRAIN": 0.85,  # Lower bar since explicit reference
                "CARTRIDGE": 0.70,
            },
            recommended_cartridges=[],
            use_mamba_context=False,
            cache_result=True,
            reasoning="Explicit fact reference detected - try grain directly"
        )
    
    def _route_very_short_query(self, request: TriageRequest) -> TriageDecision:
        """
        Route very short queries (1-2 words).
        
        These are likely direct lookups like "ATP", "photosynthesis"
        → GRAIN first (fastest), then cartridge
        """
        return TriageDecision(
            layer_sequence=["GRAIN", "CARTRIDGE", "ESCALATE"],
            confidence_thresholds={
                "GRAIN": 0.90,
                "CARTRIDGE": 0.70,
            },
            recommended_cartridges=[],
            use_mamba_context=False,
            cache_result=True,
            reasoning="Very short query - likely explicit lookup, try GRAIN first"
        )
    
    def _route_short_query(self, request: TriageRequest) -> TriageDecision:
        """
        Route short queries (3-10 words).
        
        Balanced approach: try both layers in order
        → GRAIN (precise) → CARTRIDGE (broader) → ESCALATE
        """
        return TriageDecision(
            layer_sequence=["GRAIN", "CARTRIDGE", "ESCALATE"],
            confidence_thresholds={
                "GRAIN": 0.90,
                "CARTRIDGE": 0.70,
            },
            recommended_cartridges=[],
            use_mamba_context=False,
            cache_result=True,
            reasoning="Short query - balanced multi-layer approach"
        )
    
    def _route_long_query(self, request: TriageRequest) -> TriageDecision:
        """
        Route long queries (>10 words).
        
        These are complex and need reasoning.
        In MVP, escalate to SillyTavern LLM (no BITNET yet).
        → Skip GRAIN (unlikely to have crystallized), go to CARTRIDGE then escalate
        """
        return TriageDecision(
            layer_sequence=["CARTRIDGE", "ESCALATE"],
            confidence_thresholds={
                "CARTRIDGE": 0.70,
            },
            recommended_cartridges=[],
            use_mamba_context=True,  # Use context for complex queries
            cache_result=True,
            reasoning="Long query - skip GRAIN, focus on cartridge, then escalate to LLM"
        )
    
    # ========================================================================
    # BACKGROUND ROUTING RULES
    # ========================================================================
    
    def route_background(self, request: BackgroundTriageRequest) -> BackgroundTriageDecision:
        """
        Route background maintenance work using hardcoded rules.
        
        Args:
            request: BackgroundTriageRequest with system state
        
        Returns:
            BackgroundTriageDecision with priority and parameters
        """
        self.background_count += 1
        
        if not request.resonance_patterns:
            # No patterns tracked yet
            decision = BackgroundTriageDecision(
                priority="routine",
                reasoning="No patterns to analyze yet",
                urgency=0.1
            )
        
        # RULE 1: High resonance cruft (many low-weight patterns)?
        elif self._has_high_resonance_cruft(request.resonance_patterns):
            decision = BackgroundTriageDecision(
                priority="decay",
                reasoning="High resonance cruft - cleaning up low-weight patterns",
                urgency=0.5,
                estimated_duration_ms=100.0
            )
        
        # RULE 2: Large cartridges (preparation for auto-split in Phase 4)?
        elif self._has_large_cartridges(request.cartridge_stats):
            decision = BackgroundTriageDecision(
                priority="analyze_split",
                reasoning="Large cartridges detected - analyzing for potential split",
                urgency=0.3,
                estimated_duration_ms=200.0,
                parameters={"analyze_threshold_mb": 5.0}
            )
        
        # RULE 3: Default - routine maintenance
        else:
            decision = BackgroundTriageDecision(
                priority="decay",
                reasoning="Routine maintenance - standard resonance decay",
                urgency=0.2,
                estimated_duration_ms=50.0
            )
        
        if self.verbose:
            logger.info(
                f"Background routing (#{self.background_count}): "
                f"→ {decision.priority}, urgency: {decision.urgency}"
            )
        
        return decision
    
    def _has_high_resonance_cruft(self, resonance_patterns: Dict[str, Any]) -> bool:
        """
        Check if resonance has high cruft (many low-weight patterns).
        
        Low-weight = weight < 0.01
        High cruft = >50% of patterns are low-weight
        """
        if not resonance_patterns:
            return False
        
        low_weight_count = 0
        for pattern_data in resonance_patterns.values():
            # Handle both dict and object with weight attribute
            if isinstance(pattern_data, dict):
                weight = pattern_data.get('weight', 0.0)
            else:
                weight = getattr(pattern_data, 'weight', 0.0)
            
            if weight < 0.01:
                low_weight_count += 1
        
        cruft_ratio = low_weight_count / len(resonance_patterns)
        return cruft_ratio > 0.5
    
    def _has_large_cartridges(self, cartridge_stats: Dict[str, Any]) -> bool:
        """
        Check if any cartridge is large (>5MB).
        
        Signals preparation for auto-split in Phase 4.
        """
        if not cartridge_stats:
            return False
        
        for cart_name, stats in cartridge_stats.items():
            if isinstance(stats, dict):
                size_mb = stats.get('size_mb', 0.0)
            else:
                size_mb = getattr(stats, 'size_mb', 0.0)
            
            if size_mb > 5.0:
                return True
        
        return False
    
    # ========================================================================
    # STATISTICS & DEBUGGING
    # ========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Return routing statistics."""
        return {
            "query_count": self.query_count,
            "background_count": self.background_count,
            "total_routing_decisions": self.query_count + self.background_count,
            "current_turn": self.current_turn
        }
    
    def print_routing_summary(self) -> None:
        """Print routing statistics."""
        stats = self.get_stats()
        print("\n" + "="*70)
        print("RULE-BASED TRIAGE AGENT STATISTICS")
        print("="*70)
        print(f"Query routing decisions: {stats['query_count']}")
        print(f"Background routing decisions: {stats['background_count']}")
        print(f"Total decisions: {stats['total_routing_decisions']}")
        print("="*70 + "\n")
