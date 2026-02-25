"""
MTR ↔ Grain Bridge: Integration layer between Phase 5.5 and Phase 2C

Consolidates:
- MTR dissonance → phantom tracking
- Hat context → epistemic kappa mapping
- Crystallization cycle triggering
- Unified grain activation

Author: Kitbash Team
Date: February 2026
"""

import torch
from typing import Optional, Dict, List, Any, Set
from datetime import datetime

from grain_system import ShannonGrainOrchestrator, PhantomTracker


# ============================================================================
# COMPONENT 1: MTR PHANTOM BRIDGE
# ============================================================================

class MTRPhantomBridge:
    """
    Logs MTR inference results as phantom hits.
    
    When MTR processes queries and generates predictions, this bridge captures:
    - Query concepts (tokens processed)
    - Error signal (confidence metric)
    - Epistemic snapshot (layer saliences)
    - Maps to phantom tracking for crystallization
    
    Philosophy: High MTR error = low confidence = valuable learning signal
    These become phantom candidates for grain crystallization.
    """
    
    def __init__(self, orchestrator: ShannonGrainOrchestrator):
        """
        Initialize phantom bridge.
        
        Args:
            orchestrator: ShannonGrainOrchestrator instance for phantom tracking
        """
        self.orchestrator = orchestrator
        self.phantom_tracker = orchestrator.phantom_tracker
        self.query_count = 0
        self.total_hits_logged = 0
    
    def record_mtr_hit(self, fact_ids: Set[int], query_tokens: List[int],
                      error_signal: torch.Tensor, 
                      epistemic_snapshot: Dict[str, tuple],
                      dissonance_result: Optional[Dict] = None) -> None:
        """
        Log an MTR inference as a phantom hit.
        
        Called after MTR processes a query. Converts error signal to confidence
        and feeds into phantom tracker.
        
        Args:
            fact_ids: Set of fact IDs used in this query (from CartridgeLoader/GrainRouter)
            query_tokens: Token IDs from query
            error_signal: MTR error (lower = more confident)
            epistemic_snapshot: Dict of epistemic layer outputs
            dissonance_result: Optional output from DissonanceSensor
        """
        self.query_count += 1
        
        if not fact_ids:
            return
        
        # Convert MTR error to confidence
        # error_signal is (batch, seq_len, 1), take mean across sequence
        try:
            error_mean = float(error_signal.mean().clamp(0, 1))
            confidence = 1.0 - error_mean  # Invert: low error = high confidence
        except:
            confidence = 0.5  # Safe default
        
        # Extract concepts from epistemic layers if available
        concepts = self._extract_concepts_from_snapshot(epistemic_snapshot)
        
        # Record as phantom hit
        self.phantom_tracker.record_phantom_hit(
            fact_ids=fact_ids,
            concepts=concepts,
            confidence=confidence
        )
        
        self.total_hits_logged += 1
        
        # Optional: Log dissonance if provided
        if dissonance_result and dissonance_result.get('dissonance_active', False).any():
            dissonance_count = int(dissonance_result['dissonance_active'].sum())
            if dissonance_count > 0:
                self._record_dissonance_event(
                    fact_ids, dissonance_count, error_mean, dissonance_result
                )
    
    def _extract_concepts_from_snapshot(self, epistemic_snapshot: Dict[str, tuple]) -> List[str]:
        """
        Extract meaningful concepts from epistemic layer snapshot.
        
        Uses layer saliences to weight which concepts to extract.
        Higher salience = more relevant concept.
        
        Args:
            epistemic_snapshot: Dict mapping layer names to (representation, salience)
        
        Returns:
            List of concept strings
        """
        concepts = []
        
        if not epistemic_snapshot:
            return concepts
        
        # Extract from each epistemic layer
        layer_weights = {
            'L0_empirical': 0.20,    # Facts (low weight, too specific)
            'L1_axiomatic': 0.25,    # Rules (medium weight)
            'L2_narrative': 0.20,    # Story (medium weight)
            'L3_heuristic': 0.15,    # Analogies (lower weight)
            'L4_intent': 0.15,       # Goals (lower weight)
            'L5_phatic': 0.05,       # Tone (very low weight)
        }
        
        for layer_name, (representation, salience) in epistemic_snapshot.items():
            weight = layer_weights.get(layer_name, 0.1)
            
            # Use salience to decide if we extract from this layer
            try:
                salience_val = float(salience.mean().clamp(0, 1))
                if salience_val > 0.3:  # Only extract if salient
                    concept = f"{layer_name}:{salience_val:.2f}"
                    concepts.append(concept)
            except:
                pass
        
        return concepts
    
    def _record_dissonance_event(self, fact_ids: Set[int], dissonance_count: int,
                                error_mean: float, dissonance_result: Dict) -> None:
        """
        Record when MTR detected high dissonance (layer disagreement).
        
        Useful for sleep process to identify confused regions.
        """
        # Store for later analysis (optional)
        pass
    
    def advance_cycle(self) -> None:
        """Advance phantom tracker cycle (call every ~100 queries)."""
        self.phantom_tracker.advance_cycle()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get phantom tracking statistics."""
        return {
            'query_count': self.query_count,
            'total_hits_logged': self.total_hits_logged,
            'phantom_tracker_stats': self.phantom_tracker.get_stats(),
        }


# ============================================================================
# COMPONENT 2: HAT ↔ KAPPA MAPPER
# ============================================================================

class HatKappaMapper:
    """
    Maps behavioral context (Hat) to epistemic rigidity (kappa).
    
    Kappa controls how strictly MTR enforces consistency across epistemic layers:
    - kappa > 1.0: Rigid (logical, analytical, consistent)
    - kappa = 1.0: Balanced (default)
    - kappa < 1.0: Fluid (creative, exploratory, associative)
    
    Hat modes from grain_activation.py:
    - ANALYTICAL: Logical reasoning, facts first
    - DELIBERATIVE: Careful, cautious reasoning
    - NEUTRAL: Balanced approach
    - CREATIVE: Imaginative, associative
    - EMPATHETIC: Emotional, contextual
    """
    
    # Map Hat modes to kappa values
    # Import these from grain_activation.py in real implementation
    HAT_TO_KAPPA = {
        'ANALYTICAL': 2.0,       # Rigid: enforce consistency strictly
        'DELIBERATIVE': 1.5,     # Careful: slightly rigid
        'NEUTRAL': 1.0,          # Default balanced
        'CREATIVE': 0.5,         # Fluid: allow divergence
        'EMPATHETIC': 0.8,       # Soft: context-sensitive
    }
    
    @staticmethod
    def get_kappa(hat: Any) -> float:
        """
        Get kappa value for a behavioral context.
        
        Args:
            hat: Hat enum value or string name
        
        Returns:
            Kappa value (float, typically 0.5-2.0)
        """
        if hat is None:
            return 1.0  # Default
        
        # Handle both enum and string
        hat_name = hat.name if hasattr(hat, 'name') else str(hat).upper()
        
        return HatKappaMapper.HAT_TO_KAPPA.get(hat_name, 1.0)
    
    @staticmethod
    def describe_mode(hat: Any) -> str:
        """
        Get human-readable description of hat+kappa mode.
        
        Args:
            hat: Hat enum value
        
        Returns:
            Description string
        """
        kappa = HatKappaMapper.get_kappa(hat)
        
        if kappa > 1.5:
            return "Rigid (Analytical): Strict consistency enforcement"
        elif kappa > 1.0:
            return "Careful: Moderate consistency enforcement"
        elif kappa == 1.0:
            return "Balanced: Default epistemic routing"
        elif kappa > 0.5:
            return "Soft: Context-sensitive reasoning"
        else:
            return "Fluid (Creative): Exploratory associations"


# ============================================================================
# COMPONENT 3: GRAIN ORCHESTRATION TRIGGER
# ============================================================================

class GrainOrchestrationTrigger:
    """
    Manages when to run crystallization cycles.
    
    Monitors query count and decides when:
    1. Enough phantoms have accumulated
    2. To trigger validation cycle
    3. To load new grains into cache
    
    Philosophy: Crystallization is expensive (validation + crushing + storage),
    so batch it and run periodically rather than on every query.
    """
    
    def __init__(self, orchestrator: ShannonGrainOrchestrator,
                 grain_router: Optional[Any] = None,
                 trigger_interval: int = 100,
                 cartridge: Optional[Any] = None):
        """
        Initialize orchestration trigger.
        
        Args:
            orchestrator: ShannonGrainOrchestrator instance
            grain_router: Optional GrainRouter for loading new grains
            trigger_interval: How many queries between crystallization attempts
            cartridge: Optional Cartridge instance for crushing
        """
        self.orchestrator = orchestrator
        self.grain_router = grain_router
        self.trigger_interval = trigger_interval
        self.cartridge = cartridge
        self.query_count = 0
        self.crystallization_count = 0
    
    def maybe_crystallize(self) -> Optional[Dict[str, Any]]:
        """
        Check if it's time to run crystallization cycle.
        
        Returns:
            Crystallization report if triggered, None otherwise
        """
        self.query_count += 1
        
        # Check if we should crystallize
        if self.query_count % self.trigger_interval != 0:
            return None
        
        # Time to crystallize
        if self.cartridge is None:
            return None  # Can't crystallize without cartridge
        
        try:
            result = self.orchestrator.run_crystallization_cycle(self.cartridge)
            self.crystallization_count += 1
            
            # Reload grains in router if we crystallized anything
            grains_created = result.get('phase_3_crystallization', {}).get('grains_created', 0)
            if grains_created > 0 and self.grain_router is not None:
                self._reload_grains_in_router()
            
            return result
        
        except Exception as e:
            print(f"Warning: Crystallization cycle failed: {e}")
            return None
    
    def _reload_grains_in_router(self) -> None:
        """
        Reload grains in router to pick up newly crystallized grains.
        
        This allows new grains to be used immediately in future queries.
        """
        try:
            self.grain_router._load_grains()
        except Exception as e:
            print(f"Warning: Could not reload grains in router: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get trigger statistics."""
        return {
            'query_count': self.query_count,
            'crystallization_count': self.crystallization_count,
            'next_crystallization_in': self.trigger_interval - (self.query_count % self.trigger_interval),
        }


# ============================================================================
# COMPONENT 4: UNIFIED MTR-GRAIN PIPELINE
# ============================================================================

class MTRGrainUnifiedPipeline:
    """
    Coordinates the full MTR + Grain pipeline.
    
    This is the main coordinator that ties everything together:
    1. MTR processes queries with learned state
    2. Phantom bridge logs confidence metrics
    3. Hat context controls epistemic routing
    4. Crystallization trigger runs periodically
    5. New grains are loaded into router
    
    Responsibilities:
    - Manage all three bridge components
    - Provide single interface for orchestrator
    - Track end-to-end pipeline statistics
    """
    
    def __init__(self, mtr_grain_orchestrator: ShannonGrainOrchestrator,
                 grain_router: Optional[Any] = None,
                 trigger_interval: int = 100,
                 cartridge: Optional[Any] = None):
        """
        Initialize unified pipeline.
        
        Args:
            mtr_grain_orchestrator: ShannonGrainOrchestrator instance
            grain_router: Optional GrainRouter
            trigger_interval: Crystallization trigger interval
            cartridge: Optional Cartridge for crushing
        """
        self.orchestrator = mtr_grain_orchestrator
        self.grain_router = grain_router
        
        # Create bridge components
        self.phantom_bridge = MTRPhantomBridge(mtr_grain_orchestrator)
        self.kappa_mapper = HatKappaMapper()
        self.crystallization_trigger = GrainOrchestrationTrigger(
            mtr_grain_orchestrator,
            grain_router,
            trigger_interval,
            cartridge
        )
        
        # Statistics
        self.pipeline_stats = {
            'mtr_queries': 0,
            'phantom_hits': 0,
            'crystallizations': 0,
            'grains_loaded': 0,
            'started_at': datetime.now().isoformat(),
        }
    
    def process_mtr_query(self, fact_ids: Set[int], query_tokens: List[int],
                         error_signal: torch.Tensor,
                         epistemic_snapshot: Dict[str, tuple],
                         hat: Optional[Any] = None,
                         dissonance_result: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process a single MTR query through the unified pipeline.
        
        Args:
            fact_ids: Set of fact IDs from cartridge/grain lookup
            query_tokens: Token IDs from query
            error_signal: MTR error signal
            epistemic_snapshot: Epistemic layer outputs
            hat: Optional behavioral context
            dissonance_result: Optional DissonanceSensor output
        
        Returns:
            Pipeline metadata
        """
        self.pipeline_stats['mtr_queries'] += 1
        
        # Step 1: Log to phantom tracker
        self.phantom_bridge.record_mtr_hit(
            fact_ids, query_tokens, error_signal,
            epistemic_snapshot, dissonance_result
        )
        self.pipeline_stats['phantom_hits'] += 1
        
        # Step 2: Get kappa for epistemic routing
        kappa = self.kappa_mapper.get_kappa(hat) if hat else 1.0
        
        # Step 3: Check if crystallization should run
        crystallization_result = self.crystallization_trigger.maybe_crystallize()
        if crystallization_result:
            self.pipeline_stats['crystallizations'] += 1
            grains_created = crystallization_result.get(
                'phase_3_crystallization', {}
            ).get('grains_created', 0)
            self.pipeline_stats['grains_loaded'] += grains_created
        
        # Return metadata about pipeline state
        return {
            'query_number': self.pipeline_stats['mtr_queries'],
            'kappa': kappa,
            'crystallization': crystallization_result,
            'phantom_stats': self.phantom_bridge.get_stats(),
        }
    
    def advance_phantom_cycle(self) -> None:
        """Advance phantom tracker cycle manually."""
        self.phantom_bridge.advance_cycle()
    
    def get_full_stats(self) -> Dict[str, Any]:
        """Get complete pipeline statistics."""
        return {
            'pipeline': self.pipeline_stats,
            'phantom_bridge': self.phantom_bridge.get_stats(),
            'crystallization_trigger': self.crystallization_trigger.get_stats(),
            'orchestrator': self.orchestrator.get_stats(),
        }
    
    def print_summary(self) -> None:
        """Print human-readable pipeline summary."""
        stats = self.get_full_stats()
        
        print("\n" + "="*70)
        print("MTR GRAIN UNIFIED PIPELINE SUMMARY")
        print("="*70)
        
        print("\nPipeline Metrics:")
        print(f"  MTR Queries: {stats['pipeline']['mtr_queries']}")
        print(f"  Phantom Hits Logged: {stats['pipeline']['phantom_hits']}")
        print(f"  Crystallization Cycles: {stats['pipeline']['crystallizations']}")
        print(f"  Grains Loaded: {stats['pipeline']['grains_loaded']}")
        
        print("\nPhantom Tracking:")
        phantom_stats = stats['phantom_bridge']['phantom_tracker_stats']
        print(f"  Total Phantoms: {phantom_stats['total_phantoms']}")
        print(f"  Persistent: {phantom_stats['persistent_count']}")
        print(f"  Locked (Ready): {phantom_stats['locked_count']}")
        
        print("\nCrystallization Trigger:")
        trigger_stats = stats['crystallization_trigger']
        print(f"  Next Crystallization In: {trigger_stats['next_crystallization_in']} queries")
        
        print("\nOrchestrator State:")
        orch_stats = stats['orchestrator']
        print(f"  Total Crystallized: {orch_stats['total_crystallized']}")
        
        print("="*70 + "\n")


# ============================================================================
# EXPORTS & EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("MTR Grain Bridge - Integration Layer")
    print("="*70)
    print("\nAvailable classes:")
    print("  - MTRPhantomBridge (log MTR results as phantom hits)")
    print("  - HatKappaMapper (behavioral context → epistemic rigidity)")
    print("  - GrainOrchestrationTrigger (crystallization scheduling)")
    print("  - MTRGrainUnifiedPipeline (complete orchestration)")
    print("\n" + "="*70)
    print("\nUsage in phase3e_orchestrator.py:")
    print("""
    from mtr_grain_bridge import MTRGrainUnifiedPipeline
    from grain_system import ShannonGrainOrchestrator
    from grain_router import GrainRouter
    
    # Initialize
    grain_orch = ShannonGrainOrchestrator("default")
    grain_router = GrainRouter("./cartridges")
    pipeline = MTRGrainUnifiedPipeline(
        grain_orch, grain_router, 
        trigger_interval=100,
        cartridge=cartridge_obj
    )
    
    # In process_query():
    metadata = pipeline.process_mtr_query(
        fact_ids=set([1, 2, 3]),
        query_tokens=token_ids[0].tolist(),
        error_signal=error_signal,
        epistemic_snapshot=epistemic_snapshot,
        hat=context.hat,
        dissonance_result=dissonance_result
    )
    
    # Print stats
    pipeline.print_summary()
    """)
