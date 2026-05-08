"""
Shannon Grain Orchestrator - End-to-End Pipeline Coordination

End-to-end orchestration of grain crystallization.

Pipeline:
1. Track phantoms (persistent patterns)
2. Detect harmonic lock (50+ cycles stable)
3. Validate with axiom rules
4. Crystallize to grains
5. Activate in L3 cache

Author: Kitbash Team
Date: February 2026
Phase: 2C Consolidation → Modularized (May 2026)
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from datetime import datetime, timezone

from kitbash_cartridge import Cartridge
from axiom_validator import AxiomValidator

from .phantom_tracker import PhantomTracker
from .grain_registry import GrainRegistry
from .ternary_crush import TernaryCrush
from .grain_crystallizer import GrainCrystallizer
from .data_structures import GrainMetadata, GrainState, PhantomCandidate, EpistemicLevel


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
    
    def __init__(self, cartridge_id: str, storage_path: str = "./grains", cartridge_engine=None):
        """Initialize orchestrator
        
        Args:
            cartridge_id: Identifier for this cartridge
            storage_path: Where to store crystallized grains
            cartridge_engine: Optional CartridgeInferenceEngine for validation
        """
        self.cartridge_id = cartridge_id
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Core components
        self.phantom_tracker = PhantomTracker(cartridge_id)
        # Initialize validator with cartridge if provided
        self.axiom_validator = None  # Will be set if cartridge_engine provided
        self.cartridge_engine = cartridge_engine
        self.grain_registry = GrainRegistry(cartridge_id, str(self.storage_path))
        
        # Note: TernaryCrush and AxiomValidator require Cartridge, which will be provided at crystallize time
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
        # If no validator available, pass through all phantoms
        # (validator requires Cartridge which is provided at crystallization time)
        if self.axiom_validator is None:
            return {
                'total': len(phantoms),
                'validated': len(phantoms),
                'passed': phantoms,
                'failed': [],
                'pass_rate': 1.0 if phantoms else 0.0
            }
        
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
                
                # Determine grain type based on confidence (NEW - Phase 5A)
                confidence = phantom_info.get('confidence', phantom.avg_confidence())
                if confidence >= 0.95:
                    grain_type = "axiom"
                    confidence_mutable = False
                elif confidence >= 0.70:
                    grain_type = "observation"
                    confidence_mutable = True
                else:
                    # Don't crystallize sub-0.70 confidence
                    continue
                
                # Convert to GrainMetadata with L1/L2 typing
                grain = GrainMetadata(
                    grain_id=grain_dict['grain_id'],
                    source_phantom_id=phantom.phantom_id,
                    cartridge_id=self.cartridge_id,
                    grain_type=grain_type,          # NEW: L1/L2 distinction
                    confidence=confidence,          # NEW: primary confidence metric
                    confidence_mutable=confidence_mutable,  # NEW: mutability flag
                    state=GrainState.CRYSTALLIZED,
                    crystallized_at=datetime.now(timezone.utc).isoformat(),
                    avg_confidence=confidence,      # Legacy compatibility
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
            "axioms_created": len([g for g in crystallized_grains if g.grain_type == "axiom"]),
            "observations_created": len([g for g in crystallized_grains if g.grain_type == "observation"]),
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
    
    def get_locked_phantoms(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Return top-N locked phantoms for L2 auditability.
        
        Used by L2WorkingTheoryService (Phase 5A) to expose active patterns.
        Called by sleep pipeline to understand emerging hypotheses.
        """
        all_locked = self.phantom_tracker.get_locked_phantoms()
        
        return [
            {
                "phantom_id": p.phantom_id,
                "lock_strength": p.avg_confidence(),
                "supporting_queries": len(p.hit_history),
                "expected_grain": f"grain_{p.phantom_id[:16]}",
                "concepts": list(set(p.query_concepts))[:5],  # Top 5 unique concepts
            }
            for p in sorted(all_locked, key=lambda x: x.avg_confidence(), reverse=True)[:top_n]
        ]
