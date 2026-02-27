"""
Axiom Validator - Sicherman Validation Rules for Grain Crystallization

Validates locked phantoms against three quality gates:
1. Persistence: Phantom pointers resolve to valid facts
2. Least Resistance: Ternary representation achieves compression
3. Independence: Phantom pattern aligns with domain axioms

Phase 3E.1 - Enhanced Validation with Structural Rules
"""

import json
import statistics
from typing import Dict, Any, List, Optional, Tuple, Set
from pathlib import Path
from kitbash_cartridge import Cartridge, FactStatus


class AxiomValidator:
    """
    Validates phantoms before crystallization using Sicherman rules.
    
    Three validation gates ensure grains are structurally sound and
    ready for compression into ternary representation.
    """
    
    def __init__(self, cartridge: Cartridge):
        """Initialize validator for a specific cartridge."""
        self.cartridge = cartridge
        self.cartridge_id = cartridge.name
        self.validation_log: List[Dict] = []
    
    # ========================================================================
    # RULE 1: PERSISTENCE
    # ========================================================================
    
    def _check_persistence(self, phantom: Any) -> bool:
        """
        Verify all phantom fact_ids resolve in cartridge.
        
        Args:
            phantom: PhantomCandidate with fact_ids: Set[int]
        
        Returns:
            True if all facts exist and are ACTIVE
        """
        if not phantom.fact_ids:
            return False
        
        for fact_id in phantom.fact_ids:
            try:
                # Check fact exists in cartridge
                fact_content = self.cartridge.get_fact(fact_id)
                if not fact_content:
                    return False
                
                # Check fact is not DEPRECATED or ARCHIVED
                annotation = self.cartridge.annotations.get(fact_id)
                if annotation and hasattr(annotation, 'status'):
                    if annotation.status in [FactStatus.DEPRECATED, FactStatus.ARCHIVED]:
                        return False
            except:
                return False
        
        return True
    
    # ========================================================================
    # RULE 2: LEAST RESISTANCE
    # ========================================================================
    
    def _count_ternary_derivations(self, derivations: List[Any]) -> int:
        """
        Count derivations that can be expressed as ternary {-1, 0, +1}.
        
        Args:
            derivations: List of Derivation objects
        
        Returns:
            Count of ternary-expressible derivations
        """
        if not derivations:
            return 0
        
        ternary_types = {
            # Positive dependencies → +1
            'positive_dependency',
            'dependency',
            'requires',
            'depends_on',
            'implies',
            'entails',
            # Negative dependencies → -1
            'negative_dependency',
            'negation',
            'inverse',
            'opposite',
            'excludes',
            'contradicts',
            # Independence → 0
            'void',
            'orthogonal',
            'independent',
            'boundary',
        }
        
        count = 0
        for deriv in derivations:
            # Check both the type attribute and string representation
            if hasattr(deriv, 'type') and deriv.type in ternary_types:
                count += 1
            elif hasattr(deriv, 'type'):
                # Try string matching as fallback
                deriv_type = str(deriv.type).lower()
                for ttype in ternary_types:
                    if ttype in deriv_type:
                        count += 1
                        break
        
        return count
    
    def _check_contradictions(self, derivations: List[Any]) -> List[Tuple]:
        """
        Detect circular/mutual dependencies that indicate contradictions.
        
        Returns:
            List of (target_a, target_b) pairs in contradiction
        """
        if not derivations:
            return []
        
        contradictions = []
        
        for i, d1 in enumerate(derivations):
            if not hasattr(d1, 'target') or not hasattr(d1, 'type'):
                continue
            
            # Only check circular dependencies if targets are actually set
            if d1.target is None:
                continue
            
            for d2 in derivations[i+1:]:
                if not hasattr(d2, 'target') or not hasattr(d2, 'type'):
                    continue
                
                # Skip if d2 has no target
                if d2.target is None:
                    continue
                
                # Check for circular: both are "requires"/"depends_on" and targets are swapped
                d1_type = str(d1.type).lower() if hasattr(d1, 'type') else ""
                d2_type = str(d2.type).lower() if hasattr(d2, 'type') else ""
                
                requires_types = {'requires', 'depends_on'}
                
                if (any(rt in d1_type for rt in requires_types) and
                    any(rt in d2_type for rt in requires_types)):
                    
                    # Circular: A requires B AND B requires A (targets must be swapped and non-None)
                    if d1.target == d2.target and d2.target == d1.target:
                        contradictions.append((d1.target, d2.target))
        
        return contradictions
    
    def _check_resistance(self, phantom: Any) -> bool:
        """
        Verify pattern can be compressed to ternary representation.
        
        A pattern has "least resistance" if:
        1. Its derivations naturally map to {-1, 0, +1}
        2. At least 70% of derivations are ternary-expressible
        3. No contradictory patterns
        
        Args:
            phantom: PhantomCandidate with fact_ids
        
        Returns:
            True if pattern is compressible
        """
        if not phantom.fact_ids:
            return False
        
        # Collect all derivations from all facts
        all_derivations = []
        for fact_id in phantom.fact_ids:
            try:
                annotation = self.cartridge.annotations.get(fact_id)
                if annotation and hasattr(annotation, 'derivations'):
                    all_derivations.extend(annotation.derivations)
            except:
                pass
        
        if not all_derivations:
            # No derivations = cannot compress, fail
            return False
        
        # Count ternary-expressible derivations
        ternary_count = self._count_ternary_derivations(all_derivations)
        ternary_ratio = ternary_count / len(all_derivations)
        
        # Check for contradictions
        contradictions = self._check_contradictions(all_derivations)
        
        # Pass if: 70%+ ternary AND no contradictions
        return (ternary_ratio >= 0.70 and len(contradictions) == 0)
    
    # ========================================================================
    # RULE 3: INDEPENDENCE
    # ========================================================================
    
    def _count_oscillations(self, scores: List[float]) -> int:
        """
        Count oscillations (peaks and valleys) in confidence scores.
        
        Args:
            scores: List of confidence scores
        
        Returns:
            Count of oscillations
        """
        if len(scores) < 3:
            return 0
        
        oscillations = 0
        for i in range(1, len(scores) - 1):
            # Peak: score[i] > score[i-1] AND score[i] > score[i+1]
            if scores[i] > scores[i-1] and scores[i] > scores[i+1]:
                oscillations += 1
            # Valley: score[i] < score[i-1] AND score[i] < score[i+1]
            elif scores[i] < scores[i-1] and scores[i] < scores[i+1]:
                oscillations += 1
        
        return oscillations
    
    def _check_independence(self, phantom: Any) -> bool:
        """
        Verify phantom aligns with domain axioms and epistemic level.
        
        Independence means:
        1. Confidence scores stable (variance < 0.02)
        2. Epistemic levels consistent across facts
        3. No excessive oscillation in confidence
        4. No contradictions with domain
        
        Args:
            phantom: PhantomCandidate with confidence_scores list
        
        Returns:
            True if pattern is independent of axioms
        """
        
        # Check 1: Confidence stability
        if len(phantom.confidence_scores) > 1:
            try:
                variance = statistics.variance(phantom.confidence_scores)
                # Threshold: 0.02 for stable patterns
                if variance > 0.02:
                    return False
            except:
                return False
        elif len(phantom.confidence_scores) == 0:
            # No confidence data, fail
            return False
        
        # Check 2: Epistemic level consistency across facts
        try:
            epistemic_levels: Set = set()
            for fact_id in phantom.fact_ids:
                annotation = self.cartridge.annotations.get(fact_id)
                if annotation and hasattr(annotation, 'epistemic_level'):
                    epistemic_levels.add(annotation.epistemic_level)
            
            # All facts should be at same level or within 1 level
            if len(epistemic_levels) > 1:
                # Extract numeric values (assume Enum with .value)
                level_values = []
                for level in epistemic_levels:
                    if hasattr(level, 'value'):
                        level_values.append(level.value)
                    else:
                        level_values.append(int(level))
                
                if max(level_values) - min(level_values) > 1:
                    return False
        except:
            return False
        
        # Check 3: No excessive oscillation
        if len(phantom.confidence_scores) >= 3:
            oscillations = self._count_oscillations(phantom.confidence_scores)
            # Threshold: more than 3 oscillations = unstable
            if oscillations > 3:
                return False
        
        return True
    
    # ========================================================================
    # MAIN VALIDATION METHODS
    # ========================================================================
    
    def validate_phantom(self, phantom: Any, 
                        cartridge_facts: Dict[int, str] = None) -> Dict[str, Any]:
        """
        Apply all three Sicherman validation rules.
        
        Args:
            phantom: PhantomCandidate from locked registry
            cartridge_facts: (unused) All facts in this cartridge (for context)
        
        Returns:
            Validation result with lock_state and rule checks
        """
        
        result = {
            'phantom_id': phantom.phantom_id,
            'cartridge_id': self.cartridge_id,
            'persistent_check': self._check_persistence(phantom),
            'resistance_check': self._check_resistance(phantom),
            'independence_check': self._check_independence(phantom),
            'locked': False,
            'lock_state': 'failed_validation',
            'rule_failures': [],
            'confidence': phantom.avg_confidence(),
            'hit_count': phantom.hit_count,
            'cycles_locked': (phantom.last_cycle_seen - phantom.first_cycle_seen
                             if hasattr(phantom, 'last_cycle_seen') else 0),
        }
        
        # Collect failures for debugging
        if not result['persistent_check']:
            result['rule_failures'].append('persistence: facts do not resolve')
        if not result['resistance_check']:
            result['rule_failures'].append('resistance: not compressible to ternary')
        if not result['independence_check']:
            result['rule_failures'].append('independence: violates axioms or unstable')
        
        # All three rules must pass
        if all([result['persistent_check'],
                result['resistance_check'],
                result['independence_check']]):
            result['locked'] = True
            result['lock_state'] = 'Sicherman_Validated'
        
        # Log the validation
        self.validation_log.append(result)
        return result
    
    def validate_batch(self, phantoms: List[Any], 
                      existing_grains: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Validate a batch of locked PhantomCandidate objects using all three rules.
        
        Args:
            phantoms: List of PhantomCandidate objects (already locked)
            existing_grains: List of existing grains (for deduplication check)
        
        Returns:
            Dict with 'crystallization_ready' key containing validated phantoms
        """
        
        validated = []
        failed = []
        
        for phantom in phantoms:
            # Apply all three Sicherman rules
            result = self.validate_phantom(phantom, {})
            
            if result['locked']:  # All three rules passed
                validated.append({
                    'phantom_id': phantom.phantom_id,
                    'fact_ids': list(phantom.fact_ids),
                    'confidence': result['confidence'],
                    'locked': True,
                    'cycles_locked': result['cycles_locked']
                })
            else:
                failed.append(phantom.phantom_id)
        
        return {
            'total': len(phantoms),
            'validated': len(validated),
            'crystallization_ready': validated,
            'failed': failed,
            'pass_rate': len(validated) / len(phantoms) if phantoms else 0.0
        }
    
    def validate_all_phantoms(self, phantoms: Dict[int, Any],
                             cartridge_facts: Dict[int, str] = None) -> Dict[str, Any]:
        """
        Validate all phantoms in a registry.
        
        Args:
            phantoms: Dict of phantom_id -> PhantomCandidate
            cartridge_facts: All facts in cartridge (unused)
        
        Returns:
            Summary of validation results
        """
        
        locked_count = 0
        failed_count = 0
        
        for phantom_id, phantom in phantoms.items():
            result = self.validate_phantom(phantom, {})
            if result['locked']:
                locked_count += 1
            else:
                failed_count += 1
        
        return {
            'total': len(phantoms),
            'locked': locked_count,
            'failed': failed_count,
            'pass_rate': locked_count / len(phantoms) if phantoms else 0,
            'cartridge_id': self.cartridge_id,
            'validation_log': self.validation_log
        }
    
    def get_locked_phantoms(self, phantoms: Dict[int, Any],
                           cartridge_facts: Dict[int, str] = None) -> List[Any]:
        """Get list of phantoms that passed all validation rules."""
        
        locked = []
        for phantom_id, phantom in phantoms.items():
            result = self.validate_phantom(phantom, {})
            if result['locked']:
                locked.append(phantom)
        
        return locked
    
    def print_summary(self):
        """Print validation summary."""
        if not self.validation_log:
            print("No validations performed")
            return
        
        total = len(self.validation_log)
        locked = sum(1 for v in self.validation_log if v['locked'])
        failed = total - locked
        
        print("\n" + "="*70)
        print(f"AXIOM VALIDATION SUMMARY ({self.cartridge_id})")
        print("="*70)
        print(f"Total phantoms: {total}")
        print(f"Locked (passed): {locked}")
        print(f"Failed: {failed}")
        print(f"Pass rate: {locked/total*100:.1f}%")
        
        # Show failures
        failures = [v for v in self.validation_log if not v['locked']]
        if failures:
            print(f"\nFailures:")
            for fail in failures[:5]:  # Show top 5
                print(f"  {fail['phantom_id']}: {fail['rule_failures']}")
            if len(failures) > 5:
                print(f"  ... and {len(failures)-5} more")
        
        print("="*70 + "\n")
