"""
Ternary Crush - Phantom to Grain Compression

Compress validated phantoms to ternary grain representation.

Strategy:
- Extract top 5 derivations from fact annotations
- Map to ternary relationships {-1, 0, 1}
- Build pointer map for O(1) lookup
- Calculate 1.58-bit weight encoding

Author: Kitbash Team
Date: February 2026
Phase: 2C Consolidation → Modularized (May 2026)
"""

import hashlib
import statistics
from typing import Dict, List, Any

from kitbash_cartridge import Cartridge
from .data_structures import GrainMetadata, TernaryDelta, PhantomCandidate


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
        
        # Get fact content (use first fact for text)
        first_fact_id = list(phantom.fact_ids)[0] if phantom.fact_ids else None
        if not first_fact_id:
            raise ValueError("Phantom has no fact IDs")
        
        fact_text = self.cartridge.get_fact(first_fact_id)
        if not fact_text:
            raise ValueError(f"Fact {first_fact_id} not found in cartridge")
        
        # Extract derivations from all facts' annotations
        all_derivations = []
        for fact_id in phantom.fact_ids:
            annotation = self.cartridge.annotations.get(fact_id)
            if annotation and hasattr(annotation, 'derivations'):
                all_derivations.extend(annotation.derivations)
        
        # Map to ternary
        ternary_delta = self._extract_ternary_delta(fact_text, all_derivations)
        
        # Build pointer map (fast lookup structure)
        pointer_map = self._build_pointer_map(phantom, ternary_delta)
        
        # Calculate weight (1.58-bit equivalent)
        weight = self._calculate_weight(ternary_delta)
        
        # Generate grain ID (deterministic from fact)
        grain_id = self._generate_grain_id(first_fact_id, self.cartridge_id)
        
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
            'fact_snippet': fact_text[:100] if fact_text else '',
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
            
            # Handle both Derivation objects and dicts
            if isinstance(deriv, dict):
                deriv_type = deriv.get('type', '').lower()
                target = deriv.get('target', '')
                description = deriv.get('description', '')
            else:
                # Derivation object
                deriv_type = deriv.type.lower() if hasattr(deriv, 'type') else ''
                target = deriv.target if hasattr(deriv, 'target') else None
                description = deriv.description if hasattr(deriv, 'description') else ''
            
            if not deriv_type:
                continue
            
            # Use target if available, otherwise use description
            value = target if target else description
            if not value:
                continue
            
            # Classify by type (check more specific patterns first)
            # Must check negation patterns BEFORE positive_dependency patterns
            # since "negative_dependency" contains "dependency"
            if any(x in deriv_type for x in ['negation', 'negative_', 'inverse', 'opposite', 'excludes', 'contradicts']):
                negative.append(str(value))
            elif any(x in deriv_type for x in ['dependency', 'requires', 'depends', 'implies', 'entails']):
                positive.append(str(value))
            elif any(x in deriv_type for x in ['independent', 'orthogonal', 'void', 'boundary']):
                void.append(str(value))
        
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
