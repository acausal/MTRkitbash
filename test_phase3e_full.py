"""
Phase 3E Integration Tests - Full Pipeline Validation

Tests the complete flow:
  Validation (3E.1) → Crushing (3E.2) → Caching (3E.3)

Run with: python test_phase3e_full.py
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import Set, List

from kitbash_cartridge import (
    Cartridge, AnnotationMetadata, Derivation, EpistemicLevel
)
from axiom_validator import AxiomValidator
from grain_system import PhantomCandidate, TernaryCrush
from grain_activation import GrainActivation


# ============================================================================
# MOCK PHANTOM (same as in test_axiom_validator.py)
# ============================================================================

@dataclass
class MockPhantom:
    """Mock PhantomCandidate for testing"""
    phantom_id: str
    fact_ids: Set[int]
    cartridge_id: str
    hit_count: int = 10
    confidence_scores: List[float] = field(default_factory=lambda: [0.85])
    query_concepts: List[str] = field(default_factory=list)
    first_cycle_seen: int = 0
    last_cycle_seen: int = 50
    cycle_consistency: float = 0.98
    status: str = "locked"
    
    def avg_confidence(self) -> float:
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores) / len(self.confidence_scores)


# ============================================================================
# TEST SUITE
# ============================================================================

class TestPhase3E1Validation(unittest.TestCase):
    """Test Phase 3E.1: Validation"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.cartridge = Cartridge("test_cart", self.temp_dir)
        self.cartridge.create()
    
    def tearDown(self):
        if self.cartridge and self.cartridge.db:
            self.cartridge.db.close()
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_validate_valid_phantom(self):
        """Test: Valid phantom passes all validation rules"""
        # Create facts with good derivations
        derivations = [
            Derivation(type='positive_dependency', description='dep1'),
            Derivation(type='negative_dependency', description='neg1'),
            Derivation(type='void', description='void1'),
        ]
        ann = AnnotationMetadata(fact_id=1, derivations=derivations)
        fact_id = self.cartridge.add_fact("Test fact", annotation=ann)
        
        phantom = MockPhantom(
            phantom_id="p1",
            fact_ids={fact_id},
            cartridge_id="test_cart",
            confidence_scores=[0.85, 0.84, 0.86]
        )
        
        validator = AxiomValidator(self.cartridge)
        result = validator.validate_phantom(phantom)
        
        # Should pass all three rules
        self.assertTrue(result['persistent_check'])
        self.assertTrue(result['resistance_check'])
        self.assertTrue(result['independence_check'])
        self.assertTrue(result['locked'])
    
    def test_validate_invalid_phantom_persistence(self):
        """Test: Phantom with missing fact fails persistence"""
        phantom = MockPhantom(
            phantom_id="p1",
            fact_ids={9999},  # Non-existent
            cartridge_id="test_cart"
        )
        
        validator = AxiomValidator(self.cartridge)
        result = validator.validate_phantom(phantom)
        
        self.assertFalse(result['persistent_check'])
        self.assertFalse(result['locked'])


class TestPhase3E2Crushing(unittest.TestCase):
    """Test Phase 3E.2: Crushing"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.cartridge = Cartridge("test_cart", self.temp_dir)
        self.cartridge.create()
    
    def tearDown(self):
        if self.cartridge and self.cartridge.db:
            self.cartridge.db.close()
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_crush_valid_phantom(self):
        """Test: Valid phantom crushes to grain with ternary delta"""
        # Create fact with derivations
        derivations = [
            Derivation(type='positive_dependency', description='depends on X'),
            Derivation(type='negative_dependency', description='conflicts with Y'),
            Derivation(type='void', description='orthogonal to Z'),
        ]
        ann = AnnotationMetadata(fact_id=1, derivations=derivations)
        fact_id = self.cartridge.add_fact("Test fact", annotation=ann)
        
        phantom = MockPhantom(
            phantom_id="p1",
            fact_ids={fact_id},
            cartridge_id="test_cart"
        )
        
        crusher = TernaryCrush(self.cartridge)
        validation_result = {
            'locked': True,
            'confidence': 0.85,
            'cycles_locked': 50
        }
        
        grain = crusher.crush_phantom(phantom, validation_result)
        
        # Verify grain structure
        self.assertIn('grain_id', grain)
        self.assertIn('delta', grain)
        self.assertTrue(len(grain['delta']['positive']) > 0)
        self.assertTrue(len(grain['delta']['negative']) > 0)
        self.assertTrue(len(grain['delta']['void']) > 0)
        self.assertGreater(grain['weight'], 0)
        self.assertEqual(grain['confidence'], 0.85)
    
    def test_crush_multi_fact_phantom(self):
        """Test: Phantom with multiple facts crushes correctly"""
        # Create two facts with derivations
        derivations1 = [
            Derivation(type='positive_dependency', description='dep1'),
        ]
        derivations2 = [
            Derivation(type='negative_dependency', description='neg1'),
        ]
        
        ann1 = AnnotationMetadata(fact_id=1, derivations=derivations1)
        ann2 = AnnotationMetadata(fact_id=2, derivations=derivations2)
        
        fact_id_1 = self.cartridge.add_fact("Fact 1", annotation=ann1)
        fact_id_2 = self.cartridge.add_fact("Fact 2", annotation=ann2)
        
        phantom = MockPhantom(
            phantom_id="p1",
            fact_ids={fact_id_1, fact_id_2},
            cartridge_id="test_cart"
        )
        
        crusher = TernaryCrush(self.cartridge)
        validation_result = {'locked': True, 'confidence': 0.80, 'cycles_locked': 50}
        
        grain = crusher.crush_phantom(phantom, validation_result)
        
        # Both facts' derivations should be in grain
        self.assertTrue(len(grain['delta']['positive']) > 0 or len(grain['delta']['negative']) > 0)


class TestPhase3E3Caching(unittest.TestCase):
    """Test Phase 3E.3: Caching"""
    
    def test_cache_load_grain(self):
        """Test: Grain loads into cache"""
        cache = GrainActivation(max_cache_mb=1.0)
        
        grain = {
            'grain_id': 'sg_test1',
            'delta': {'positive': ['dep1'], 'negative': ['neg1'], 'void': ['void1']},
            'weight': 4.75
        }
        
        success = cache.load_grain(grain)
        
        self.assertTrue(success)
        self.assertEqual(len(cache.cache), 1)
    
    def test_cache_lookup_hit(self):
        """Test: Cache hit on loaded grain"""
        cache = GrainActivation(max_cache_mb=1.0)
        
        grain = {
            'grain_id': 'sg_test1',
            'delta': {'positive': ['dep1']},
            'weight': 4.75
        }
        cache.load_grain(grain)
        
        result = cache.lookup('sg_test1')
        
        self.assertIsNotNone(result)
        self.assertEqual(result['grain_id'], 'sg_test1')
        self.assertEqual(cache.stats['hits'], 1)
        self.assertEqual(cache.stats['misses'], 0)
    
    def test_cache_lookup_miss(self):
        """Test: Cache miss on non-existent grain"""
        cache = GrainActivation(max_cache_mb=1.0)
        
        result = cache.lookup('sg_nonexistent')
        
        self.assertIsNone(result)
        self.assertEqual(cache.stats['hits'], 0)
        self.assertEqual(cache.stats['misses'], 1)
    
    def test_cache_capacity(self):
        """Test: Cache respects size limit"""
        cache = GrainActivation(max_cache_mb=0.001)  # Very small: ~1KB
        
        # Try to load many grains
        for i in range(100):
            grain = {
                'grain_id': f'sg_{i:08x}',
                'delta': {'positive': ['x']*10, 'negative': ['y']*10, 'void': ['z']*10},
                'weight': 4.75
            }
            cache.load_grain(grain)
        
        # Should have some evictions
        self.assertGreater(cache.stats['evictions'], 0)
        self.assertLess(len(cache.cache), 100)
    
    def test_cache_statistics(self):
        """Test: Cache statistics tracked correctly"""
        cache = GrainActivation(max_cache_mb=1.0)
        
        # Load grain
        grain = {'grain_id': 'sg_1', 'delta': {}, 'weight': 1.0}
        cache.load_grain(grain)
        
        # Hit
        cache.lookup('sg_1')
        
        # Miss
        cache.lookup('sg_999')
        
        stats = cache.get_stats()
        
        self.assertEqual(stats['total_grains'], 1)
        self.assertEqual(stats['hits'], 1)
        self.assertEqual(stats['misses'], 1)
        self.assertEqual(stats['total_lookups'], 2)
        self.assertAlmostEqual(stats['hit_rate'], 0.5, places=2)


class TestPhase3EFullPipeline(unittest.TestCase):
    """Test complete Phase 3E pipeline: Validation → Crushing → Caching"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.cartridge = Cartridge("test_cart", self.temp_dir)
        self.cartridge.create()
    
    def tearDown(self):
        if self.cartridge and self.cartridge.db:
            self.cartridge.db.close()
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_full_pipeline(self):
        """Test: Full pipeline from phantom to cached grain"""
        print("\n" + "="*70)
        print("PHASE 3E FULL PIPELINE TEST")
        print("="*70)
        
        # Step 1: Create cartridge with facts
        print("\n1. Creating facts in cartridge...")
        derivations = [
            Derivation(type='positive_dependency', description='depends on X'),
            Derivation(type='negative_dependency', description='conflicts with Y'),
            Derivation(type='void', description='orthogonal to Z'),
        ]
        ann = AnnotationMetadata(fact_id=1, derivations=derivations)
        fact_id = self.cartridge.add_fact("Test fact", annotation=ann)
        print(f"   ✓ Created fact {fact_id}")
        
        # Step 2: Create and validate phantom
        print("\n2. Validating phantom...")
        phantom = MockPhantom(
            phantom_id="p1",
            fact_ids={fact_id},
            cartridge_id="test_cart",
            confidence_scores=[0.85, 0.84, 0.86]
        )
        
        validator = AxiomValidator(self.cartridge)
        validation_result = validator.validate_phantom(phantom)
        print(f"   ✓ Persistent: {validation_result['persistent_check']}")
        print(f"   ✓ Resistance: {validation_result['resistance_check']}")
        print(f"   ✓ Independence: {validation_result['independence_check']}")
        print(f"   ✓ Locked: {validation_result['locked']}")
        
        self.assertTrue(validation_result['locked'])
        
        # Step 3: Crush to grain
        print("\n3. Crushing to ternary grain...")
        crusher = TernaryCrush(self.cartridge)
        grain = crusher.crush_phantom(phantom, validation_result)
        print(f"   ✓ Grain ID: {grain['grain_id']}")
        print(f"   ✓ Positive: {len(grain['delta']['positive'])} items")
        print(f"   ✓ Negative: {len(grain['delta']['negative'])} items")
        print(f"   ✓ Void: {len(grain['delta']['void'])} items")
        print(f"   ✓ Weight: {grain['weight']:.2f}")
        
        # Step 4: Activate to cache
        print("\n4. Activating grain to L3 cache...")
        cache = GrainActivation(max_cache_mb=1.0)
        success = cache.load_grain(grain)
        print(f"   ✓ Activated: {success}")
        
        self.assertTrue(success)
        
        # Step 5: Test cache lookups
        print("\n5. Testing cache lookups...")
        cached_grain = cache.lookup(grain['grain_id'])
        self.assertIsNotNone(cached_grain)
        print(f"   ✓ Cache hit on {grain['grain_id']}")
        
        # Step 6: Print stats
        print("\n6. Cache statistics:")
        stats = cache.get_stats()
        print(f"   ✓ Cached grains: {stats['total_grains']}")
        print(f"   ✓ Cache usage: {stats['cache_mb']:.2f} MB")
        print(f"   ✓ Hit rate: {stats['hit_rate']*100:.1f}%")
        
        print("\n" + "="*70)
        print("✅ PHASE 3E FULL PIPELINE TEST PASSED")
        print("="*70 + "\n")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
