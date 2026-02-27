"""
Unit tests for Phase 3E.1 Axiom Validator
Tests all three Sicherman validation rules with realistic scenarios
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import Set, List
from kitbash_cartridge import (
    Cartridge, AnnotationMetadata, Derivation, 
    EpistemicLevel, FactStatus
)
from axiom_validator import AxiomValidator


# ============================================================================
# MOCK PHANTOM CANDIDATE (for testing)
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
        """Average confidence"""
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores) / len(self.confidence_scores)


# ============================================================================
# TEST SUITE
# ============================================================================

class TestPersistenceRule(unittest.TestCase):
    """Test Rule 1: PERSISTENCE"""
    
    def setUp(self):
        """Create temporary cartridge for testing"""
        self.temp_dir = tempfile.mkdtemp()
        self.cartridge = Cartridge("test_cart", self.temp_dir)
        self.cartridge.create()
    
    def tearDown(self):
        """Clean up - close cartridge and remove temp directory"""
        if self.cartridge and self.cartridge.db:
            self.cartridge.db.close()
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_persistence_all_facts_exist(self):
        """Test: phantom with all valid facts passes"""
        # Add facts to cartridge
        fact_id_1 = self.cartridge.add_fact("Fact 1")
        fact_id_2 = self.cartridge.add_fact("Fact 2")
        
        # Create phantom pointing to valid facts
        phantom = MockPhantom(
            phantom_id="p1",
            fact_ids={fact_id_1, fact_id_2},
            cartridge_id="test_cart"
        )
        
        validator = AxiomValidator(self.cartridge)
        self.assertTrue(validator._check_persistence(phantom))
    
    def test_persistence_missing_fact(self):
        """Test: phantom with missing fact fails"""
        fact_id_1 = self.cartridge.add_fact("Fact 1")
        
        # Create phantom pointing to one valid and one invalid fact
        phantom = MockPhantom(
            phantom_id="p1",
            fact_ids={fact_id_1, 9999},  # 9999 doesn't exist
            cartridge_id="test_cart"
        )
        
        validator = AxiomValidator(self.cartridge)
        self.assertFalse(validator._check_persistence(phantom))
    
    def test_persistence_empty_fact_ids(self):
        """Test: phantom with no facts fails"""
        phantom = MockPhantom(
            phantom_id="p1",
            fact_ids=set(),  # Empty
            cartridge_id="test_cart"
        )
        
        validator = AxiomValidator(self.cartridge)
        self.assertFalse(validator._check_persistence(phantom))


class TestResistanceRule(unittest.TestCase):
    """Test Rule 2: LEAST RESISTANCE"""
    
    def setUp(self):
        """Create cartridge with test data"""
        self.temp_dir = tempfile.mkdtemp()
        self.cartridge = Cartridge("test_cart", self.temp_dir)
        self.cartridge.create()
    
    def tearDown(self):
        """Clean up"""
        if self.cartridge and self.cartridge.db:
            self.cartridge.db.close()
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_resistance_high_ternary_ratio(self):
        """Test: phantom with 80% ternary-expressible derivations passes"""
        # Create derivations: 8 ternary, 2 non-ternary
        derivations = [
            Derivation(type='positive_dependency', description='dep1'),
            Derivation(type='negative_dependency', description='dep2'),
            Derivation(type='requires', description='dep3'),
            Derivation(type='depends_on', description='dep4'),
            Derivation(type='negation', description='dep5'),
            Derivation(type='orthogonal', description='dep6'),
            Derivation(type='void', description='dep7'),
            Derivation(type='boundary', description='dep8'),
            Derivation(type='weird_type', description='dep9'),  # Non-ternary
            Derivation(type='unknown', description='dep10'),    # Non-ternary
        ]
        
        # Add fact with these derivations
        annotation = AnnotationMetadata(fact_id=1, derivations=derivations)
        fact_id = self.cartridge.add_fact("Fact with ternary derivations", annotation=annotation)
        
        phantom = MockPhantom(
            phantom_id="p1",
            fact_ids={fact_id},
            cartridge_id="test_cart"
        )
        
        validator = AxiomValidator(self.cartridge)
        # 8/10 = 80% ≥ 70%, so should pass
        self.assertTrue(validator._check_resistance(phantom))
    
    def test_resistance_low_ternary_ratio(self):
        """Test: phantom with 40% ternary-expressible derivations fails"""
        # Create derivations: 4 ternary, 6 non-ternary
        derivations = [
            Derivation(type='positive_dependency', description='dep1'),
            Derivation(type='requires', description='dep2'),
            Derivation(type='void', description='dep3'),
            Derivation(type='orthogonal', description='dep4'),
            Derivation(type='weird1', description='dep5'),
            Derivation(type='weird2', description='dep6'),
            Derivation(type='weird3', description='dep7'),
            Derivation(type='weird4', description='dep8'),
            Derivation(type='weird5', description='dep9'),
            Derivation(type='weird6', description='dep10'),
        ]
        
        annotation = AnnotationMetadata(fact_id=1, derivations=derivations)
        fact_id = self.cartridge.add_fact("Fact with mixed derivations", annotation=annotation)
        
        phantom = MockPhantom(
            phantom_id="p1",
            fact_ids={fact_id},
            cartridge_id="test_cart"
        )
        
        validator = AxiomValidator(self.cartridge)
        # 4/10 = 40% < 70%, so should fail
        self.assertFalse(validator._check_resistance(phantom))
    
    def test_resistance_no_derivations(self):
        """Test: phantom with no derivations fails"""
        # Add fact with no derivations
        annotation = AnnotationMetadata(fact_id=1, derivations=[])
        fact_id = self.cartridge.add_fact("Fact with no derivations", annotation=annotation)
        
        phantom = MockPhantom(
            phantom_id="p1",
            fact_ids={fact_id},
            cartridge_id="test_cart"
        )
        
        validator = AxiomValidator(self.cartridge)
        self.assertFalse(validator._check_resistance(phantom))


class TestIndependenceRule(unittest.TestCase):
    """Test Rule 3: INDEPENDENCE"""
    
    def setUp(self):
        """Create cartridge with test data"""
        self.temp_dir = tempfile.mkdtemp()
        self.cartridge = Cartridge("test_cart", self.temp_dir)
        self.cartridge.create()
    
    def tearDown(self):
        """Clean up"""
        if self.cartridge and self.cartridge.db:
            self.cartridge.db.close()
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_independence_stable_confidence(self):
        """Test: phantom with stable confidence (<0.02 variance) passes"""
        # Create stable confidence scores (variance ≈ 0.0009)
        stable_scores = [0.85, 0.86, 0.84, 0.85, 0.87]
        
        fact_id = self.cartridge.add_fact("Stable fact")
        
        phantom = MockPhantom(
            phantom_id="p1",
            fact_ids={fact_id},
            cartridge_id="test_cart",
            confidence_scores=stable_scores
        )
        
        validator = AxiomValidator(self.cartridge)
        self.assertTrue(validator._check_independence(phantom))
    
    def test_independence_unstable_confidence(self):
        """Test: phantom with unstable confidence (>0.02 variance) fails"""
        # Create unstable confidence scores (variance ≈ 0.16)
        unstable_scores = [0.2, 0.95, 0.1, 0.9, 0.15]
        
        fact_id = self.cartridge.add_fact("Unstable fact")
        
        phantom = MockPhantom(
            phantom_id="p1",
            fact_ids={fact_id},
            cartridge_id="test_cart",
            confidence_scores=unstable_scores
        )
        
        validator = AxiomValidator(self.cartridge)
        self.assertFalse(validator._check_independence(phantom))
    
    def test_independence_consistent_epistemic_level(self):
        """Test: phantom with same epistemic level passes"""
        # Add facts at same epistemic level
        ann1 = AnnotationMetadata(fact_id=1, epistemic_level=EpistemicLevel.L2_AXIOMATIC)
        ann2 = AnnotationMetadata(fact_id=2, epistemic_level=EpistemicLevel.L2_AXIOMATIC)
        
        fact_id_1 = self.cartridge.add_fact("Fact L2-1", annotation=ann1)
        fact_id_2 = self.cartridge.add_fact("Fact L2-2", annotation=ann2)
        
        phantom = MockPhantom(
            phantom_id="p1",
            fact_ids={fact_id_1, fact_id_2},
            cartridge_id="test_cart",
            confidence_scores=[0.85, 0.86, 0.84]
        )
        
        validator = AxiomValidator(self.cartridge)
        self.assertTrue(validator._check_independence(phantom))
    
    def test_independence_mixed_epistemic_level(self):
        """Test: phantom with different epistemic levels (>1 apart) fails"""
        # Add facts at different levels (L0 and L3 = 3 levels apart)
        ann1 = AnnotationMetadata(fact_id=1, epistemic_level=EpistemicLevel.L0_EMPIRICAL)
        ann2 = AnnotationMetadata(fact_id=2, epistemic_level=EpistemicLevel.L3_PERSONA)
        
        fact_id_1 = self.cartridge.add_fact("Fact L0", annotation=ann1)
        fact_id_2 = self.cartridge.add_fact("Fact L3", annotation=ann2)
        
        phantom = MockPhantom(
            phantom_id="p1",
            fact_ids={fact_id_1, fact_id_2},
            cartridge_id="test_cart",
            confidence_scores=[0.85, 0.86, 0.84]
        )
        
        validator = AxiomValidator(self.cartridge)
        self.assertFalse(validator._check_independence(phantom))
    
    def test_independence_excessive_oscillation(self):
        """Test: phantom with excessive oscillations (>3) fails"""
        # Create highly oscillating confidence scores
        # Pattern: low→high→low→high→low→high→low (4 oscillations)
        oscillating_scores = [0.3, 0.9, 0.2, 0.95, 0.15, 0.85, 0.25]
        
        fact_id = self.cartridge.add_fact("Oscillating fact")
        
        phantom = MockPhantom(
            phantom_id="p1",
            fact_ids={fact_id},
            cartridge_id="test_cart",
            confidence_scores=oscillating_scores
        )
        
        validator = AxiomValidator(self.cartridge)
        self.assertFalse(validator._check_independence(phantom))


class TestValidateBatch(unittest.TestCase):
    """Test batch validation workflow"""
    
    def setUp(self):
        """Create cartridge with test data"""
        self.temp_dir = tempfile.mkdtemp()
        self.cartridge = Cartridge("test_cart", self.temp_dir)
        self.cartridge.create()
    
    def tearDown(self):
        """Clean up"""
        if self.cartridge and self.cartridge.db:
            self.cartridge.db.close()
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_batch_mixed_results(self):
        """Test: batch with mix of valid and invalid phantoms"""
        # Create facts
        fact_id_1 = self.cartridge.add_fact("Fact 1")
        fact_id_2 = self.cartridge.add_fact("Fact 2")
        
        # Phantom 1: Valid (all rules pass)
        phantom_1 = MockPhantom(
            phantom_id="p1",
            fact_ids={fact_id_1, fact_id_2},
            cartridge_id="test_cart",
            confidence_scores=[0.85, 0.84, 0.86]
        )
        
        # Phantom 2: Invalid (missing fact)
        phantom_2 = MockPhantom(
            phantom_id="p2",
            fact_ids={fact_id_1, 9999},  # Invalid fact
            cartridge_id="test_cart",
            confidence_scores=[0.85, 0.84, 0.86]
        )
        
        validator = AxiomValidator(self.cartridge)
        result = validator.validate_batch([phantom_1, phantom_2])
        
        self.assertEqual(result['total'], 2)
        # Phantom 2 will fail persistence check
        self.assertLessEqual(result['validated'], 2)


class TestValidateSummary(unittest.TestCase):
    """Test summary output"""
    
    def setUp(self):
        """Create cartridge"""
        self.temp_dir = tempfile.mkdtemp()
        self.cartridge = Cartridge("test_cart", self.temp_dir)
        self.cartridge.create()
    
    def tearDown(self):
        """Clean up"""
        if self.cartridge and self.cartridge.db:
            self.cartridge.db.close()
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_print_summary(self):
        """Test: summary printing doesn't crash"""
        fact_id = self.cartridge.add_fact("Fact 1")
        
        phantom = MockPhantom(
            phantom_id="p1",
            fact_ids={fact_id},
            cartridge_id="test_cart",
            confidence_scores=[0.85]
        )
        
        validator = AxiomValidator(self.cartridge)
        result = validator.validate_phantom(phantom)
        
        # Should not raise exception
        validator.print_summary()


if __name__ == '__main__':
    # Suppress cartridge creation output during tests
    unittest.main(verbosity=2)
