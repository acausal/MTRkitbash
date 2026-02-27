#!/usr/bin/env python3
"""
Tests for Stage 1: Sleep Consolidation

Tests collision detection, fact aggregation, violation analysis.
Verifies data accuracy and index integrity.
"""

import tempfile
import json
from pathlib import Path
from datetime import datetime

from sleep_consolidator import DreamBucketConsolidator
from sleep_orchestrator import SleepOrchestrator
from dream_bucket import DreamBucketWriter


def test_collision_detection():
    """Test collision pair detection."""
    print("\n[TEST] Collision Detection")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        consolidator = DreamBucketConsolidator(tmpdir)
        
        # Create test false positives
        false_positives = [
            {
                'timestamp': '2026-02-27T10:00:00Z',
                'query_text': 'photosynthesis',
                'returned_id': 42,
                'correct_id': 137,
                'returned_confidence': 0.85,
                'error_signal': 0.40,
                'source_layer': 'cartridge',
            },
            {
                'timestamp': '2026-02-27T10:01:00Z',
                'query_text': 'plant energy',
                'returned_id': 42,
                'correct_id': 137,
                'returned_confidence': 0.82,
                'error_signal': 0.38,
                'source_layer': 'cartridge',
            },
            {
                'timestamp': '2026-02-27T10:02:00Z',
                'query_text': 'cellular respiration',
                'returned_id': 89,
                'correct_id': 203,
                'returned_confidence': 0.80,
                'error_signal': 0.35,
                'source_layer': 'grain',
            },
        ]
        
        # Detect collisions
        result = consolidator.detect_collisions(false_positives)
        
        # Verify results
        assert 'collision_index' in result
        assert len(result['collision_index']) == 2, f"Expected 2 collisions, got {len(result['collision_index'])}"
        
        # Check first collision
        collision_42_137 = result['collision_index'].get('(42, 137)')
        assert collision_42_137 is not None, "Collision (42, 137) not found"
        assert collision_42_137['collision_count'] == 2
        assert 'photosynthesis' in collision_42_137['query_patterns']
        assert 'plant energy' in collision_42_137['query_patterns']
        assert 'cartridge' in collision_42_137['source_layers']
        
        # Check statistics
        assert collision_42_137['avg_confidence_on_collision'] > 0
        assert collision_42_137['error_signal_stats']['min'] <= 0.40
        
        print("  ✓ Collision detection working")
        print(f"    - Detected {len(result['collision_index'])} collision pairs")
        print(f"    - Tracked query patterns correctly")
        print(f"    - Calculated statistics correctly")


def test_fact_aggregation():
    """Test false positive aggregation by fact."""
    print("\n[TEST] Fact Aggregation")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        consolidator = DreamBucketConsolidator(tmpdir)
        
        # Create test false positives
        false_positives = [
            {
                'query_text': 'photosynthesis',
                'returned_id': 42,
                'correct_id': 137,
                'returned_confidence': 0.85,
                'error_signal': 0.40,
                'source_layer': 'cartridge',
            },
            {
                'query_text': 'energy conversion',
                'returned_id': 42,
                'correct_id': 203,
                'returned_confidence': 0.88,
                'error_signal': 0.35,
                'source_layer': 'cartridge',
            },
            {
                'query_text': 'plant biology',
                'returned_id': 42,
                'correct_id': 137,
                'returned_confidence': 0.80,
                'error_signal': 0.38,
                'source_layer': 'grain',
            },
        ]
        
        # Aggregate by fact
        result = consolidator.aggregate_by_fact(false_positives)
        
        # Verify results
        assert '42' in result, "Fact 42 not in aggregation"
        fact_42 = result['42']
        
        assert fact_42['fp_count'] == 3
        assert fact_42['total_uses'] == 3
        fp_rate = fact_42['fp_rate']
        assert 0.99 <= fp_rate <= 1.01, f"FP rate should be ~1.0, got {fp_rate}"
        
        # Check confused pairs
        assert 137 in fact_42['most_confused_with']
        assert 203 in fact_42['most_confused_with']
        
        # Check query patterns tracked
        assert 'photosynthesis' in fact_42['query_patterns']
        assert 'energy conversion' in fact_42['query_patterns']
        assert 'plant biology' in fact_42['query_patterns']
        
        # Check statistics
        assert fact_42['avg_confidence'] > 0
        assert fact_42['avg_error_signal'] > 0
        
        print("  ✓ Fact aggregation working")
        print(f"    - Aggregated {len(result)} facts")
        print(f"    - Tracked query patterns correctly")
        print(f"    - Calculated FP rates correctly")


def test_violation_analysis():
    """Test violation timeline analysis."""
    print("\n[TEST] Violation Analysis")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        consolidator = DreamBucketConsolidator(tmpdir)
        
        # Create test violations
        violations = [
            {
                'timestamp': '2026-02-27T10:00:00Z',
                'returned_fact_id': 42,
                'returned_confidence': 0.85,
                'mtr_error_signal': 0.50,
                'dissonance_type': 'high_confidence_low_coherence',
            },
            {
                'timestamp': '2026-02-27T10:01:00Z',
                'returned_fact_id': 42,
                'returned_confidence': 0.88,
                'mtr_error_signal': 0.55,
                'dissonance_type': 'context_switch_failure',
            },
            {
                'timestamp': '2026-02-27T10:02:00Z',
                'returned_fact_id': 89,
                'returned_confidence': 0.80,
                'mtr_error_signal': 0.48,
                'dissonance_type': 'high_confidence_low_coherence',
            },
        ]
        
        # Analyze violations
        result = consolidator.analyze_violations(violations)
        
        # Verify results
        assert '42' in result, "Fact 42 not in violations"
        fact_42 = result['42']
        
        assert fact_42['total_violations'] == 2
        assert fact_42['dissonance_types']['high_confidence_low_coherence'] == 1
        assert fact_42['dissonance_types']['context_switch_failure'] == 1
        assert fact_42['avg_error_signal'] > 0
        
        # Check fact 89
        assert '89' in result
        fact_89 = result['89']
        assert fact_89['total_violations'] == 1
        
        print("  ✓ Violation analysis working")
        print(f"    - Analyzed {len(result)} facts")
        print(f"    - Tracked dissonance types correctly")
        print(f"    - Calculated error statistics correctly")


def test_empty_logs():
    """Test handling of empty logs."""
    print("\n[TEST] Empty Logs Handling")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        consolidator = DreamBucketConsolidator(tmpdir)
        
        # Test with empty lists
        empty_fps = []
        empty_violations = []
        
        # Should not crash
        collision_result = consolidator.detect_collisions(empty_fps)
        assert collision_result['total_collisions'] == 0
        
        agg_result = consolidator.aggregate_by_fact(empty_fps)
        assert len(agg_result) == 0
        
        vio_result = consolidator.analyze_violations(empty_violations)
        assert len(vio_result) == 0
        
        print("  ✓ Empty logs handled correctly")


def test_malformed_records():
    """Test handling of malformed records."""
    print("\n[TEST] Malformed Records Handling")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        consolidator = DreamBucketConsolidator(tmpdir)
        
        # Mix good and bad records
        mixed_fps = [
            {
                'query_text': 'test',
                'returned_id': 42,
                'correct_id': 137,
                'returned_confidence': 0.85,
                'error_signal': 0.40,
            },
            {
                # Missing returned_id
                'query_text': 'bad',
                'correct_id': 137,
            },
            {
                # Missing correct_id
                'query_text': 'bad2',
                'returned_id': 42,
            },
            {
                'query_text': 'test2',
                'returned_id': 42,
                'correct_id': 137,
                'returned_confidence': 0.88,
                'error_signal': 0.35,
            },
        ]
        
        # Should process valid records and skip malformed ones
        result = consolidator.detect_collisions(mixed_fps)
        
        # Should have 1 collision pair from 2 valid records
        assert len(result['collision_index']) == 1
        assert result['total_collisions'] == 2
        
        print("  ✓ Malformed records skipped gracefully")
        print(f"    - Processed 2/4 records (skipped 2 malformed)")


def test_full_consolidation():
    """Test full consolidation workflow."""
    print("\n[TEST] Full Consolidation Workflow")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write test logs
        writer = DreamBucketWriter(tmpdir)
        
        false_positives = [
            {
                'timestamp': '2026-02-27T10:00:00Z',
                'query_text': 'photosynthesis',
                'returned_id': 42,
                'correct_id': 137,
                'returned_confidence': 0.85,
                'error_signal': 0.40,
                'source_layer': 'cartridge',
                'session_id': 'test_session_001',
            },
            {
                'timestamp': '2026-02-27T10:01:00Z',
                'query_text': 'plant energy',
                'returned_id': 42,
                'correct_id': 137,
                'returned_confidence': 0.82,
                'error_signal': 0.38,
                'source_layer': 'cartridge',
                'session_id': 'test_session_001',
            },
        ]
        
        violations = [
            {
                'timestamp': '2026-02-27T10:00:30Z',
                'returned_fact_id': 42,
                'returned_confidence': 0.85,
                'mtr_error_signal': 0.50,
                'dissonance_type': 'high_confidence_low_coherence',
                'session_id': 'test_session_001',
            },
        ]
        
        # Write logs
        for fp in false_positives:
            writer.append('false_positives', fp)
        for vio in violations:
            writer.append('violations', vio)
        
        # Wait for background writer
        import time
        time.sleep(0.5)
        
        # Consolidate
        consolidator = DreamBucketConsolidator(tmpdir)
        report = consolidator.consolidate_all()
        
        # Verify report
        assert report['false_positives_processed'] == 2
        assert report['violations_processed'] == 1
        assert report['unique_collisions'] > 0
        assert report['facts_with_fp'] > 0
        assert report['facts_with_violations'] > 0
        
        print("  ✓ Full consolidation working")
        print(f"    - Processed {report['false_positives_processed']} FPs")
        print(f"    - Processed {report['violations_processed']} violations")
        print(f"    - Generated {report['unique_collisions']} collision indices")


def test_sleep_orchestrator():
    """Test sleep orchestrator Stage 1."""
    print("\n[TEST] Sleep Orchestrator Stage 1")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write test logs
        writer = DreamBucketWriter(tmpdir)
        
        false_positives = [
            {
                'timestamp': '2026-02-27T10:00:00Z',
                'query_text': 'test',
                'returned_id': 42,
                'correct_id': 137,
                'returned_confidence': 0.85,
                'error_signal': 0.40,
                'source_layer': 'cartridge',
                'session_id': 'test_session_001',
            },
        ]
        
        for fp in false_positives:
            writer.append('false_positives', fp)
        
        import time
        time.sleep(0.5)
        
        # Run orchestrator
        orchestrator = SleepOrchestrator(tmpdir)
        report = orchestrator.run_stage_1()
        
        # Verify report
        assert report['false_positives_processed'] == 1
        
        print("  ✓ Sleep orchestrator Stage 1 working")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("STAGE 1 CONSOLIDATION TESTS")
    print("="*70)
    
    try:
        test_collision_detection()
        test_fact_aggregation()
        test_violation_analysis()
        test_empty_logs()
        test_malformed_records()
        test_full_consolidation()
        test_sleep_orchestrator()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED")
        print("="*70 + "\n")
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
