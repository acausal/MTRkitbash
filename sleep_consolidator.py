#!/usr/bin/env python3
"""
Sleep Consolidator - Stage 1: Log Consolidation

Converts raw JSONL logs into queryable JSON indices.

Input:
  dream_bucket/live/false_positives.jsonl
  dream_bucket/live/violations.jsonl

Output:
  dream_bucket/indices/collision_index.json
  dream_bucket/indices/false_positive_by_grain.json
  dream_bucket/indices/violation_timeline.json

Time: 5-10 minutes for typical volume
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from collections import defaultdict
from datetime import datetime
import json

from dream_bucket import DreamBucketReader, DreamBucketWriter


class DreamBucketConsolidator:
    """
    Consolidate raw dream bucket signals into aggregated indices.
    
    Reads JSONL logs from live session, aggregates statistics,
    and writes JSON indices for downstream analysis.
    """
    
    def __init__(self, dream_bucket_dir: str):
        """
        Initialize consolidator with dream bucket paths.
        
        Args:
            dream_bucket_dir: Path to dream_bucket directory root
        """
        self.reader = DreamBucketReader(dream_bucket_dir)
        self.writer = DreamBucketWriter(dream_bucket_dir)
    
    def consolidate_all(self) -> Dict[str, Any]:
        """
        Run full Stage 1 consolidation: read logs, aggregate, write indices.
        
        Returns:
            Report dict with processing statistics
        """
        print("\n[Stage 1] Consolidating raw logs into indices...")
        
        # Read all logs
        print("  Reading false positives...")
        false_positives = list(self.reader.read_live_log('false_positives'))
        print(f"    → {len(false_positives)} records")
        
        print("  Reading violations...")
        violations = list(self.reader.read_live_log('violations'))
        print(f"    → {len(violations)} records")
        
        # Process each index type
        report = {
            'false_positives_processed': len(false_positives),
            'violations_processed': len(violations),
            'unique_collisions': 0,
            'facts_with_fp': 0,
            'facts_with_violations': 0,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
        }
        
        # Consolidate false positives
        if false_positives:
            print("  Detecting collisions...")
            collision_index = self.detect_collisions(false_positives)
            report['unique_collisions'] = len(collision_index.get('collision_index', {}))
            
            print(f"    → {report['unique_collisions']} unique collision pairs")
            print("  Writing collision_index.json...")
            self.writer.write_index('collision_index', collision_index)
            
            print("  Aggregating by fact...")
            fp_by_grain = self.aggregate_by_fact(false_positives)
            report['facts_with_fp'] = len(fp_by_grain)
            print(f"    → {report['facts_with_fp']} facts with false positives")
            print("  Writing false_positive_by_grain.json...")
            self.writer.write_index('false_positive_by_grain', fp_by_grain)
        
        # Consolidate violations
        if violations:
            print("  Analyzing violations...")
            violation_stats = self.analyze_violations(violations)
            report['facts_with_violations'] = len(violation_stats)
            print(f"    → {report['facts_with_violations']} facts with violations")
            print("  Writing violation_timeline.json...")
            self.writer.write_index('violation_timeline', violation_stats)
        
        print("\n✓ Stage 1 complete\n")
        return report
    
    def detect_collisions(self, false_positives: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Group false positives by collision pairs.
        
        A collision is when two facts get confused: fact_A returns high confidence
        but MTR error indicates fact_B was correct.
        
        Args:
            false_positives: List of false positive events from JSONL
        
        Returns:
            collision_index dict with aggregated stats
        """
        collisions: Dict[Tuple[int, int], Dict[str, Any]] = defaultdict(
            lambda: {
                'collision_count': 0,
                'query_patterns': set(),
                'source_layers': set(),
                'error_signals': [],
                'first_observed': None,
                'last_observed': None,
                'avg_confidence_on_collision': 0.0,
            }
        )
        
        for fp in false_positives:
            returned_id = fp.get('returned_id')
            correct_id = fp.get('correct_id')
            
            # Only count if we have both IDs
            if returned_id is None or correct_id is None:
                continue
            
            # Create collision key (ordered pair)
            collision_key = tuple(sorted([returned_id, correct_id]))
            
            collision = collisions[collision_key]
            collision['collision_count'] += 1
            
            # Track query patterns
            query = fp.get('query_text', '')
            if query:
                collision['query_patterns'].add(query)
            
            # Track source layers
            source = fp.get('source_layer', '')
            if source:
                collision['source_layers'].add(source)
            
            # Track error signals
            error = fp.get('error_signal', 0.0)
            collision['error_signals'].append(error)
            
            # Track timestamps
            timestamp = fp.get('timestamp')
            if timestamp:
                if collision['first_observed'] is None:
                    collision['first_observed'] = timestamp
                collision['last_observed'] = timestamp
        
        # Convert to serializable format
        collision_index = {}
        for (id1, id2), stats in collisions.items():
            key = f"({id1}, {id2})"
            
            error_signals = stats['error_signals']
            collision_index[key] = {
                'collision_count': stats['collision_count'],
                'query_patterns': sorted(list(stats['query_patterns'])),
                'source_layers': sorted(list(stats['source_layers'])),
                'avg_confidence_on_collision': sum(error_signals) / len(error_signals) if error_signals else 0.0,
                'first_observed': stats['first_observed'],
                'last_observed': stats['last_observed'],
                'error_signal_stats': {
                    'mean': sum(error_signals) / len(error_signals) if error_signals else 0.0,
                    'min': min(error_signals) if error_signals else 0.0,
                    'max': max(error_signals) if error_signals else 0.0,
                    'count': len(error_signals),
                }
            }
        
        return {
            'collision_index': collision_index,
            'total_collisions': sum(c['collision_count'] for c in collision_index.values()),
            'total_unique_pairs': len(collision_index),
        }
    
    def aggregate_by_fact(self, false_positives: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate false positive statistics per fact.
        
        For each fact that's returned incorrectly, track:
        - How often it appears
        - What it gets confused with
        - Confidence/error patterns
        
        Args:
            false_positives: List of false positive events
        
        Returns:
            Dict mapping fact_id -> statistics
        """
        facts: Dict[int, Dict[str, Any]] = defaultdict(
            lambda: {
                'fp_count': 0,
                'total_uses': 0,
                'confused_with': defaultdict(int),
                'query_patterns': set(),
                'confidences': [],
                'error_signals': [],
            }
        )
        
        for fp in false_positives:
            returned_id = fp.get('returned_id')
            if returned_id is None:
                continue
            
            fact = facts[returned_id]
            fact['fp_count'] += 1
            fact['total_uses'] += 1
            
            # Track what it gets confused with
            correct_id = fp.get('correct_id')
            if correct_id is not None:
                fact['confused_with'][correct_id] += 1
            
            # Track query patterns
            query = fp.get('query_text', '')
            if query:
                fact['query_patterns'].add(query)
            
            # Track confidence and error
            confidence = fp.get('returned_confidence', 0.0)
            fact['confidences'].append(confidence)
            
            error = fp.get('error_signal', 0.0)
            fact['error_signals'].append(error)
        
        # Convert to serializable format
        fp_by_grain = {}
        for fact_id, stats in facts.items():
            confused_with = stats['confused_with']
            top_confused = sorted(confused_with.items(), key=lambda x: x[1], reverse=True)
            top_confused_ids = [fid for fid, count in top_confused[:5]]
            
            confidences = stats['confidences']
            errors = stats['error_signals']
            
            fp_rate = stats['fp_count'] / stats['total_uses'] if stats['total_uses'] > 0 else 0.0
            
            fp_by_grain[str(fact_id)] = {
                'fp_count': stats['fp_count'],
                'fp_rate': round(fp_rate, 4),
                'total_uses': stats['total_uses'],
                'most_confused_with': top_confused_ids,
                'query_patterns': sorted(list(stats['query_patterns'])),
                'avg_confidence': round(sum(confidences) / len(confidences), 3) if confidences else 0.0,
                'avg_error_signal': round(sum(errors) / len(errors), 3) if errors else 0.0,
                'confidence_stats': {
                    'mean': round(sum(confidences) / len(confidences), 3) if confidences else 0.0,
                    'min': min(confidences) if confidences else 0.0,
                    'max': max(confidences) if confidences else 1.0,
                },
                'error_stats': {
                    'mean': round(sum(errors) / len(errors), 3) if errors else 0.0,
                    'min': min(errors) if errors else 0.0,
                    'max': max(errors) if errors else 1.0,
                }
            }
        
        return fp_by_grain
    
    def analyze_violations(self, violations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze MTR consistency violations per fact.
        
        Tracks when MTR detects dissonance: high search confidence but low coherence.
        
        Args:
            violations: List of violation events from JSONL
        
        Returns:
            Dict mapping fact_id -> violation statistics
        """
        facts: Dict[int, Dict[str, Any]] = defaultdict(
            lambda: {
                'violation_count': 0,
                'dissonance_types': defaultdict(int),
                'error_signals': [],
                'confidences': [],
                'timestamps': [],
                'contexts': [],
            }
        )
        
        for violation in violations:
            fact_id = violation.get('returned_fact_id')
            if fact_id is None:
                continue
            
            fact = facts[fact_id]
            fact['violation_count'] += 1
            
            # Track dissonance type
            dissonance_type = violation.get('dissonance_type', 'unknown')
            fact['dissonance_types'][dissonance_type] += 1
            
            # Track signals
            error = violation.get('mtr_error_signal', 0.0)
            fact['error_signals'].append(error)
            
            confidence = violation.get('returned_confidence', 0.0)
            fact['confidences'].append(confidence)
            
            # Track timestamp
            timestamp = violation.get('timestamp')
            if timestamp:
                fact['timestamps'].append(timestamp)
            
            # Track context if available
            context = violation.get('context', {})
            if context:
                fact['contexts'].append(context)
        
        # Convert to serializable format
        violation_timeline = {}
        for fact_id, stats in facts.items():
            total_violations = stats['violation_count']
            errors = stats['error_signals']
            
            violation_timeline[str(fact_id)] = {
                'total_violations': total_violations,
                'dissonance_types': dict(stats['dissonance_types']),
                'avg_error_signal': round(sum(errors) / len(errors), 3) if errors else 0.0,
                'error_stats': {
                    'mean': round(sum(errors) / len(errors), 3) if errors else 0.0,
                    'min': min(errors) if errors else 0.0,
                    'max': max(errors) if errors else 1.0,
                },
                'first_violation': min(stats['timestamps']) if stats['timestamps'] else None,
                'last_violation': max(stats['timestamps']) if stats['timestamps'] else None,
            }
        
        return violation_timeline


def main():
    """Run consolidation standalone."""
    import sys
    
    dream_bucket_dir = 'data/subconscious/dream_bucket'
    
    if len(sys.argv) > 1:
        dream_bucket_dir = sys.argv[1]
    
    consolidator = DreamBucketConsolidator(dream_bucket_dir)
    report = consolidator.consolidate_all()
    
    print("\n" + "="*70)
    print("CONSOLIDATION REPORT")
    print("="*70)
    for key, value in report.items():
        print(f"{key}: {value}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
