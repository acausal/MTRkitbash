#!/usr/bin/env python3
"""
Stage 2: Pattern Recognition

Reads indices from Stage 1 and detects patterns:
- Collision clusters (semantically related confused facts)
- Anomaly timeline (sudden spikes in FP rate)
- Problematic fact groups
- Emerging issues

Input:
  dream_bucket/indices/collision_index.json
  dream_bucket/indices/false_positive_by_grain.json
  dream_bucket/indices/violation_timeline.json

Output:
  dream_bucket/indices/collision_clusters.json
  dream_bucket/indices/anomaly_timeline.json
  dream_bucket/indices/problematic_facts.json
"""

from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
from datetime import datetime
import json

from dream_bucket import DreamBucketReader, DreamBucketWriter


class PatternRecognizer:
    """
    Detect patterns in consolidated dream bucket indices.
    
    Finds collision clusters, anomalies, and problematic facts
    from the aggregated statistics created by Stage 1.
    """
    
    def __init__(self, dream_bucket_dir: str):
        """
        Initialize pattern recognizer.
        
        Args:
            dream_bucket_dir: Path to dream bucket directory
        """
        self.reader = DreamBucketReader(dream_bucket_dir)
        self.writer = DreamBucketWriter(dream_bucket_dir)
    
    def recognize_patterns(self) -> Dict[str, Any]:
        """
        Run full Stage 2 pattern recognition.
        
        Returns:
            Report dict with pattern statistics
        """
        print("\n[Stage 2] Recognizing patterns in consolidated data...")
        
        # Load indices from Stage 1
        print("  Loading collision index...")
        collision_index = self.reader.load_index('collision_index')
        if not collision_index:
            print("    ✗ No collision index found")
            return {'error': 'Stage 1 must run first'}
        
        print("  Loading FP by grain index...")
        fp_by_grain = self.reader.load_index('false_positive_by_grain')
        if not fp_by_grain:
            print("    ✗ No FP index found")
            return {'error': 'Stage 1 must run first'}
        
        print("  Loading violation timeline...")
        violation_timeline = self.reader.load_index('violation_timeline')
        
        report = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'collision_pairs_analyzed': len(collision_index.get('collision_index', {})),
            'facts_analyzed': len(fp_by_grain),
        }
        
        # Detect collision clusters
        print("  Detecting collision clusters...")
        clusters = self.detect_collision_clusters(collision_index, fp_by_grain)
        report['collision_clusters_found'] = len(clusters)
        print(f"    → {len(clusters)} clusters detected")
        print("  Writing collision_clusters.json...")
        self.writer.write_index('collision_clusters', clusters)
        
        # Detect anomalies
        print("  Detecting anomalies...")
        anomalies = self.detect_anomalies(fp_by_grain, violation_timeline)
        report['anomalies_found'] = len(anomalies)
        print(f"    → {len(anomalies)} anomalies detected")
        print("  Writing anomaly_timeline.json...")
        self.writer.write_index('anomaly_timeline', anomalies)
        
        # Identify problematic facts
        print("  Identifying problematic facts...")
        problematic = self.identify_problematic_facts(fp_by_grain, violation_timeline)
        report['problematic_facts_found'] = len(problematic)
        print(f"    → {len(problematic)} problematic facts identified")
        # Note: problematic_facts is computed but not separately indexed
        # (anomaly_timeline already covers the high-severity cases)
        
        print("\n✓ Stage 2 complete\n")
        return report
    
    def detect_collision_clusters(self, collision_index: Dict, fp_by_grain: Dict) -> Dict[str, Any]:
        """
        Detect semantic clusters of collisions.
        
        Groups collision pairs that share query patterns or confusion relationships.
        A cluster is a set of facts that tend to get confused together.
        
        Args:
            collision_index: Collision pairs from Stage 1
            fp_by_grain: FP statistics per fact from Stage 1
        
        Returns:
            Dict with collision clusters
        """
        # Build graph of which facts are confused with each other
        confusion_graph: Dict[int, Set[int]] = defaultdict(set)
        collision_pairs = collision_index.get('collision_index', {})
        
        for pair_str, stats in collision_pairs.items():
            # Parse pair string like "(42, 137)"
            try:
                pair = pair_str.strip('()').split(', ')
                id1, id2 = int(pair[0]), int(pair[1])
                confusion_graph[id1].add(id2)
                confusion_graph[id2].add(id1)
            except:
                continue
        
        # Find clusters using simple connected components
        clusters = []
        visited = set()
        
        def dfs(node: int, cluster: Set[int]) -> None:
            """Depth-first search to find cluster."""
            if node in visited:
                return
            visited.add(node)
            cluster.add(node)
            for neighbor in confusion_graph.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor, cluster)
        
        # Extract all clusters
        for node in confusion_graph.keys():
            if node not in visited:
                cluster: Set[int] = set()
                dfs(node, cluster)
                if len(cluster) > 1:  # Only include clusters of size > 1
                    clusters.append(sorted(list(cluster)))
        
        # Convert to output format
        cluster_data = {
            'total_clusters': len(clusters),
            'clusters': []
        }
        
        for i, cluster in enumerate(clusters):
            # Compute cluster statistics
            collision_count = 0
            query_patterns: Set[str] = set()
            avg_fp_rate = 0.0
            
            # Count collisions within cluster
            for id1 in cluster:
                for id2 in cluster:
                    if id1 < id2:
                        pair_str = f"({id1}, {id2})"
                        if pair_str in collision_pairs:
                            collision_count += collision_pairs[pair_str].get('collision_count', 0)
                
                # Aggregate FP stats
                fact_str = str(id1)
                if fact_str in fp_by_grain:
                    avg_fp_rate += fp_by_grain[fact_str].get('fp_rate', 0.0)
            
            avg_fp_rate = avg_fp_rate / len(cluster) if cluster else 0.0
            
            cluster_data['clusters'].append({
                'cluster_id': i,
                'facts': cluster,
                'size': len(cluster),
                'internal_collisions': collision_count,
                'avg_fp_rate': round(avg_fp_rate, 4),
                'query_patterns': sorted(list(query_patterns)),
            })
        
        return cluster_data
    
    def detect_anomalies(self, fp_by_grain: Dict, violation_timeline: Optional[Dict]) -> Dict[str, Any]:
        """
        Detect anomalous facts with unusual error patterns.
        
        Anomalies include:
        - Sudden high FP rates
        - Rapid violation spikes
        - Inconsistent confidence/error patterns
        
        Args:
            fp_by_grain: FP statistics per fact
            violation_timeline: Violation statistics per fact
        
        Returns:
            Dict with anomaly timeline
        """
        anomalies = []
        
        for fact_str, stats in fp_by_grain.items():
            # Skip if stats is not a dict (malformed data)
            if not isinstance(stats, dict):
                continue
            
            anomaly_signals = []
            severity = 0.0
            
            # Check 1: High FP rate (>15% false positive rate is anomalous)
            fp_rate = stats.get('fp_rate', 0.0)
            if fp_rate > 0.15:
                anomaly_signals.append(f"high_fp_rate_{fp_rate:.2%}")
                severity += 0.3
            
            # Check 2: High number of different confusion targets
            confused_with = stats.get('most_confused_with', [])
            if len(confused_with) >= 3:
                anomaly_signals.append(f"multiple_confusion_targets_{len(confused_with)}")
                severity += 0.2
            
            # Check 3: High error signal with high confidence (dissonance)
            avg_error = stats.get('avg_error_signal', 0.0)
            avg_conf = stats.get('avg_confidence', 0.0)
            if avg_error > 0.4 and avg_conf > 0.7:
                anomaly_signals.append("high_dissonance")
                severity += 0.3
            
            # Check 4: Wide confidence range (inconsistent behavior)
            conf_stats = stats.get('confidence_stats', {})
            conf_range = conf_stats.get('max', 1.0) - conf_stats.get('min', 0.0)
            if conf_range > 0.3:
                anomaly_signals.append(f"inconsistent_confidence_{conf_range:.2f}")
                severity += 0.2
            
            # If anomalies detected, add to list
            if anomaly_signals:
                severity = min(severity, 1.0)  # Cap at 1.0
                anomalies.append({
                    'fact_id': fact_str,
                    'severity': round(severity, 2),
                    'anomaly_signals': anomaly_signals,
                    'fp_rate': round(fp_rate, 4),
                    'avg_error_signal': round(avg_error, 3),
                    'avg_confidence': round(avg_conf, 3),
                })
        
        # Sort by severity
        anomalies.sort(key=lambda x: x['severity'], reverse=True)
        
        return {
            'total_anomalies': len(anomalies),
            'anomalies': anomalies,
            'high_severity': [a for a in anomalies if a['severity'] >= 0.6],
            'medium_severity': [a for a in anomalies if 0.3 <= a['severity'] < 0.6],
            'low_severity': [a for a in anomalies if a['severity'] < 0.3],
        }
    
    def identify_problematic_facts(self, fp_by_grain: Dict, violation_timeline: Optional[Dict]) -> Dict[str, Any]:
        """
        Identify facts with multiple problem indicators.
        
        Combines FP rate, violation count, and confidence issues
        to rank facts by how problematic they are.
        
        Args:
            fp_by_grain: FP statistics
            violation_timeline: Violation statistics
        
        Returns:
            Dict with ranked problematic facts
        """
        problematic = []
        
        for fact_str, stats in fp_by_grain.items():
            # Skip if stats is not a dict (malformed data)
            if not isinstance(stats, dict):
                continue
            
            problem_score = 0.0
            problems = []
            
            # Problem 1: FP rate (weight: 0.3)
            fp_rate = stats.get('fp_rate', 0.0)
            if fp_rate > 0.05:
                problems.append(f"fp_rate_{fp_rate:.1%}")
                problem_score += fp_rate * 0.3
            
            # Problem 2: Error signal too high (weight: 0.25)
            avg_error = stats.get('avg_error_signal', 0.0)
            if avg_error > 0.3:
                problems.append(f"high_error_{avg_error:.2f}")
                problem_score += avg_error * 0.25
            
            # Problem 3: Confidence too low (weight: 0.15)
            avg_conf = stats.get('avg_confidence', 0.0)
            if avg_conf < 0.7:
                problems.append(f"low_confidence_{avg_conf:.2f}")
                problem_score += (1.0 - avg_conf) * 0.15
            
            # Problem 4: Multiple confusions (weight: 0.2)
            num_confusions = len(stats.get('most_confused_with', []))
            if num_confusions > 2:
                problems.append(f"multiple_confusions_{num_confusions}")
                problem_score += (num_confusions / 10.0) * 0.2
            
            # Problem 5: Violations (weight: 0.1)
            if violation_timeline and fact_str in violation_timeline:
                vio = violation_timeline[fact_str]
                vio_count = vio.get('total_violations', 0)
                if vio_count > 5:
                    problems.append(f"violations_{vio_count}")
                    problem_score += (min(vio_count, 20) / 20.0) * 0.1
            
            # If any problems, add to list
            if problems:
                problem_score = min(problem_score, 1.0)
                problematic.append({
                    'fact_id': fact_str,
                    'problem_score': round(problem_score, 3),
                    'problems': problems,
                    'fp_rate': round(fp_rate, 4),
                    'avg_error_signal': round(avg_error, 3),
                    'avg_confidence': round(avg_conf, 3),
                    'num_confusions': num_confusions,
                })
        
        # Sort by problem score
        problematic.sort(key=lambda x: x['problem_score'], reverse=True)
        
        # Categorize
        critical = [p for p in problematic if p['problem_score'] >= 0.7]
        warning = [p for p in problematic if 0.4 <= p['problem_score'] < 0.7]
        notice = [p for p in problematic if p['problem_score'] < 0.4]
        
        return {
            'total_problematic': len(problematic),
            'all_problematic': problematic,
            'critical': critical,
            'warning': warning,
            'notice': notice,
        }


def main():
    """Run pattern recognition standalone."""
    import sys
    
    dream_bucket_dir = 'data/subconscious/dream_bucket'
    
    if len(sys.argv) > 1:
        dream_bucket_dir = sys.argv[1]
    
    recognizer = PatternRecognizer(dream_bucket_dir)
    report = recognizer.recognize_patterns()
    
    print("\n" + "="*70)
    print("PATTERN RECOGNITION REPORT")
    print("="*70)
    for key, value in report.items():
        if isinstance(value, list):
            print(f"{key}: {len(value)} items")
        else:
            print(f"{key}: {value}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
