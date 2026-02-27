#!/usr/bin/env python3
"""
Stage 3: Hypothesis Generation

Converts patterns detected in Stage 2 into testable hypotheses.

Input:
  dream_bucket/indices/collision_clusters.json
  dream_bucket/indices/anomaly_timeline.json
  dream_bucket/indices/problematic_facts.json

Output:
  dream_bucket/live/hypotheses.jsonl (one hypothesis per line)
  dream_bucket/indices/hypothesis_graph.json (relationships between hypotheses)

Hypotheses are testable claims about knowledge representation issues.

Examples:
  "Facts 42, 137, 203 are semantically overlapping and need finer discrimination"
  "Fact 89 has a fundamental representation issue - high error across all contexts"
  "Query patterns like 'photosynthesis' trigger systematic confusion"
"""

from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple
from collections import defaultdict
from datetime import datetime
import json
import hashlib

from dream_bucket import DreamBucketReader, DreamBucketWriter


class HypothesisGenerator:
    """
    Generate testable hypotheses from collision clusters and anomalies.
    
    A hypothesis is a specific, testable claim about a knowledge issue:
    - Overlapping facts that need discrimination
    - Systematic errors in fact representation
    - Query pattern sensitivity
    - MTR coherence issues
    """
    
    def __init__(self, dream_bucket_dir: str):
        """
        Initialize hypothesis generator.
        
        Args:
            dream_bucket_dir: Path to dream bucket directory
        """
        self.reader = DreamBucketReader(dream_bucket_dir)
        self.writer = DreamBucketWriter(dream_bucket_dir)
        self.hypotheses: List[Dict[str, Any]] = []
        self.hypothesis_id_counter = 0
    
    def generate_hypotheses(self) -> Dict[str, Any]:
        """
        Run full Stage 3: generate hypotheses from patterns.
        
        Returns:
            Report dict with hypothesis statistics
        """
        print("\n[Stage 3] Generating hypotheses from patterns...")
        
        # Load indices from Stage 2
        print("  Loading collision clusters...")
        clusters = self.reader.load_index('collision_clusters')
        if not clusters:
            print("    ✗ No collision clusters found (run Stage 2 first)")
            return {'error': 'Stage 2 must run first'}
        
        print("  Loading anomaly timeline...")
        anomalies = self.reader.load_index('anomaly_timeline')
        
        print("  Loading problematic facts...")
        problematic = self.reader.load_index('problematic_facts')
        
        report = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'hypotheses_generated': 0,
        }
        
        # Generate hypotheses from each source
        print("  Generating cluster hypotheses...")
        cluster_hypos = self.generate_cluster_hypotheses(clusters)
        self.hypotheses.extend(cluster_hypos)
        report['cluster_hypotheses'] = len(cluster_hypos)
        print(f"    → {len(cluster_hypos)} cluster hypotheses")
        
        print("  Generating anomaly hypotheses...")
        anomaly_hypos = self.generate_anomaly_hypotheses(anomalies)
        self.hypotheses.extend(anomaly_hypos)
        report['anomaly_hypotheses'] = len(anomaly_hypos)
        print(f"    → {len(anomaly_hypos)} anomaly hypotheses")
        
        print("  Generating problem hypotheses...")
        problem_hypos = self.generate_problem_hypotheses(problematic)
        self.hypotheses.extend(problem_hypos)
        report['problem_hypotheses'] = len(problem_hypos)
        print(f"    → {len(problem_hypos)} problem hypotheses")
        
        report['hypotheses_generated'] = len(self.hypotheses)
        
        # Write hypotheses to JSONL
        print("  Writing hypotheses.jsonl...")
        for hypo in self.hypotheses:
            self.writer.append('hypotheses', hypo)
        
        # Build hypothesis graph
        print("  Building hypothesis graph...")
        hypothesis_graph = self.build_hypothesis_graph()
        print(f"    → {len(hypothesis_graph['relationships'])} relationships detected")
        print("  Writing hypothesis_graph.json...")
        self.writer.write_index('hypothesis_graph', hypothesis_graph)
        
        print("\n✓ Stage 3 complete\n")
        return report
    
    def _next_hypothesis_id(self) -> str:
        """Generate unique hypothesis ID."""
        self.hypothesis_id_counter += 1
        return f"H{self.hypothesis_id_counter:04d}"
    
    def _compute_confidence(self, evidence_count: int, severity: float) -> float:
        """
        Compute hypothesis confidence from evidence.
        
        Args:
            evidence_count: Number of pieces of evidence
            severity: Severity rating (0.0-1.0)
        
        Returns:
            Confidence score (0.0-1.0)
        """
        # More evidence = higher confidence
        evidence_factor = min(evidence_count / 5.0, 1.0)  # 5+ events = 1.0
        # Higher severity = higher confidence
        severity_factor = severity
        # Combine: ~60% weight on evidence, 40% on severity
        return round(evidence_factor * 0.6 + severity_factor * 0.4, 3)
    
    def generate_cluster_hypotheses(self, clusters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate hypotheses from collision clusters.
        
        Each cluster suggests facts that need better discrimination.
        
        Args:
            clusters: Collision clusters from Stage 2
        
        Returns:
            List of hypotheses
        """
        hypotheses = []
        cluster_list = clusters.get('clusters', [])
        
        for cluster in cluster_list:
            facts = cluster['facts']
            size = cluster['size']
            internal_collisions = cluster['internal_collisions']
            patterns = cluster.get('query_patterns', [])
            
            # Hypothesis: Facts in this cluster need better discrimination
            hypo_id = self._next_hypothesis_id()
            confidence = self._compute_confidence(internal_collisions, 0.7)
            
            hypothesis = {
                'hypothesis_id': hypo_id,
                'type': 'cluster_discrimination',
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'claim': f"Facts {facts} are semantically overlapping and need finer discrimination",
                'evidence': {
                    'collision_count': internal_collisions,
                    'cluster_size': size,
                    'query_patterns': patterns,
                    'evidence_items': internal_collisions,
                },
                'confidence': confidence,
                'severity': 0.7,
                'testable_predictions': [
                    f"Providing more context will reduce confusion between facts {facts}",
                    f"A more specific fact representation will resolve collisions in {facts}",
                    f"Queries containing {patterns[0] if patterns else 'related patterns'} will show reduced collision rate",
                ],
                'investigation_strategy': [
                    "Analyze query contexts when collision occurs",
                    "Compare fact definitions and identify overlaps",
                    "Propose refined fact boundaries",
                    "Test refined facts with historical queries",
                ],
                'priority': 'high' if size >= 3 else 'medium',
                'related_facts': facts,
            }
            
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    def generate_anomaly_hypotheses(self, anomalies: Optional[Dict]) -> List[Dict[str, Any]]:
        """
        Generate hypotheses from anomalies.
        
        High-severity anomalies suggest fundamental representation issues.
        
        Args:
            anomalies: Anomaly timeline from Stage 2
        
        Returns:
            List of hypotheses
        """
        hypotheses = []
        
        if not anomalies:
            return hypotheses
        
        anomaly_list = anomalies.get('high_severity', [])  # Only high-severity
        
        for anomaly in anomaly_list:
            fact_id = anomaly['fact_id']
            severity = anomaly['severity']
            signals = anomaly['anomaly_signals']
            
            # Hypothesis: This fact has systematic representation issues
            hypo_id = self._next_hypothesis_id()
            confidence = self._compute_confidence(len(signals), severity)
            
            hypothesis = {
                'hypothesis_id': hypo_id,
                'type': 'systematic_error',
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'claim': f"Fact {fact_id} has a fundamental representation issue causing systematic errors",
                'evidence': {
                    'severity': severity,
                    'anomaly_signals': signals,
                    'fp_rate': anomaly.get('fp_rate', 0.0),
                    'error_signal': anomaly.get('avg_error_signal', 0.0),
                    'confidence': anomaly.get('avg_confidence', 0.0),
                },
                'confidence': confidence,
                'severity': severity,
                'testable_predictions': [
                    f"Fact {fact_id} will show high error across different query contexts",
                    f"Fact {fact_id} will be frequently confused with multiple other facts",
                    f"Redefining fact {fact_id} will reduce overall error rate",
                ],
                'investigation_strategy': [
                    f"Examine definition of fact {fact_id}",
                    "Identify which contexts cause systematic errors",
                    "Compare with similar facts",
                    "Propose refined definition",
                    "Test refined definition",
                ],
                'priority': 'critical',
                'related_facts': [int(fact_id)],
            }
            
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    def generate_problem_hypotheses(self, problematic: Optional[Dict]) -> List[Dict[str, Any]]:
        """
        Generate hypotheses from problematic facts.
        
        Critical-level problems suggest specific fixes.
        
        Args:
            problematic: Problematic facts from Stage 2
        
        Returns:
            List of hypotheses
        """
        hypotheses = []
        
        if not problematic:
            return hypotheses
        
        problem_list = problematic.get('critical', [])  # Only critical
        
        for problem in problem_list:
            fact_id = problem['fact_id']
            score = problem['problem_score']
            problems = problem['problems']
            num_confusions = problem.get('num_confusions', 0)
            
            # Generate hypothesis based on problem type
            if num_confusions >= 3:
                # Multiple confusions → discrimination issue
                hypo_type = 'discrimination_needed'
                claim = f"Fact {fact_id} needs better discrimination from {num_confusions} conflicting facts"
                strategy = "Find distinguishing features, add contextual markers, refine definition"
                predictions = [
                    f"Adding context will reduce confusion with competing facts",
                    f"Fact {fact_id} and its competitors share key attributes",
                ]
            elif 'high_fp_rate' in str(problems):
                # High FP rate → representation issue
                hypo_type = 'representation_issue'
                claim = f"Fact {fact_id} has unclear or ambiguous definition"
                strategy = "Clarify definition, add examples, distinguish from similar concepts"
                predictions = [
                    f"Clarified definition will reduce false positive rate",
                    f"The source of confusion is semantic ambiguity",
                ]
            else:
                # Multiple problems → complex issue
                hypo_type = 'complex_issue'
                claim = f"Fact {fact_id} has multiple interrelated problems"
                strategy = "Comprehensive review and potential restructuring needed"
                predictions = [
                    f"Multiple interventions will be needed",
                    f"Issues are interconnected",
                ]
            
            hypo_id = self._next_hypothesis_id()
            confidence = self._compute_confidence(len(problems), score)
            
            hypothesis = {
                'hypothesis_id': hypo_id,
                'type': hypo_type,
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'claim': claim,
                'evidence': {
                    'problem_score': score,
                    'problem_count': len(problems),
                    'problems': problems,
                    'num_confusions': num_confusions,
                },
                'confidence': confidence,
                'severity': score,
                'testable_predictions': predictions,
                'investigation_strategy': [
                    f"Root cause analysis for fact {fact_id}",
                    strategy,
                    "Implement fix",
                    "Test with historical queries",
                    "Validate improvement",
                ],
                'priority': 'critical',
                'related_facts': [int(fact_id)],
            }
            
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    def build_hypothesis_graph(self) -> Dict[str, Any]:
        """
        Build graph of hypothesis relationships.
        
        Finds which hypotheses share facts, compete, or support each other.
        
        Returns:
            Hypothesis graph dict
        """
        graph = {
            'total_hypotheses': len(self.hypotheses),
            'relationships': [],
            'clusters': defaultdict(list),
        }
        
        # Find hypotheses that share facts
        fact_to_hypotheses: Dict[int, List[str]] = defaultdict(list)
        
        for hypo in self.hypotheses:
            hypo_id = hypo['hypothesis_id']
            for fact_id in hypo.get('related_facts', []):
                fact_to_hypotheses[fact_id].append(hypo_id)
        
        # Find relationships
        processed_pairs = set()
        
        for fact_id, hypo_ids in fact_to_hypotheses.items():
            if len(hypo_ids) > 1:
                # Multiple hypotheses about same fact = related
                for i, h1 in enumerate(hypo_ids):
                    for h2 in hypo_ids[i+1:]:
                        pair = tuple(sorted([h1, h2]))
                        if pair not in processed_pairs:
                            graph['relationships'].append({
                                'hypothesis_1': h1,
                                'hypothesis_2': h2,
                                'relationship': 'shares_fact',
                                'shared_fact': fact_id,
                            })
                            processed_pairs.add(pair)
        
        return graph


def main():
    """Run hypothesis generation standalone."""
    import sys
    
    dream_bucket_dir = 'data/subconscious/dream_bucket'
    
    if len(sys.argv) > 1:
        dream_bucket_dir = sys.argv[1]
    
    generator = HypothesisGenerator(dream_bucket_dir)
    report = generator.generate_hypotheses()
    
    print("\n" + "="*70)
    print("HYPOTHESIS GENERATION REPORT")
    print("="*70)
    for key, value in report.items():
        print(f"{key}: {value}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
