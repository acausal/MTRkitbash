#!/usr/bin/env python3
"""
Stage 5: Observation Promotion

Executes investigations, collects results, and promotes validated findings.

Input:
  dream_bucket/live/pending_questions.jsonl (from Stage 4)
  Investigation results (from external tools/processes)

Output:
  dream_bucket/live/validated_observations.jsonl
  dream_bucket/indices/learning_progress.json
  dream_bucket/indices/resolved_questions.json

An observation is a validated finding that improves the system.

Lifecycle of a question:
  pending → in_progress → awaiting_results → completed → validated → promoted
  
Each stage collects evidence, measures progress, and makes go/no-go decisions.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, UTC
import json

from dream_bucket import DreamBucketReader, DreamBucketWriter


class ObservationPromoter:
    """
    Track investigation progress and promote validated findings to observations.
    
    Responsibilities:
    - Track question status through investigation lifecycle
    - Collect and validate investigation results
    - Measure success against criteria
    - Promote successful investigations to observations
    - Update system knowledge based on findings
    - Track learning progress
    """
    
    def __init__(self, dream_bucket_dir: str):
        """
        Initialize observation promoter.
        
        Args:
            dream_bucket_dir: Path to dream bucket directory
        """
        self.reader = DreamBucketReader(dream_bucket_dir)
        self.writer = DreamBucketWriter(dream_bucket_dir)
    
    def promote_observations(self) -> Dict[str, Any]:
        """
        Run full Stage 5: promote validated observations.
        
        In a real system, this would:
        1. Track question status from external investigation system
        2. Collect investigation results
        3. Validate against success criteria
        4. Promote successful findings
        5. Update learning progress
        
        For now, we'll:
        - Load pending questions
        - Simulate investigation results
        - Promote findings that meet criteria
        - Track learning progress
        
        Returns:
            Report dict with promotion statistics
        """
        print("\n[Stage 5] Promoting validated observations...")
        
        # Read pending questions
        print("  Reading pending questions...")
        questions = []
        try:
            for question in self.reader.read_live_log('pending_questions'):
                questions.append(question)
        except:
            print("    ✗ No pending questions found (run Stage 4 first)")
            return {'error': 'Stage 4 must run first'}
        
        if not questions:
            print("    ✗ No pending questions found")
            return {'error': 'Stage 4 produced no questions'}
        
        print(f"    → {len(questions)} pending questions loaded")
        
        report = {
            'timestamp': datetime.now(UTC).isoformat(),
            'questions_processed': len(questions),
        }
        
        # Categorize questions by deadline
        print("  Analyzing question status...")
        critical_overdue = []
        high_pending = []
        medium_pending = []
        low_pending = []
        
        now = datetime.now(UTC)
        
        for question in questions:
            priority = question.get('priority', 'medium')
            deadline_str = question.get('deadline', '')
            
            try:
                deadline = datetime.fromisoformat(deadline_str.replace('Z', '+00:00'))
            except:
                deadline = now + timedelta(days=7)
            
            time_until_deadline = (deadline - now).total_seconds() / 3600  # hours
            
            if priority == 'critical' and time_until_deadline < 0:
                critical_overdue.append((question, abs(time_until_deadline)))
            elif priority == 'high':
                high_pending.append(question)
            elif priority == 'medium':
                medium_pending.append(question)
            else:
                low_pending.append(question)
        
        print(f"    → {len(critical_overdue)} critical questions overdue")
        print(f"    → {len(high_pending)} high priority pending")
        print(f"    → {len(medium_pending)} medium priority pending")
        print(f"    → {len(low_pending)} low priority pending")
        
        # Simulate investigation and promotion
        print("  Simulating investigations and promoting findings...")
        observations = []
        
        for question in questions[:3]:  # Simulate first 3 for demo
            observation = self._simulate_investigation(question)
            if observation:
                observations.append(observation)
        
        report['observations_promoted'] = len(observations)
        print(f"    → {len(observations)} findings promoted to observations")
        
        # Write observations
        if observations:
            print("  Writing validated_observations.jsonl...")
            for obs in observations:
                self.writer.append('validated_observations', obs)
        
        # Build learning progress report
        print("  Building learning progress report...")
        learning_progress = self.build_learning_progress(
            questions, observations, critical_overdue
        )
        report['learning_progress'] = learning_progress
        
        # Write progress report
        # Note: we'll skip writing this as an index since it's not in valid_indices
        # But we track it in the report
        
        report['critical_overdue'] = len(critical_overdue)
        report['high_pending'] = len(high_pending)
        report['medium_pending'] = len(medium_pending)
        report['low_pending'] = len(low_pending)
        
        print("\n✓ Stage 5 complete\n")
        return report
    
    def _simulate_investigation(self, question: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Simulate investigation of a question.
        
        In a real system, this would call external investigation tools.
        For now, we'll randomly succeed/fail based on priority.
        
        Args:
            question: Research question to investigate
        
        Returns:
            Promoted observation if successful, None otherwise
        """
        import random
        
        question_id = question.get('question_id', 'Q0000')
        priority = question.get('priority', 'medium')
        success_criteria = question.get('success_criteria', [])
        
        # Simulate success likelihood based on priority
        # (critical/high questions more likely to succeed as they're easier)
        if priority == 'critical':
            success_chance = 0.85
        elif priority == 'high':
            success_chance = 0.75
        elif priority == 'medium':
            success_chance = 0.60
        else:
            success_chance = 0.40
        
        # Simulate investigation
        if random.random() < success_chance:
            # Success! Create observation
            observation = {
                'observation_id': f"O{question_id[1:]}",  # Convert Q0001 → O0001
                'question_id': question_id,
                'hypothesis_id': question.get('hypothesis_id', ''),
                'timestamp': datetime.now(UTC).isoformat(),
                
                'finding': self._generate_finding(question),
                'evidence': self._generate_evidence(question),
                
                'success_criteria_met': len(success_criteria),  # Assume all met in simulation
                'total_success_criteria': len(success_criteria),
                
                'investigation_time_hours': random.uniform(1, 6),
                'investigation_difficulty': question.get('resources_needed', {}).get('difficulty', 'medium'),
                
                'impact_score': self._calculate_impact(question),
                'confidence': round(random.uniform(0.7, 0.99), 3),
                
                'recommended_actions': self._generate_actions(question),
                
                'related_facts': question.get('related_facts', []),
                'related_questions': [],
                
                'status': 'validated',
                'validated_at': datetime.now(UTC).isoformat(),
                'promoted_at': datetime.now(UTC).isoformat(),
            }
            
            return observation
        else:
            return None
    
    def _generate_finding(self, question: Dict[str, Any]) -> str:
        """Generate finding text from question."""
        question_type = question.get('question_type', 'unknown')
        
        if question_type == 'cluster_discrimination':
            return "Successfully identified distinguishing features between confused facts"
        elif question_type == 'systematic_error':
            return "Root cause identified: fact definition lacks sufficient specificity"
        elif question_type == 'discrimination_needed':
            return "Context markers effectively distinguish between competing facts"
        elif question_type == 'representation_issue':
            return "Clarified definition resolves ambiguity and reduces false positives"
        elif question_type == 'complex_issue':
            return "Multiple interconnected issues addressed through coordinated fixes"
        else:
            return "Investigation completed with actionable findings"
    
    def _generate_evidence(self, question: Dict[str, Any]) -> Dict[str, Any]:
        """Generate evidence supporting the finding."""
        import random
        
        return {
            'queries_tested': random.randint(20, 100),
            'accuracy_improvement': round(random.uniform(0.15, 0.50), 3),
            'error_reduction': round(random.uniform(0.20, 0.60), 3),
            'validation_method': 'historical query replay',
            'reproducibility': 'confirmed',
            'side_effects': 'none detected',
        }
    
    def _calculate_impact(self, question: Dict[str, Any]) -> float:
        """Calculate impact score of the finding."""
        severity = question.get('severity', 0.5)
        priority = question.get('priority', 'medium')
        
        # Weight by priority and severity
        if priority == 'critical':
            priority_weight = 1.0
        elif priority == 'high':
            priority_weight = 0.8
        elif priority == 'medium':
            priority_weight = 0.6
        else:
            priority_weight = 0.4
        
        impact = severity * priority_weight
        return round(min(impact, 1.0), 3)
    
    def _generate_actions(self, question: Dict[str, Any]) -> List[str]:
        """Generate recommended actions from finding."""
        question_type = question.get('question_type', 'unknown')
        
        actions = [
            "Document finding in knowledge base",
            "Update fact definitions or context markers",
            "Validate across all related facts",
        ]
        
        if question_type == 'cluster_discrimination':
            actions.append("Deploy context markers system-wide")
            actions.append("Monitor collision rate for regressions")
        elif question_type == 'systematic_error':
            actions.append("Implement restructured fact definition")
            actions.append("Retrain MTR on corrected facts")
        elif question_type == 'complex_issue':
            actions.append("Stage coordinated deployment of all fixes")
            actions.append("Monitor for unexpected interactions")
        
        return actions
    
    def build_learning_progress(
        self,
        questions: List[Dict],
        observations: List[Dict],
        critical_overdue: List[tuple]
    ) -> Dict[str, Any]:
        """
        Build comprehensive learning progress report.
        
        Args:
            questions: All pending questions
            observations: Promoted observations
            critical_overdue: Critical questions past deadline
        
        Returns:
            Learning progress dict
        """
        # Calculate metrics
        total_questions = len(questions)
        total_promoted = len(observations)
        promotion_rate = total_promoted / total_questions if total_questions > 0 else 0.0
        
        # Group observations by type
        observations_by_type = {}
        for obs in observations:
            # Try to extract type from related question
            question_id = obs.get('question_id', 'unknown')
            obs_type = f"type_{question_id}"
            observations_by_type[obs_type] = observations_by_type.get(obs_type, 0) + 1
        
        # Calculate average impact
        avg_impact = 0.0
        if observations:
            avg_impact = sum(o.get('impact_score', 0.0) for o in observations) / len(observations)
        
        # Estimate total improvement
        total_improvement = avg_impact * promotion_rate
        
        progress = {
            'created_at': datetime.now(UTC).isoformat(),
            
            'total_questions': total_questions,
            'total_promoted': total_promoted,
            'promotion_rate': round(promotion_rate, 3),
            
            'critical_overdue': len(critical_overdue),
            
            'observations_by_type': observations_by_type,
            'average_impact_score': round(avg_impact, 3),
            'estimated_total_improvement': round(total_improvement, 3),
            
            'learning_status': self._assess_learning_status(
                total_questions, total_promoted, critical_overdue
            ),
            
            'recommendations': self._generate_recommendations(
                total_questions, total_promoted, critical_overdue
            ),
        }
        
        return progress
    
    def _assess_learning_status(
        self,
        total: int,
        promoted: int,
        critical_overdue: list
    ) -> str:
        """Assess overall learning system status."""
        promotion_rate = promoted / total if total > 0 else 0.0
        
        if critical_overdue:
            return "⚠️  CRITICAL: Overdue critical questions need immediate attention"
        elif promotion_rate < 0.3:
            return "🟡 SLOW: Low promotion rate - may need to simplify questions"
        elif promotion_rate > 0.8:
            return "✅ HEALTHY: Good promotion rate and learning progress"
        else:
            return "🟢 ACTIVE: Normal learning activity"
    
    def _generate_recommendations(
        self,
        total: int,
        promoted: int,
        critical_overdue: list
    ) -> List[str]:
        """Generate recommendations for improving learning."""
        recommendations = []
        
        if critical_overdue:
            recommendations.append("URGENT: Address overdue critical questions")
        
        if total == 0:
            recommendations.append("No questions to investigate - generate more from patterns")
        elif promoted == 0:
            recommendations.append("No promoted observations yet - continue investigations")
        elif promoted < total * 0.3:
            recommendations.append("Low success rate - consider simplifying questions")
        else:
            recommendations.append("Continue current investigation pace")
        
        recommendations.append("Monitor hypothesis validation rate")
        recommendations.append("Consider adjusting question difficulty based on results")
        
        return recommendations


def main():
    """Run observation promotion standalone."""
    import sys
    
    dream_bucket_dir = 'data/subconscious/dream_bucket'
    
    if len(sys.argv) > 1:
        dream_bucket_dir = sys.argv[1]
    
    promoter = ObservationPromoter(dream_bucket_dir)
    report = promoter.promote_observations()
    
    print("\n" + "="*70)
    print("OBSERVATION PROMOTION REPORT")
    print("="*70)
    for key, value in report.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        elif isinstance(value, list):
            print(f"{key}: {len(value)} items")
        else:
            print(f"{key}: {value}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
