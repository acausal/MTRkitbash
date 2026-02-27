#!/usr/bin/env python3
"""
Stage 4: Question Generation

Converts hypotheses from Stage 3 into prioritized research questions.

Input:
  dream_bucket/live/hypotheses.jsonl
  dream_bucket/indices/hypothesis_graph.json

Output:
  dream_bucket/live/pending_questions.jsonl (research questions)
  dream_bucket/indices/investigation_plan.json (prioritized roadmap)

A question is an actionable research task derived from a hypothesis.
Each question has:
  - Clear goal
  - Success criteria
  - Investigation deadline
  - Priority level
  - Resource requirements
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json

from dream_bucket import DreamBucketReader, DreamBucketWriter


class QuestionGenerator:
    """
    Convert hypotheses into actionable research questions.
    
    Questions are time-bound, goal-oriented tasks that investigate hypotheses.
    Each question has success criteria and a deadline.
    """
    
    def __init__(self, dream_bucket_dir: str):
        """
        Initialize question generator.
        
        Args:
            dream_bucket_dir: Path to dream bucket directory
        """
        self.reader = DreamBucketReader(dream_bucket_dir)
        self.writer = DreamBucketWriter(dream_bucket_dir)
    
    def generate_questions(self) -> Dict[str, Any]:
        """
        Run full Stage 4: generate questions from hypotheses.
        
        Returns:
            Report dict with question statistics
        """
        print("\n[Stage 4] Generating research questions from hypotheses...")
        
        # Read hypotheses from Stage 3
        print("  Reading hypotheses...")
        hypotheses = []
        try:
            for hypo in self.reader.read_live_log('hypotheses'):
                hypotheses.append(hypo)
        except:
            print("    ✗ No hypotheses found (run Stage 3 first)")
            return {'error': 'Stage 3 must run first'}
        
        if not hypotheses:
            print("    ✗ No hypotheses found")
            return {'error': 'Stage 3 produced no hypotheses'}
        
        print(f"    → {len(hypotheses)} hypotheses loaded")
        
        report = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'hypotheses_processed': len(hypotheses),
        }
        
        # Generate questions
        print("  Converting hypotheses to questions...")
        questions = self.hypotheses_to_questions(hypotheses)
        report['questions_generated'] = len(questions)
        print(f"    → {len(questions)} questions generated")
        
        # Write questions
        print("  Writing pending_questions.jsonl...")
        for question in questions:
            self.writer.append('pending_questions', question)
        
        # Build investigation plan
        print("  Building investigation plan...")
        investigation_plan = self.build_investigation_plan(questions)
        print(f"    → Prioritized {len(investigation_plan['priority_order'])} questions")
        # Note: investigation_plan is computed but not separately indexed
        # (questions themselves contain all necessary information)
        
        # Categorize by priority
        critical = [q for q in questions if q['priority'] == 'critical']
        high = [q for q in questions if q['priority'] == 'high']
        medium = [q for q in questions if q['priority'] == 'medium']
        low = [q for q in questions if q['priority'] == 'low']
        
        report['critical_questions'] = len(critical)
        report['high_priority_questions'] = len(high)
        report['medium_priority_questions'] = len(medium)
        report['low_priority_questions'] = len(low)
        
        print("\n✓ Stage 4 complete\n")
        return report
    
    def hypotheses_to_questions(self, hypotheses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert hypotheses into research questions.
        
        Args:
            hypotheses: List of hypotheses from Stage 3
        
        Returns:
            List of research questions
        """
        questions = []
        
        for i, hypo in enumerate(hypotheses, 1):
            question_id = f"Q{i:04d}"
            hypo_type = hypo.get('type', 'unknown')
            claim = hypo.get('claim', '')
            confidence = hypo.get('confidence', 0.5)
            severity = hypo.get('severity', 0.5)
            priority = hypo.get('priority', 'medium')
            
            # Generate question text from claim
            question_text = self._claim_to_question(claim, hypo_type)
            
            # Generate success criteria
            success_criteria = self._generate_success_criteria(
                hypo_type, hypo, claim
            )
            
            # Set deadline based on priority
            deadline = self._set_deadline(priority)
            
            # Get investigation steps from hypothesis
            investigation_steps = hypo.get('investigation_strategy', [])
            
            question = {
                'question_id': question_id,
                'hypothesis_id': hypo.get('hypothesis_id', f'H{i:04d}'),
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                
                'question': question_text,
                'question_type': hypo_type,
                
                'original_claim': claim,
                'evidence': hypo.get('evidence', {}),
                
                'hypothesis_confidence': round(confidence, 3),
                'severity': round(severity, 3),
                'priority': priority,
                
                'success_criteria': success_criteria,
                'investigation_steps': investigation_steps,
                
                'deadline': deadline,
                'deadline_days': self._priority_to_days(priority),
                
                'expected_outcome': self._expected_outcome(hypo_type, claim),
                'resources_needed': self._resources_needed(hypo_type),
                
                'related_facts': hypo.get('related_facts', []),
                'testable_predictions': hypo.get('testable_predictions', []),
                
                'status': 'pending',
                'created_at': datetime.utcnow().isoformat() + 'Z',
            }
            
            questions.append(question)
        
        return questions
    
    def _claim_to_question(self, claim: str, hypo_type: str) -> str:
        """Convert hypothesis claim to research question."""
        # Convert statement to question
        if hypo_type == 'cluster_discrimination':
            # "Facts X, Y, Z need discrimination" → "Can we improve discrimination..."
            return f"Can we improve discrimination between the facts involved in this collision cluster?"
        elif hypo_type == 'systematic_error':
            # "Fact X has representation issue" → "What is the representation issue with fact X?"
            return f"What is the fundamental representation issue causing systematic errors?"
        elif hypo_type == 'discrimination_needed':
            return f"How can we better distinguish the facts that are being confused?"
        elif hypo_type == 'representation_issue':
            return f"What clarifications to the fact definition would reduce errors?"
        elif hypo_type == 'complex_issue':
            return f"What are the interconnected problems and how should we address them?"
        else:
            return f"Can we validate and act on this hypothesis?"
    
    def _generate_success_criteria(self, hypo_type: str, hypo: Dict, claim: str) -> List[str]:
        """Generate success criteria for question."""
        criteria = []
        
        if hypo_type == 'cluster_discrimination':
            criteria = [
                "Identify distinguishing features between confused facts",
                "Propose refined fact definitions or context markers",
                "Show that refined definitions reduce collision rate by >30%",
                "Validate predictions with historical query analysis",
            ]
        elif hypo_type == 'systematic_error':
            criteria = [
                "Root cause analysis identifies specific representation issue",
                "Proposed fix is documented and justified",
                "Fix reduces error rate by >50%",
                "No regression in other facts",
            ]
        elif hypo_type == 'discrimination_needed':
            criteria = [
                "Find >2 distinguishing features between facts",
                "Create context-aware distinction mechanism",
                "Test on 50+ historical queries showing improvement",
                "Document new fact boundaries",
            ]
        elif hypo_type == 'representation_issue':
            criteria = [
                "Identify sources of ambiguity in definition",
                "Create clarified definition with examples",
                "Show false positive rate reduction of >50%",
                "Validate with manual testing",
            ]
        elif hypo_type == 'complex_issue':
            criteria = [
                "Map all interconnected problems",
                "Propose comprehensive fix strategy",
                "Implement and test all fixes",
                "Overall improvement >30%",
            ]
        else:
            criteria = [
                "Gather evidence for/against hypothesis",
                "Document findings clearly",
                "Make go/no-go decision",
            ]
        
        return criteria
    
    def _set_deadline(self, priority: str) -> str:
        """Set investigation deadline based on priority."""
        now = datetime.utcnow()
        
        if priority == 'critical':
            deadline = now + timedelta(hours=24)
        elif priority == 'high':
            deadline = now + timedelta(days=3)
        elif priority == 'medium':
            deadline = now + timedelta(days=7)
        else:  # low
            deadline = now + timedelta(days=14)
        
        return deadline.isoformat() + 'Z'
    
    def _priority_to_days(self, priority: str) -> int:
        """Convert priority to days until deadline."""
        if priority == 'critical':
            return 1
        elif priority == 'high':
            return 3
        elif priority == 'medium':
            return 7
        else:  # low
            return 14
    
    def _expected_outcome(self, hypo_type: str, claim: str) -> str:
        """Describe expected outcome of investigation."""
        if hypo_type == 'cluster_discrimination':
            return "Refined fact definitions that reduce collision rate"
        elif hypo_type == 'systematic_error':
            return "Root cause identified and fix implemented"
        elif hypo_type == 'discrimination_needed':
            return "Context-aware distinction mechanism"
        elif hypo_type == 'representation_issue':
            return "Clarified fact definition with examples"
        elif hypo_type == 'complex_issue':
            return "Comprehensive restructuring of related facts"
        else:
            return "Validated or refuted hypothesis with actionable outcome"
    
    def _resources_needed(self, hypo_type: str) -> Dict[str, Any]:
        """Estimate resources needed for investigation."""
        base = {
            'time_hours': 2,
            'difficulty': 'medium',
            'tools_needed': ['query analysis', 'fact comparison'],
        }
        
        if hypo_type == 'cluster_discrimination':
            base['time_hours'] = 4
            base['difficulty'] = 'medium'
        elif hypo_type == 'systematic_error':
            base['time_hours'] = 6
            base['difficulty'] = 'hard'
        elif hypo_type == 'discrimination_needed':
            base['time_hours'] = 3
            base['difficulty'] = 'medium'
        elif hypo_type == 'representation_issue':
            base['time_hours'] = 2
            base['difficulty'] = 'easy'
        elif hypo_type == 'complex_issue':
            base['time_hours'] = 8
            base['difficulty'] = 'hard'
        
        return base
    
    def build_investigation_plan(self, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build prioritized investigation plan.
        
        Args:
            questions: List of research questions
        
        Returns:
            Investigation plan dict with priority ordering
        """
        # Group by priority
        by_priority = {
            'critical': [],
            'high': [],
            'medium': [],
            'low': [],
        }
        
        for question in questions:
            priority = question['priority']
            by_priority[priority].append(question['question_id'])
        
        # Build priority order
        priority_order = (
            by_priority['critical'] +
            by_priority['high'] +
            by_priority['medium'] +
            by_priority['low']
        )
        
        # Estimate total investigation time
        total_time = sum(
            q.get('resources_needed', {}).get('time_hours', 2)
            for q in questions
        )
        
        plan = {
            'created_at': datetime.utcnow().isoformat() + 'Z',
            'total_questions': len(questions),
            'priority_counts': {
                'critical': len(by_priority['critical']),
                'high': len(by_priority['high']),
                'medium': len(by_priority['medium']),
                'low': len(by_priority['low']),
            },
            'priority_order': priority_order,
            'estimated_total_time_hours': total_time,
            'recommended_investigation_sequence': [
                {
                    'phase': 1,
                    'priority': 'critical',
                    'questions': by_priority['critical'],
                    'deadline_days': 1,
                    'estimated_hours': sum(
                        q.get('resources_needed', {}).get('time_hours', 2)
                        for q in questions
                        if q['priority'] == 'critical'
                    ),
                },
                {
                    'phase': 2,
                    'priority': 'high',
                    'questions': by_priority['high'],
                    'deadline_days': 3,
                    'estimated_hours': sum(
                        q.get('resources_needed', {}).get('time_hours', 2)
                        for q in questions
                        if q['priority'] == 'high'
                    ),
                },
                {
                    'phase': 3,
                    'priority': 'medium',
                    'questions': by_priority['medium'],
                    'deadline_days': 7,
                    'estimated_hours': sum(
                        q.get('resources_needed', {}).get('time_hours', 2)
                        for q in questions
                        if q['priority'] == 'medium'
                    ),
                },
                {
                    'phase': 4,
                    'priority': 'low',
                    'questions': by_priority['low'],
                    'deadline_days': 14,
                    'estimated_hours': sum(
                        q.get('resources_needed', {}).get('time_hours', 2)
                        for q in questions
                        if q['priority'] == 'low'
                    ),
                },
            ]
        }
        
        return plan


def main():
    """Run question generation standalone."""
    import sys
    
    dream_bucket_dir = 'data/subconscious/dream_bucket'
    
    if len(sys.argv) > 1:
        dream_bucket_dir = sys.argv[1]
    
    generator = QuestionGenerator(dream_bucket_dir)
    report = generator.generate_questions()
    
    print("\n" + "="*70)
    print("QUESTION GENERATION REPORT")
    print("="*70)
    for key, value in report.items():
        print(f"{key}: {value}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
