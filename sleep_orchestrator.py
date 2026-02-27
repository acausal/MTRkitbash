#!/usr/bin/env python3
"""
Sleep Orchestrator - Coordinates dream bucket processing stages

Stage 1: Log Consolidation (reads JSONL, writes indices)
Stage 2: Pattern Recognition (detects clusters)
Stage 3: Hypothesis Generation (creates testable questions)
Stage 4: Question Generation (prioritizes investigations)
Stage 5: Observation Promotion (promotes high-confidence findings)
Stage 6: Cleanup & Archive (maintenance)

Each stage runs 1-4x per night, produces outputs for next stage.
"""

from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime, UTC
import json
import time

from sleep_consolidator import DreamBucketConsolidator
from dream_bucket import DreamBucketWriter, DreamBucketReader
from sleep_cleanup_archive import SleepCleanupArchiver


class SleepOrchestrator:
    """
    Coordinate all sleep processing stages.
    
    Runs after queries complete or during designated sleep window.
    Manages 6-stage pipeline that turns raw signals into knowledge.
    """
    
    def __init__(self, dream_bucket_dir: str = 'data/subconscious/dream_bucket'):
        """
        Initialize sleep orchestrator.
        
        Args:
            dream_bucket_dir: Path to dream bucket root directory
        """
        self.dream_bucket_dir = Path(dream_bucket_dir)
        self.consolidator = DreamBucketConsolidator(str(self.dream_bucket_dir))
        self.writer = DreamBucketWriter(str(self.dream_bucket_dir))
        self.reader = DreamBucketReader(str(self.dream_bucket_dir))
    
    def run_full_sleep_cycle(self) -> Dict[str, Any]:
        """
        Run full 6-stage sleep cycle.
        
        Returns:
            Report dict with results from all stages
        """
        print("\n" + "="*70)
        print("FULL SLEEP CYCLE".center(70))
        print("="*70)
        
        start_time = time.perf_counter()
        report = {
            'started_at': datetime.now(UTC).isoformat(),
            'stages': {},
            'total_time_seconds': 0,
        }
        
        # Stage 1: Consolidation (REQUIRED)
        print("\n[STAGE 1/6] Log Consolidation")
        stage1_start = time.perf_counter()
        try:
            report['stages']['stage_1'] = self.run_stage_1()
            report['stages']['stage_1']['time_seconds'] = time.perf_counter() - stage1_start
            print(f"  ✓ Complete ({report['stages']['stage_1']['time_seconds']:.1f}s)")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            report['stages']['stage_1'] = {'error': str(e)}
            return report  # Can't proceed without Stage 1
        
        # Stage 2: Pattern Recognition
        print("\n[STAGE 2/6] Pattern Recognition")
        stage2_start = time.perf_counter()
        try:
            report['stages']['stage_2'] = self.run_stage_2()
            report['stages']['stage_2']['time_seconds'] = time.perf_counter() - stage2_start
            print(f"  ✓ Complete ({report['stages']['stage_2']['time_seconds']:.1f}s)")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            report['stages']['stage_2'] = {'error': str(e)}
        
        # Stage 3: Hypothesis Generation
        print("\n[STAGE 3/6] Hypothesis Generation")
        stage3_start = time.perf_counter()
        try:
            report['stages']['stage_3'] = self.run_stage_3()
            report['stages']['stage_3']['time_seconds'] = time.perf_counter() - stage3_start
            print(f"  ✓ Complete ({report['stages']['stage_3']['time_seconds']:.1f}s)")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            report['stages']['stage_3'] = {'error': str(e)}
        
        print("\n[STAGE 4/6] Question Generation")
        stage4_start = time.perf_counter()
        try:
            report['stages']['stage_4'] = self.run_stage_4()
            report['stages']['stage_4']['time_seconds'] = time.perf_counter() - stage4_start
            print(f"  ✓ Complete ({report['stages']['stage_4']['time_seconds']:.1f}s)")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            report['stages']['stage_4'] = {'error': str(e)}
        
        print("\n[STAGE 5/6] Observation Promotion")
        stage5_start = time.perf_counter()
        try:
            report['stages']['stage_5'] = self.run_stage_5()
            report['stages']['stage_5']['time_seconds'] = time.perf_counter() - stage5_start
            print(f"  ✓ Complete ({report['stages']['stage_5']['time_seconds']:.1f}s)")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            report['stages']['stage_5'] = {'error': str(e)}
        
        print("\n[STAGE 6/6] Cleanup & Archive")
        stage6_start = time.perf_counter()
        try:
            report['stages']['stage_6'] = self.run_stage_6()
            report['stages']['stage_6']['time_seconds'] = time.perf_counter() - stage6_start
            print(f"  ✓ Complete ({report['stages']['stage_6']['time_seconds']:.1f}s)")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            report['stages']['stage_6'] = {'error': str(e)}
        
        # Finalize report
        report['completed_at'] = datetime.now(UTC).isoformat()
        report['total_time_seconds'] = time.perf_counter() - start_time
        
        print("\n" + "="*70)
        print("SLEEP CYCLE COMPLETE".center(70))
        print("="*70)
        print(f"Total time: {report['total_time_seconds']:.1f} seconds")
        print("="*70 + "\n")
        
        return report
    
    def run_stage_1(self) -> Dict[str, Any]:
        """
        Stage 1: Log Consolidation
        
        Reads raw JSONL logs, aggregates into JSON indices.
        Creates foundation for all downstream stages.
        
        Returns:
            Report dict with consolidation statistics
        """
        report = self.consolidator.consolidate_all()
        return report
    
    def run_stage_2(self) -> Dict[str, Any]:
        """
        Stage 2: Pattern Recognition
        
        Reads indices from Stage 1, detects patterns:
        - Collision clusters
        - Anomalies
        - Problematic facts
        
        Returns:
            Report dict with pattern statistics
        """
        from sleep_pattern_recognition import PatternRecognizer
        recognizer = PatternRecognizer(str(self.dream_bucket_dir))
        return recognizer.recognize_patterns()
    
    def run_stage_3(self) -> Dict[str, Any]:
        """
        Stage 3: Hypothesis Generation
        
        Reads patterns from Stage 2, generates testable hypotheses:
        - Cluster discrimination hypotheses
        - Systematic error hypotheses
        - Problem-specific hypotheses
        
        Returns:
            Report dict with hypothesis statistics
        """
        from sleep_hypothesis_generation import HypothesisGenerator
        generator = HypothesisGenerator(str(self.dream_bucket_dir))
        return generator.generate_hypotheses()
    
    def run_stage_4(self) -> Dict[str, Any]:
        """
        Stage 4: Question Generation
        
        Reads hypotheses from Stage 3, generates research questions:
        - Convert hypotheses to actionable questions
        - Set investigation deadlines
        - Prioritize by impact
        - Build investigation plan
        
        Returns:
            Report dict with question statistics
        """
        from sleep_question_generation import QuestionGenerator
        generator = QuestionGenerator(str(self.dream_bucket_dir))
        return generator.generate_questions()
    
    def run_stage_5(self) -> Dict[str, Any]:
        """
        Stage 5: Observation Promotion
        
        Executes investigations and promotes validated findings:
        - Track question investigation status
        - Collect investigation results
        - Validate against success criteria
        - Promote successful findings to observations
        - Track learning progress
        
        Returns:
            Report dict with promotion statistics
        """
        from sleep_observation_promotion import ObservationPromoter
        promoter = ObservationPromoter(str(self.dream_bucket_dir))
        return promoter.promote_observations()
    
    def run_stage_6(self) -> Dict[str, Any]:
        """
        Stage 6: Cleanup & Archive
        
        Archive live logs, consolidate learning progress, and health check:
        - Compress and archive session logs to cold storage
        - Consolidate learning metrics across sessions
        - Clean up and consolidate indices
        - Generate system health report
        - Prepare for next session
        
        Returns:
            Report dict with archive and health statistics
        """
        archiver = SleepCleanupArchiver(str(self.dream_bucket_dir))
        now = datetime.now()
        session_id = f"sleep_{now.strftime('%Y_%m_%d')}"
        return archiver.cleanup_and_archive(session_id)
    
    def run_stage_1_only(self) -> Dict[str, Any]:
        """
        Run only Stage 1 (useful for quick consolidation).
        
        Returns:
            Report dict
        """
        return self.run_stage_1()
    
    def get_sleep_report(self) -> Dict[str, Any]:
        """
        Get report of last sleep cycle.
        
        Returns:
            Report dict from sleep_reports/ directory
        """
        sleep_reports_dir = self.dream_bucket_dir / "sleep_reports"
        if not sleep_reports_dir.exists():
            return {'error': 'No sleep reports found'}
        
        # Find latest report
        reports = list(sleep_reports_dir.glob("*.json"))
        if not reports:
            return {'error': 'No sleep reports found'}
        
        latest = max(reports, key=lambda p: p.stat().st_mtime)
        with open(latest, 'r') as f:
            return json.load(f)
    
    def print_status(self) -> None:
        """Print status of dream bucket and indices."""
        print("\n" + "="*70)
        print("DREAM BUCKET STATUS".center(70))
        print("="*70)
        
        # Check live logs
        fp_count = self.reader.count_log_records('false_positives')
        vio_count = self.reader.count_log_records('violations')
        hyp_count = self.reader.count_log_records('hypotheses')
        
        print("\nLive Logs:")
        print(f"  False positives: {fp_count}")
        print(f"  Violations:      {vio_count}")
        print(f"  Hypotheses:      {hyp_count}")
        
        # Check indices
        print("\nIndices:")
        try:
            collision_index = self.reader.load_index('collision_index')
            if collision_index:
                print(f"  ✓ collision_index.json ({len(collision_index)} collisions)")
            else:
                print(f"  ✗ collision_index.json (not created)")
        except:
            print(f"  ✗ collision_index.json (error reading)")
        
        try:
            fp_index = self.reader.load_index('false_positive_by_grain')
            if fp_index:
                print(f"  ✓ false_positive_by_grain.json ({len(fp_index)} facts)")
            else:
                print(f"  ✗ false_positive_by_grain.json (not created)")
        except:
            print(f"  ✗ false_positive_by_grain.json (error reading)")
        
        try:
            vio_index = self.reader.load_index('violation_timeline')
            if vio_index:
                print(f"  ✓ violation_timeline.json ({len(vio_index)} facts)")
            else:
                print(f"  ✗ violation_timeline.json (not created)")
        except:
            print(f"  ✗ violation_timeline.json (error reading)")
        
        print("\n" + "="*70 + "\n")


def main():
    """Run sleep orchestrator from command line."""
    import sys
    
    dream_bucket_dir = 'data/subconscious/dream_bucket'
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == 'full':
            orchestrator = SleepOrchestrator(dream_bucket_dir)
            report = orchestrator.run_full_sleep_cycle()
        elif command == 'stage1':
            orchestrator = SleepOrchestrator(dream_bucket_dir)
            report = orchestrator.run_stage_1_only()
            print(json.dumps(report, indent=2))
        elif command == 'status':
            orchestrator = SleepOrchestrator(dream_bucket_dir)
            orchestrator.print_status()
        elif command == 'report':
            orchestrator = SleepOrchestrator(dream_bucket_dir)
            report = orchestrator.get_sleep_report()
            print(json.dumps(report, indent=2))
        else:
            print(f"Unknown command: {command}")
            print("\nUsage:")
            print("  python sleep_orchestrator.py full      # Run full 6-stage cycle")
            print("  python sleep_orchestrator.py stage1    # Run only Stage 1")
            print("  python sleep_orchestrator.py status    # Show dream bucket status")
            print("  python sleep_orchestrator.py report    # Show last sleep report")
    else:
        orchestrator = SleepOrchestrator(dream_bucket_dir)
        orchestrator.print_status()


if __name__ == "__main__":
    main()
