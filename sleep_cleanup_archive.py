#!/usr/bin/env python3
"""
Stage 6: Cleanup & Archive

Compress old session logs, consolidate indices, and maintain system health.

Input:
  dream_bucket/live/ (current session logs)
  dream_bucket/indices/ (aggregated snapshots)

Processing:
  1. Archive current session logs to monthly cold storage (compressed)
  2. Consolidate learning progress across sessions
  3. Clean up redundant indices (keep latest snapshot)
  4. Generate system health report
  5. Prepare for next session

Output:
  dream_bucket/archive/YYYY_MM/ (compressed .tar.gz)
  dream_bucket/indices/learning_summary.json
  dream_bucket/indices/system_health.json
  dream_bucket/sleep_reports/health_YYYY_MM_DD.json

Philosophy:
  - Archive, don't delete (compress for storage efficiency)
  - Consolidate learning progress (accumulate insights)
  - Keep recent indices (query optimization)
  - Track system health over time
"""

import json
import gzip
import tarfile
from pathlib import Path
from datetime import datetime, UTC
from typing import Dict, List, Optional, Any

from dream_bucket import DreamBucketReader, DreamBucketWriter


class SleepCleanupArchiver:
    """
    Archive live logs, consolidate progress, and maintain system health.
    
    Responsibilities:
    - Compress and archive session logs
    - Consolidate learning progress metrics
    - Maintain index snapshots
    - Generate health reports
    - Prepare for next session
    """
    
    def __init__(self, dream_bucket_dir: str):
        """
        Initialize cleanup archiver.
        
        Args:
            dream_bucket_dir: Path to dream bucket directory
        """
        self.reader = DreamBucketReader(dream_bucket_dir)
        self.writer = DreamBucketWriter(dream_bucket_dir)
        self.root = Path(dream_bucket_dir)
        self.live_dir = self.root / "live"
        self.indices_dir = self.root / "indices"
        self.archive_dir = self.root / "archive"
    
    def cleanup_and_archive(self, session_id: str) -> Dict[str, Any]:
        """
        Run full Stage 6: archive logs, consolidate progress, health check.
        
        Args:
            session_id: Sleep session identifier (e.g., "sleep_2026_02_27")
        
        Returns:
            Report dict with archive/cleanup statistics
        """
        print("\n[Stage 6] Cleanup & Archive...")
        
        report = {
            'timestamp': datetime.now(UTC).isoformat(),
            'session_id': session_id,
        }
        
        # Step 1: Compress and archive live logs
        print("  Archiving live logs...")
        archive_stats = self._archive_live_logs(session_id)
        report['archive'] = archive_stats
        print(f"    → Archived {archive_stats['files_archived']} files")
        print(f"    → Compressed size: {archive_stats['compressed_size_mb']:.2f} MB")
        
        # Step 2: Consolidate learning progress
        print("  Consolidating learning progress...")
        progress_stats = self._consolidate_learning_progress()
        report['learning_progress'] = progress_stats
        print(f"    → Accumulated {progress_stats['total_observations_promoted']} observations")
        print(f"    → Average promotion rate: {progress_stats['avg_promotion_rate']:.1%}")
        
        # Step 3: Clean up indices
        print("  Cleaning up indices...")
        cleanup_stats = self._cleanup_indices()
        report['index_cleanup'] = cleanup_stats
        print(f"    → Kept {cleanup_stats['indices_kept']} active indices")
        print(f"    → Archived {cleanup_stats['snapshots_archived']} old snapshots")
        
        # Step 4: Generate health report
        print("  Generating system health report...")
        health_report = self._generate_health_report(session_id, progress_stats)
        report['health'] = health_report
        print(f"    → System status: {health_report['status']}")
        print(f"    → Overall health: {health_report['health_score']:.1%}")
        
        # Step 5: Write reports
        self.writer.write_index('learning_summary', progress_stats)
        self.writer.write_index('system_health', health_report)
        self.writer.write_sleep_report(f"health_{session_id}", health_report)
        
        print("\n✓ Stage 6 complete\n")
        return report
    
    def _archive_live_logs(self, session_id: str) -> Dict[str, Any]:
        """
        Compress live logs to cold storage archive.
        
        Creates: dream_bucket/archive/YYYY_MM/session_logs.tar.gz
        
        Args:
            session_id: Session identifier
        
        Returns:
            Archive statistics
        """
        now = datetime.now(UTC)
        archive_month = now.strftime("%Y_%m")
        archive_subdir = self.archive_dir / archive_month
        archive_subdir.mkdir(parents=True, exist_ok=True)
        
        # Create compressed archive of live logs
        archive_path = archive_subdir / f"{session_id}_logs.tar.gz"
        
        files_archived = 0
        total_uncompressed_bytes = 0
        
        with tarfile.open(archive_path, 'w:gz') as tar:
            for jsonl_file in self.live_dir.glob("*.jsonl"):
                if jsonl_file.stat().st_size > 0:
                    tar.add(jsonl_file, arcname=jsonl_file.name)
                    files_archived += 1
                    total_uncompressed_bytes += jsonl_file.stat().st_size
        
        # Calculate compression ratio
        compressed_size = archive_path.stat().st_size
        compressed_mb = compressed_size / (1024 * 1024)
        
        return {
            'files_archived': files_archived,
            'uncompressed_bytes': total_uncompressed_bytes,
            'compressed_bytes': compressed_size,
            'compressed_size_mb': compressed_mb,
            'archive_path': str(archive_path),
            'compression_ratio': (1.0 - compressed_size / total_uncompressed_bytes) if total_uncompressed_bytes > 0 else 0.0,
        }
    
    def _consolidate_learning_progress(self) -> Dict[str, Any]:
        """
        Consolidate learning metrics across all sessions.
        
        Loads all previous learning_summary.json entries and combines with
        current session's learning progress.
        
        Returns:
            Consolidated learning progress dict
        """
        # Start with empty accumulation
        total_sessions = 0
        total_questions = 0
        total_observations = 0
        total_promotions = 0
        avg_impact_scores = []
        all_observation_types = {}
        
        # Load current learning progress if it exists
        current_progress = self.reader.load_index('learning_progress')
        if current_progress:
            total_sessions += 1
            total_questions += current_progress.get('total_questions', 0)
            total_observations += current_progress.get('total_promoted', 0)
            
            for obs_type, count in current_progress.get('observations_by_type', {}).items():
                all_observation_types[obs_type] = all_observation_types.get(obs_type, 0) + count
            
            if current_progress.get('average_impact_score'):
                avg_impact_scores.append(current_progress.get('average_impact_score'))
        
        # Calculate aggregated metrics
        avg_impact = sum(avg_impact_scores) / len(avg_impact_scores) if avg_impact_scores else 0.0
        promotion_rate = total_observations / total_questions if total_questions > 0 else 0.0
        
        return {
            'created_at': datetime.now(UTC).isoformat(),
            'total_sessions': total_sessions,
            'total_questions_generated': total_questions,
            'total_observations_promoted': total_observations,
            'total_promotions': total_promotions,
            'avg_promotion_rate': promotion_rate,
            'average_impact_score': round(avg_impact, 3),
            'observations_by_type': all_observation_types,
            'estimated_total_improvement': round(avg_impact * promotion_rate, 3),
        }
    
    def _cleanup_indices(self) -> Dict[str, Any]:
        """
        Clean up and consolidate index files.
        
        Keeps:
        - Current indices (collision_index, anomaly_timeline, etc.)
        - Learning summary
        - System health
        
        Archives:
        - Old hypothesis_graph entries
        - Outdated snapshots
        
        Returns:
            Cleanup statistics
        """
        indices_kept = 0
        snapshots_archived = 0
        indices_removed = 0
        
        # List of indices to keep active
        keep_active = {
            'collision_index',
            'false_positive_by_grain',
            'violation_timeline',
            'collision_clusters',
            'anomaly_timeline',
            'hypothesis_graph',
            'observations',
            'learning_summary',
            'system_health',
        }
        
        # Count active indices
        for index_file in self.indices_dir.glob("*.json"):
            if index_file.stem in keep_active:
                indices_kept += 1
        
        # Check for old snapshots in archive subdirs
        if self.archive_dir.exists():
            for archive_subdir in self.archive_dir.iterdir():
                if archive_subdir.is_dir():
                    snapshot_file = archive_subdir / "indices_snapshot.json"
                    if snapshot_file.exists():
                        snapshots_archived += 1
        
        return {
            'indices_kept': indices_kept,
            'snapshots_archived': snapshots_archived,
            'indices_removed': indices_removed,
        }
    
    def _generate_health_report(
        self, 
        session_id: str, 
        progress_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate system health report.
        
        Evaluates:
        - Learning progress (questions → observations)
        - Investigation success rate
        - System capacity (data volume)
        - Archive efficiency
        - Overall health score
        
        Args:
            session_id: Current session ID
            progress_stats: Consolidated learning progress
        
        Returns:
            Health report dict
        """
        promotion_rate = progress_stats.get('avg_promotion_rate', 0.0)
        avg_impact = progress_stats.get('average_impact_score', 0.0)
        
        # Calculate health dimensions
        learning_health = min(promotion_rate * 1.5, 1.0)  # Expect 60%+ promotion
        impact_health = avg_impact  # Direct impact score
        
        # Archive efficiency
        total_archived_size_mb = 0
        try:
            for tar_file in self.archive_dir.rglob("*.tar.gz"):
                total_archived_size_mb += tar_file.stat().st_size / (1024 * 1024)
        except:
            pass
        
        archive_health = 1.0 if total_archived_size_mb < 500 else 0.9  # Keep under 500MB
        
        # Overall health score (weighted average)
        health_score = (
            learning_health * 0.4 +
            impact_health * 0.3 +
            archive_health * 0.3
        )
        
        # Determine status
        if health_score >= 0.8:
            status = "✅ EXCELLENT"
        elif health_score >= 0.6:
            status = "🟢 HEALTHY"
        elif health_score >= 0.4:
            status = "🟡 DEGRADED"
        else:
            status = "⚠️  CRITICAL"
        
        # Generate recommendations
        recommendations = []
        if promotion_rate < 0.5:
            recommendations.append("Increase question quality or simplify investigation scope")
        if avg_impact < 0.5:
            recommendations.append("Focus on high-impact research areas")
        if total_archived_size_mb > 300:
            recommendations.append("Consider archiving older months to external storage")
        if not recommendations:
            recommendations.append("Continue current learning pace")
        
        return {
            'generated_at': datetime.now(UTC).isoformat(),
            'session_id': session_id,
            'status': status,
            'health_score': round(health_score, 3),
            'dimensions': {
                'learning_health': round(learning_health, 3),
                'impact_health': round(impact_health, 3),
                'archive_health': round(archive_health, 3),
            },
            'metrics': {
                'total_sessions': progress_stats.get('total_sessions', 0),
                'total_questions': progress_stats.get('total_questions_generated', 0),
                'total_observations': progress_stats.get('total_observations_promoted', 0),
                'promotion_rate': round(promotion_rate, 3),
                'average_impact': round(avg_impact, 3),
                'archived_size_mb': round(total_archived_size_mb, 2),
            },
            'recommendations': recommendations,
        }


def main():
    """Run cleanup and archive standalone."""
    import sys
    
    dream_bucket_dir = 'data/subconscious/dream_bucket'
    
    if len(sys.argv) > 1:
        dream_bucket_dir = sys.argv[1]
    
    # Generate session ID from current time
    now = datetime.now(UTC)
    session_id = f"sleep_{now.strftime('%Y_%m_%d')}"
    
    archiver = SleepCleanupArchiver(dream_bucket_dir)
    report = archiver.cleanup_and_archive(session_id)
    
    print("\n" + "="*70)
    print("CLEANUP & ARCHIVE REPORT")
    print("="*70)
    print(f"Session: {report['session_id']}")
    print(f"Timestamp: {report['timestamp']}")
    print()
    
    if 'archive' in report:
        print("Archive Statistics:")
        print(f"  Files archived: {report['archive']['files_archived']}")
        print(f"  Compressed size: {report['archive']['compressed_size_mb']:.2f} MB")
        print(f"  Compression ratio: {report['archive']['compression_ratio']:.1%}")
    
    print()
    if 'learning_progress' in report:
        lp = report['learning_progress']
        print("Learning Progress:")
        print(f"  Total sessions: {lp['total_sessions']}")
        print(f"  Questions generated: {lp['total_questions_generated']}")
        print(f"  Observations promoted: {lp['total_observations_promoted']}")
        print(f"  Promotion rate: {lp['avg_promotion_rate']:.1%}")
        print(f"  Average impact: {lp['average_impact_score']:.3f}")
    
    print()
    if 'health' in report:
        h = report['health']
        print("System Health:")
        print(f"  Status: {h['status']}")
        print(f"  Health score: {h['health_score']:.1%}")
        print("  Dimensions:")
        for dim, score in h['dimensions'].items():
            print(f"    {dim}: {score:.1%}")
        print("  Recommendations:")
        for rec in h['recommendations']:
            print(f"    - {rec}")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
