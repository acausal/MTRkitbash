#!/usr/bin/env python3
"""
Dream Bucket I/O Layer

Manages append-only JSONL logging and JSON index files for the research signal archive.
Non-blocking writes, streaming reads, and rotatable monthly archives.

Structure:
  dream_bucket/
  ├── live/                    (current session logs)
  │   ├── false_positives.jsonl
  │   ├── collisions.jsonl
  │   ├── violations.jsonl
  │   ├── hypotheses.jsonl
  │   ├── traces.jsonl
  │   └── pending_questions.jsonl
  ├── indices/                 (aggregated snapshots)
  │   ├── collision_index.json
  │   ├── false_positive_by_grain.json
  │   ├── violation_timeline.json
  │   ├── collision_clusters.json
  │   ├── anomaly_timeline.json
  │   ├── hypothesis_graph.json
  │   └── observations.json
  ├── archive/                 (rotated monthly)
  │   ├── 2026_02/
  │   ├── 2026_01/
  │   └── ...
  └── sleep_reports/           (sleep cycle outputs)
      ├── sleep_2026_02_24.json
      └── ...
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Iterator
import threading
from queue import Queue


class DreamBucketWriter:
    """Non-blocking append-only JSONL writer with in-memory queue."""
    
    def __init__(self, dream_bucket_root: str):
        """
        Initialize writer.
        
        Args:
            dream_bucket_root: Path to dream_bucket directory root
        """
        self.root = Path(dream_bucket_root)
        self.live_dir = self.root / "live"
        self.indices_dir = self.root / "indices"
        self.archive_dir = self.root / "archive"
        self.reports_dir = self.root / "sleep_reports"
        
        # Create directories
        self.live_dir.mkdir(parents=True, exist_ok=True)
        self.indices_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Write queue for non-blocking I/O
        self._write_queue = Queue(maxsize=1000)
        self._writer_thread = threading.Thread(
            target=self._background_writer,
            daemon=True,
            name="DreamBucketWriter"
        )
        self._writer_thread.start()
    
    def _background_writer(self):
        """Background thread that drains write queue to disk."""
        while True:
            try:
                filepath, record = self._write_queue.get()
                if filepath is None:  # Sentinel for shutdown
                    break
                
                # Atomic append: write to file with lock
                with open(filepath, 'a') as f:
                    json.dump(record, f)
                    f.write('\n')
            except Exception as e:
                print(f"[DreamBucket] Write error: {e}")
    
    def append(self, log_type: str, record: Dict[str, Any]) -> bool:
        """
        Append a record to a live JSONL log (non-blocking).
        
        Args:
            log_type: One of "false_positives", "collisions", "violations", 
                     "hypotheses", "traces", "pending_questions", "validated_observations"
            record: Dict to append (will be JSON-serialized)
        
        Returns:
            True if queued, False if queue full (backpressure)
        """
        valid_types = {
            "false_positives", "collisions", "violations", 
            "hypotheses", "traces", "pending_questions", "validated_observations"
        }
        
        if log_type not in valid_types:
            raise ValueError(f"Invalid log_type: {log_type}")
        
        filepath = self.live_dir / f"{log_type}.jsonl"
        
        # Add timestamp if not present
        if "timestamp" not in record:
            record["timestamp"] = datetime.utcnow().isoformat() + "Z"
        
        try:
            self._write_queue.put_nowait((filepath, record))
            return True
        except:
            return False  # Queue full
    
    def write_index(self, index_name: str, data: Dict[str, Any]) -> None:
        """
        Write or overwrite an index JSON file (blocking).
        
        Args:
            index_name: One of "collision_index", "false_positive_by_grain", 
                       "violation_timeline", "collision_clusters", "anomaly_timeline",
                       "hypothesis_graph", "observations"
            data: Dict to write as JSON
        """
        valid_indices = {
            "collision_index", "false_positive_by_grain", "violation_timeline",
            "collision_clusters", "anomaly_timeline", "hypothesis_graph", "observations",
            "learning_summary", "system_health"
        }
        
        if index_name not in valid_indices:
            raise ValueError(f"Invalid index_name: {index_name}")
        
        filepath = self.indices_dir / f"{index_name}.json"
        
        # Add metadata
        data["_generated_at"] = datetime.utcnow().isoformat() + "Z"
        
        # Atomic write: write to temp then rename
        temp_path = filepath.with_suffix(".tmp")
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        temp_path.replace(filepath)
    
    def write_sleep_report(self, session_id: str, report: Dict[str, Any]) -> None:
        """
        Write a sleep cycle report.
        
        Args:
            session_id: e.g., "sleep_2026_02_24"
            report: Dict with sleep cycle metadata
        """
        filepath = self.reports_dir / f"{session_id}.json"
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
    
    def rotate_session_logs(self, archive_date: Optional[str] = None) -> None:
        """
        Archive live logs to monthly directory.
        
        Args:
            archive_date: YYYY_MM format (default: current month)
        """
        if archive_date is None:
            now = datetime.utcnow()
            archive_date = now.strftime("%Y_%m")
        
        archive_subdir = self.archive_dir / archive_date
        archive_subdir.mkdir(parents=True, exist_ok=True)
        
        # Move all live JSONL files to archive
        for jsonl_file in self.live_dir.glob("*.jsonl"):
            dest = archive_subdir / jsonl_file.name
            jsonl_file.rename(dest)
        
        # Also archive current indices as a snapshot
        indices_snapshot = archive_subdir / "indices_snapshot.json"
        indices_data = {}
        for index_file in self.indices_dir.glob("*.json"):
            with open(index_file) as f:
                indices_data[index_file.stem] = json.load(f)
        
        with open(indices_snapshot, 'w') as f:
            json.dump(indices_data, f, indent=2)


class DreamBucketReader:
    """Streaming reader for JSONL logs and index queries."""
    
    def __init__(self, dream_bucket_root: str):
        """
        Initialize reader.
        
        Args:
            dream_bucket_root: Path to dream_bucket directory root
        """
        self.root = Path(dream_bucket_root)
        self.live_dir = self.root / "live"
        self.indices_dir = self.root / "indices"
        self.archive_dir = self.root / "archive"
    
    def read_live_log(self, log_type: str) -> Iterator[Dict[str, Any]]:
        """
        Stream records from a live JSONL log.
        
        Args:
            log_type: One of "false_positives", "collisions", etc.
        
        Yields:
            Dict records from the log
        """
        filepath = self.live_dir / f"{log_type}.jsonl"
        
        if not filepath.exists():
            return
        
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        pass
    
    def read_live_log_since(
        self, 
        log_type: str, 
        since_timestamp: str
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream records from a live log filtered by timestamp.
        
        Args:
            log_type: Log type
            since_timestamp: ISO format timestamp (e.g., "2026-02-24T02:00:00Z")
        
        Yields:
            Records with timestamp >= since_timestamp
        """
        since_dt = datetime.fromisoformat(since_timestamp.replace('Z', '+00:00'))
        
        for record in self.read_live_log(log_type):
            if "timestamp" in record:
                record_dt = datetime.fromisoformat(record["timestamp"].replace('Z', '+00:00'))
                if record_dt >= since_dt:
                    yield record
    
    def read_archived_log(
        self, 
        log_type: str, 
        archive_date: str
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream records from an archived JSONL log.
        
        Args:
            log_type: Log type
            archive_date: YYYY_MM format
        
        Yields:
            Dict records from the archived log
        """
        filepath = self.archive_dir / archive_date / f"{log_type}.jsonl"
        
        if not filepath.exists():
            return
        
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        pass
    
    def load_index(self, index_name: str) -> Optional[Dict[str, Any]]:
        """
        Load an index JSON file into memory.
        
        Args:
            index_name: Index name
        
        Returns:
            Dict from the index, or None if not found
        """
        filepath = self.indices_dir / f"{index_name}.json"
        
        if not filepath.exists():
            return None
        
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def load_sleep_report(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a sleep report by session_id.
        
        Args:
            session_id: e.g., "sleep_2026_02_24"
        
        Returns:
            Report dict, or None if not found
        """
        filepath = self.root / "sleep_reports" / f"{session_id}.json"
        
        if not filepath.exists():
            return None
        
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def list_archive_dates(self) -> List[str]:
        """
        List all archived month directories.
        
        Returns:
            List of YYYY_MM strings, sorted newest first
        """
        if not self.archive_dir.exists():
            return []
        
        dates = sorted([d.name for d in self.archive_dir.iterdir() if d.is_dir()], reverse=True)
        return dates
    
    def count_log_records(self, log_type: str) -> int:
        """
        Count records in a live log.
        
        Args:
            log_type: Log type
        
        Returns:
            Number of records
        """
        return sum(1 for _ in self.read_live_log(log_type))


# Convenience functions for common operations

def create_dream_bucket(root: str) -> tuple[DreamBucketWriter, DreamBucketReader]:
    """Create and return a (writer, reader) pair."""
    writer = DreamBucketWriter(root)
    reader = DreamBucketReader(root)
    return writer, reader


def log_false_positive(
    writer: DreamBucketWriter,
    source_layer: str,
    query_text: str,
    returned_id: int,
    returned_confidence: float,
    correct_id: Optional[int] = None,
    correct_confidence: Optional[float] = None,
    error_signal: float = 0.0,
    session_id: Optional[str] = None
) -> bool:
    """Convenience function to log a false positive event."""
    record = {
        "type": "false_positive",
        "source_layer": source_layer,
        "query_text": query_text,
        "returned_id": returned_id,
        "returned_confidence": returned_confidence,
        "correct_id": correct_id,
        "correct_confidence": correct_confidence,
        "error_signal": error_signal,
        "session_id": session_id,
    }
    return writer.append("false_positives", record)


def log_consistency_violation(
    writer: DreamBucketWriter,
    source_layer: str,
    returned_fact_id: int,
    returned_confidence: float,
    mtr_error_signal: float,
    mtr_state_time: int = 0,
    dissonance_type: str = "high_confidence_low_coherence",
    context: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None
) -> bool:
    """Convenience function to log a consistency violation."""
    record = {
        "type": "consistency_violation",
        "source_layer": source_layer,
        "returned_fact_id": returned_fact_id,
        "returned_confidence": returned_confidence,
        "mtr_error_signal": mtr_error_signal,
        "mtr_state_time": mtr_state_time,
        "dissonance_type": dissonance_type,
        "context": context or {},
        "session_id": session_id,
    }
    return writer.append("violations", record)


def log_hypothesis(
    writer: DreamBucketWriter,
    hypothesis_subtype: str,
    entities: List[int],
    hypothesis_text: str,
    confidence: float,
    evidence: List[str],
    generated_by: str = "real_time_analysis",
    parent_session_id: Optional[str] = None
) -> bool:
    """Convenience function to log a hypothesis."""
    record = {
        "type": "hypothesis",
        "subtype": hypothesis_subtype,
        "entities": entities,
        "hypothesis": hypothesis_text,
        "confidence": confidence,
        "evidence": evidence,
        "generated_by": generated_by,
        "parent_session_id": parent_session_id,
    }
    return writer.append("hypotheses", record)
