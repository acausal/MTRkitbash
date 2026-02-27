#!/usr/bin/env python3
"""
Dream Bucket I/O Test Suite

Tests:
1. Basic write/read round-trip
2. JSONL append semantics
3. Index JSON reads
4. Archive rotation
5. Timestamp filtering
6. Non-blocking queue behavior
"""

import json
import tempfile
import time
from pathlib import Path
from datetime import datetime, timedelta

from dream_bucket import (
    DreamBucketWriter, 
    DreamBucketReader,
    log_false_positive,
    log_consistency_violation,
    log_hypothesis
)


def test_directory_creation():
    """Test that directory structure is created on init."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = DreamBucketWriter(tmpdir)
        
        assert (Path(tmpdir) / "live").exists()
        assert (Path(tmpdir) / "indices").exists()
        assert (Path(tmpdir) / "archive").exists()
        assert (Path(tmpdir) / "sleep_reports").exists()
        
    print("✓ Directory creation")


def test_append_and_read():
    """Test basic append + read."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = DreamBucketWriter(tmpdir)
        reader = DreamBucketReader(tmpdir)
        
        # Append a few records
        log_false_positive(
            writer,
            source_layer="cartridge",
            query_text="photosynthesis",
            returned_id=42,
            returned_confidence=0.87,
            correct_id=137,
            correct_confidence=0.45,
            error_signal=0.31
        )
        
        log_false_positive(
            writer,
            source_layer="grain",
            query_text="plant energy",
            returned_id=42,
            returned_confidence=0.89,
            error_signal=0.28
        )
        
        # Give background writer time to flush
        time.sleep(0.2)
        
        # Read back
        records = list(reader.read_live_log("false_positives"))
        
        assert len(records) == 2
        assert records[0]["source_layer"] == "cartridge"
        assert records[0]["query_text"] == "photosynthesis"
        assert records[1]["source_layer"] == "grain"
        
    print("✓ Append and read")


def test_index_write_and_load():
    """Test index JSON write/load."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = DreamBucketWriter(tmpdir)
        reader = DreamBucketReader(tmpdir)
        
        # Write index
        collision_index = {
            "collision_index": {
                "42": {
                    "collides_with": [137, 89, 203],
                    "collision_count": 47,
                    "avg_confusion_confidence": 0.81,
                    "query_patterns": ["photosynthesis", "plant energy"]
                }
            }
        }
        
        writer.write_index("collision_index", collision_index)
        
        # Load back
        loaded = reader.load_index("collision_index")
        
        assert loaded is not None
        assert "collision_index" in loaded
        assert loaded["collision_index"]["42"]["collision_count"] == 47
        assert "_generated_at" in loaded
        
    print("✓ Index write and load")


def test_timestamp_filtering():
    """Test reading logs with timestamp filter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = DreamBucketWriter(tmpdir)
        reader = DreamBucketReader(tmpdir)
        
        # Manually write some records with specific timestamps
        now = datetime.utcnow()
        t1 = (now - timedelta(hours=1)).isoformat() + "Z"
        t2 = (now - timedelta(minutes=30)).isoformat() + "Z"
        t3 = now.isoformat() + "Z"
        
        filepath = Path(tmpdir) / "live" / "false_positives.jsonl"
        
        for ts, query in [(t1, "old"), (t2, "medium"), (t3, "recent")]:
            record = {
                "type": "false_positive",
                "query_text": query,
                "timestamp": ts
            }
            with open(filepath, 'a') as f:
                json.dump(record, f)
                f.write('\n')
        
        # Read since t2
        cutoff = (now - timedelta(minutes=35)).isoformat() + "Z"
        records = list(reader.read_live_log_since("false_positives", cutoff))
        
        assert len(records) == 2
        assert records[0]["query_text"] == "medium"
        assert records[1]["query_text"] == "recent"
        
    print("✓ Timestamp filtering")


def test_archive_rotation():
    """Test log rotation to archive."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = DreamBucketWriter(tmpdir)
        reader = DreamBucketReader(tmpdir)
        
        # Write some logs
        log_false_positive(
            writer,
            source_layer="cartridge",
            query_text="test",
            returned_id=1,
            returned_confidence=0.5,
            error_signal=0.3
        )
        
        log_consistency_violation(
            writer,
            source_layer="mtr",
            returned_fact_id=1,
            returned_confidence=0.8,
            mtr_error_signal=0.6
        )
        
        time.sleep(0.2)  # Flush queue
        
        # Rotate
        writer.rotate_session_logs("2026_02")
        
        # Verify files moved
        assert not (Path(tmpdir) / "live" / "false_positives.jsonl").exists()
        assert not (Path(tmpdir) / "live" / "violations.jsonl").exists()
        
        # Verify files in archive
        assert (Path(tmpdir) / "archive" / "2026_02" / "false_positives.jsonl").exists()
        assert (Path(tmpdir) / "archive" / "2026_02" / "violations.jsonl").exists()
        
        # Verify snapshot created
        assert (Path(tmpdir) / "archive" / "2026_02" / "indices_snapshot.json").exists()
        
        # Read from archive
        archived = list(reader.read_archived_log("false_positives", "2026_02"))
        assert len(archived) == 1
        
    print("✓ Archive rotation")


def test_convenience_functions():
    """Test the convenience logging functions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = DreamBucketWriter(tmpdir)
        reader = DreamBucketReader(tmpdir)
        
        # Log false positive
        log_false_positive(
            writer,
            source_layer="cartridge",
            query_text="photosynthesis",
            returned_id=42,
            returned_confidence=0.87,
            correct_id=137,
            error_signal=0.31,
            session_id="test_session"
        )
        
        # Log violation
        log_consistency_violation(
            writer,
            source_layer="mtr",
            returned_fact_id=42,
            returned_confidence=0.89,
            mtr_error_signal=0.71,
            dissonance_type="high_confidence_low_coherence",
            context={"recent_facts": [100, 200]}
        )
        
        # Log hypothesis
        log_hypothesis(
            writer,
            hypothesis_subtype="structural_isomorphism",
            entities=[42, 137],
            hypothesis_text="facts_42_and_137_are_structurally_isomorphic",
            confidence=0.78,
            evidence=["collision_cluster_contains_both", "similar_ternary_deltas"],
            generated_by="sleep_stage_3"
        )
        
        time.sleep(0.2)
        
        # Verify all were written
        assert reader.count_log_records("false_positives") == 1
        assert reader.count_log_records("violations") == 1
        assert reader.count_log_records("hypotheses") == 1
        
    print("✓ Convenience functions")


def test_count_records():
    """Test record counting."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = DreamBucketWriter(tmpdir)
        reader = DreamBucketReader(tmpdir)
        
        for i in range(5):
            log_false_positive(
                writer,
                source_layer="cartridge",
                query_text=f"query_{i}",
                returned_id=i,
                returned_confidence=0.5,
                error_signal=0.3
            )
        
        time.sleep(0.3)
        
        count = reader.count_log_records("false_positives")
        assert count == 5
        
    print("✓ Record counting")


def test_list_archive_dates():
    """Test listing archived months."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = DreamBucketWriter(tmpdir)
        reader = DreamBucketReader(tmpdir)
        
        # Create some archives
        writer.rotate_session_logs("2026_02")
        writer.rotate_session_logs("2026_01")
        writer.rotate_session_logs("2025_12")
        
        dates = reader.list_archive_dates()
        
        assert dates == ["2026_02", "2026_01", "2025_12"]
        
    print("✓ List archive dates")


def test_nonexistent_index():
    """Test reading a non-existent index returns None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        reader = DreamBucketReader(tmpdir)
        
        result = reader.load_index("nonexistent_index")
        assert result is None
        
    print("✓ Nonexistent index handling")


def run_all_tests():
    """Run all tests."""
    print("\n=== Dream Bucket I/O Tests ===\n")
    
    test_directory_creation()
    test_append_and_read()
    test_index_write_and_load()
    test_timestamp_filtering()
    test_archive_rotation()
    test_convenience_functions()
    test_count_records()
    test_list_archive_dates()
    test_nonexistent_index()
    
    print("\n✅ All tests passed!\n")


if __name__ == "__main__":
    run_all_tests()
