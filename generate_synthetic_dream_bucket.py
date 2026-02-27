#!/usr/bin/env python3
"""
Generate synthetic dream bucket data for testing sleep stages.

Creates realistic false positive and violation events that Stage 1-2 can process.
Useful for testing before you have real query data.
"""

from dream_bucket import DreamBucketWriter
from datetime import datetime, timedelta
import time
import random


def generate_synthetic_data(dream_bucket_dir: str = 'data/subconscious/dream_bucket'):
    """
    Generate synthetic false positives and violations.
    
    Creates data that mimics what would come from real queries,
    but with controlled patterns for testing.
    """
    writer = DreamBucketWriter(dream_bucket_dir)
    
    print("\n" + "="*70)
    print("GENERATING SYNTHETIC DREAM BUCKET DATA")
    print("="*70)
    
    # Define some fact groups that will collide
    collision_groups = [
        # Group 1: Photosynthesis-related
        {
            'facts': [42, 137, 203],
            'queries': ['photosynthesis', 'plant energy', 'chlorophyll', 'glucose production'],
            'collision_pairs': [(42, 137), (42, 203), (137, 203)],
        },
        # Group 2: Cellular respiration-related
        {
            'facts': [89, 156, 274],
            'queries': ['cellular respiration', 'ATP', 'mitochondria energy', 'glucose oxidation'],
            'collision_pairs': [(89, 156), (89, 274), (156, 274)],
        },
        # Group 3: DNA-related
        {
            'facts': [112, 245, 367],
            'queries': ['DNA replication', 'heredity', 'genetic code', 'chromosome'],
            'collision_pairs': [(112, 245), (112, 367), (245, 367)],
        },
    ]
    
    base_time = datetime.utcnow()
    false_positive_count = 0
    violation_count = 0
    
    print("\nGenerating false positives...")
    
    # Generate false positives
    for group_idx, group in enumerate(collision_groups):
        for pair_idx, (returned_id, correct_id) in enumerate(group['collision_pairs']):
            # Generate 8-12 FPs per pair
            fp_count = random.randint(8, 12)
            
            for i in range(fp_count):
                query = random.choice(group['queries'])
                timestamp = (base_time + timedelta(hours=group_idx*10 + pair_idx*3 + i*0.5)).isoformat() + 'Z'
                
                fp = {
                    'timestamp': timestamp,
                    'query_text': query,
                    'returned_id': returned_id,
                    'correct_id': correct_id,
                    'returned_confidence': random.uniform(0.75, 0.92),
                    'error_signal': random.uniform(0.25, 0.50),
                    'source_layer': random.choice(['cartridge', 'grain']),
                    'session_id': f'synthetic_session_{group_idx:02d}',
                }
                writer.append('false_positives', fp)
                false_positive_count += 1
                
                # Print progress
                if false_positive_count % 10 == 0:
                    print(f"  → {false_positive_count} false positives generated")
    
    # Wait for background writer
    time.sleep(0.5)
    
    print(f"\nGenerating violations...")
    
    # Generate violations for some facts
    violation_facts = [42, 89, 112, 137, 156]
    
    for fact_idx, fact_id in enumerate(violation_facts):
        # Generate 5-8 violations per fact
        vio_count = random.randint(5, 8)
        
        for i in range(vio_count):
            timestamp = (base_time + timedelta(hours=fact_idx*10 + i*1.5)).isoformat() + 'Z'
            
            violation = {
                'timestamp': timestamp,
                'returned_fact_id': fact_id,
                'returned_confidence': random.uniform(0.70, 0.90),
                'mtr_error_signal': random.uniform(0.45, 0.70),
                'dissonance_type': random.choice([
                    'high_confidence_low_coherence',
                    'context_switch_failure',
                    'incoherent_response',
                ]),
                'session_id': f'synthetic_session_{fact_idx:02d}',
            }
            writer.append('violations', violation)
            violation_count += 1
            
            # Print progress
            if violation_count % 5 == 0:
                print(f"  → {violation_count} violations generated")
    
    # Wait for background writer to finish
    time.sleep(1)
    
    print("\n" + "="*70)
    print("SYNTHETIC DATA GENERATION COMPLETE")
    print("="*70)
    print(f"Generated: {false_positive_count} false positives")
    print(f"Generated: {violation_count} violations")
    print(f"Total events: {false_positive_count + violation_count}")
    print("="*70 + "\n")
    print("Now run:")
    print("  python sleep_orchestrator.py full")
    print("\nOr just Stage 1:")
    print("  python sleep_orchestrator.py stage1")
    print("\n")


if __name__ == "__main__":
    import sys
    
    dream_bucket_dir = 'data/subconscious/dream_bucket'
    
    if len(sys.argv) > 1:
        dream_bucket_dir = sys.argv[1]
    
    generate_synthetic_data(dream_bucket_dir)
