"""
Deep dive: Check phantom state at exactly 100 queries
"""

import statistics
from phase3e_orchestrator import Phase3EOrchestrator, QueryContext

print("\n" + "="*70)
print("PHANTOM STATE AT 100 QUERIES")
print("="*70 + "\n")

orch = Phase3EOrchestrator(
    cartridges_dir="./cartridges",
    enable_grain_system=True
)

print("Running 100 queries...")
for i in range(100):
    ctx = QueryContext(query_text=f"Query {i}")
    result = orch.process_query(ctx)
    if (i + 1) % 20 == 0:
        print(f"  {i+1}...")

print("\nAnalyzing phantom state...\n")

if orch.grain_orchestrator:
    phantom_tracker = orch.grain_orchestrator.phantom_tracker
    
    print(f"Phantom Tracker:")
    print(f"  Cycle count: {phantom_tracker.cycle_count}")
    print(f"  Total phantoms: {len(phantom_tracker.phantoms)}")
    
    if phantom_tracker.phantoms:
        for phantom_id, phantom in list(phantom_tracker.phantoms.items())[:3]:
            print(f"\nPhantom: {phantom_id}")
            print(f"  Status: {phantom.status}")
            print(f"  Hit history length: {len(phantom.hit_history)}")
            print(f"  Hit history (last 10): {phantom.hit_history[-10:]}")
            print(f"  Confidence scores length: {len(phantom.confidence_scores)}")
            print(f"  Avg confidence: {phantom.avg_confidence():.4f}")
            print(f"  Cycle consistency: {phantom.cycle_consistency:.4f}")
            
            # Manually calculate what consistency should be
            if len(phantom.hit_history) > 1:
                try:
                    hit_var = statistics.variance(phantom.hit_history[-50:])
                    hit_cons = 1.0 - min(hit_var / 10.0, 1.0)
                    print(f"  Manual hit_consistency: {hit_cons:.4f} (variance: {hit_var:.4f})")
                except Exception as e:
                    print(f"  Manual hit_consistency: ERROR - {e}")
            
            if len(phantom.confidence_scores) > 1:
                try:
                    conf_var = statistics.variance(phantom.confidence_scores)
                    conf_cons = 1.0 - min(conf_var / 0.25, 1.0)
                    print(f"  Manual confidence_consistency: {conf_cons:.4f} (variance: {conf_var:.4f})")
                except Exception as e:
                    print(f"  Manual confidence_consistency: ERROR - {e}")
            
            print(f"  Thresholds needed:")
            print(f"    - hit_consistency > 0.6")
            print(f"    - confidence_consistency > 0.3")

print("\n" + "="*70)
