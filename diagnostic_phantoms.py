"""
Diagnostic: Check phantom tracking state
Shows why crystallization isn't triggering
"""

from phase3e_orchestrator import Phase3EOrchestrator, QueryContext

print("\n" + "="*70)
print("PHANTOM TRACKING DIAGNOSTICS")
print("="*70)

orch = Phase3EOrchestrator(
    cartridges_dir="./cartridges",
    enable_grain_system=True
)

print("\nRunning 20 queries to populate phantoms...")
confidence_values = []
for i in range(20):
    ctx = QueryContext(query_text=f"Query {i}")
    result = orch.process_query(ctx)
    confidence_values.append(result.mtr_confidence)

print("✓ Queries complete\n")

# Show confidence trend
print(f"MTR Confidence Values (first 10): {[round(c, 3) for c in confidence_values[:10]]}")
print(f"MTR Confidence Values (last 10): {[round(c, 3) for c in confidence_values[-10:]]}")
print(f"Avg MTR confidence: {sum(confidence_values)/len(confidence_values):.3f}\n")

# Check phantom state
if orch.grain_orchestrator:
    phantom_tracker = orch.grain_orchestrator.phantom_tracker
    
    print(f"Phantom Tracker State:")
    print(f"  Cycle count: {phantom_tracker.cycle_count}")
    print(f"  Total phantoms tracked: {len(phantom_tracker.phantoms)}")
    print(f"  Total hits: {phantom_tracker.total_hits}")
    
    if phantom_tracker.phantoms:
        print(f"\nPhantom Details (first 5):")
        for i, (phantom_id, phantom) in enumerate(list(phantom_tracker.phantoms.items())[:5]):
            print(f"\n  Phantom {i+1}: {phantom_id}")
            print(f"    Hit count: {phantom.hit_count}")
            print(f"    Confidence scores (all): {[round(c, 3) for c in phantom.confidence_scores]}")
            print(f"    Avg confidence: {phantom.avg_confidence():.3f}")
            print(f"    Cycle consistency: {phantom.cycle_consistency:.3f}")
            print(f"    First cycle: {phantom.first_cycle_seen}")
            print(f"    Last cycle: {phantom.last_cycle_seen}")
            print(f"    Is locked? {phantom.is_locked()}")
            print(f"    Status: {phantom.status}")
    else:
        print("\n  No phantoms tracked yet!")
    
    # Check locking requirements
    print(f"\nLocking Requirements:")
    print(f"  Min cycles needed: 50")
    print(f"  Min consistency needed: 0.80")
    print(f"  Current cycles: {phantom_tracker.cycle_count}")
    print(f"  Progress to locking: {phantom_tracker.cycle_count}/50 cycles ({phantom_tracker.cycle_count/50*100:.1f}%)")
    
    # Check crystallization state
    locked = [p for p in phantom_tracker.phantoms.values() if p.is_locked()]
    persistent = [p for p in phantom_tracker.phantoms.values() if p.status == "persistent"]
    print(f"\nCrystallization Status:")
    print(f"  Persistent phantoms: {len(persistent)}")
    print(f"  Locked phantoms: {len(locked)}")
    print(f"  Ready to crystallize: {len(locked) > 0}")
    
    if orch.mtr_grain_pipeline:
        trigger = orch.mtr_grain_pipeline.crystallization_trigger
        print(f"\nCrystallization Trigger:")
        print(f"  Interval: {trigger.trigger_interval}")
        print(f"  Queries processed: {orch.query_count}")
        queries_until = trigger.trigger_interval - (orch.query_count % trigger.trigger_interval)
        print(f"  Queries until next trigger: {queries_until}")

else:
    print("Grain orchestrator not available!")

print("\n" + "="*70)
