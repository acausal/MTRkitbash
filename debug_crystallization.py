"""
Debug crystallization trigger
Run 120 queries and see if crystallization fires at 100
"""

from phase3e_orchestrator import Phase3EOrchestrator, QueryContext

print("\n" + "="*70)
print("CRYSTALLIZATION TRIGGER DEBUG")
print("="*70 + "\n")

orch = Phase3EOrchestrator(
    cartridges_dir="./cartridges",
    enable_grain_system=True
)

print("Running 120 queries...")
crystallizations_found = []

for i in range(120):
    ctx = QueryContext(query_text=f"Query {i}")
    result = orch.process_query(ctx)
    
    if result.crystallization_report:
        crystallizations_found.append({
            'query': i+1,
            'report': result.crystallization_report
        })
        print(f"\n✓ CRYSTALLIZATION at query {i+1}!")
        print(f"  Report: {result.crystallization_report}")
    
    if (i + 1) % 20 == 0:
        print(f"  Query {i+1}...")

print(f"\n" + "="*70)
print(f"RESULTS")
print("="*70)
print(f"Total queries: 120")
print(f"Crystallizations found: {len(crystallizations_found)}")

if crystallizations_found:
    for cryst in crystallizations_found:
        print(f"\nCrystallization at query {cryst['query']}:")
        print(f"  {cryst['report']}")
else:
    print("\nNo crystallizations found!")
    
    # Debug info
    if orch.grain_orchestrator:
        phantom_tracker = orch.grain_orchestrator.phantom_tracker
        print(f"\nPhantom Tracker State:")
        print(f"  Cycles: {phantom_tracker.cycle_count}")
        print(f"  Total phantoms: {len(phantom_tracker.phantoms)}")
        print(f"  Total hits: {phantom_tracker.total_hits}")
        
        persistent = [p for p in phantom_tracker.phantoms.values() if p.status == "persistent"]
        locked = [p for p in phantom_tracker.phantoms.values() if p.status == "locked"]
        print(f"  Persistent: {len(persistent)}")
        print(f"  Locked: {len(locked)}")
        
        if orch.mtr_grain_pipeline:
            trigger = orch.mtr_grain_pipeline.crystallization_trigger
            print(f"\nCrystallization Trigger:")
            print(f"  Query count in trigger: {trigger.query_count}")
            print(f"  Interval: {trigger.trigger_interval}")
            print(f"  Should trigger at: {trigger.trigger_interval} and {trigger.trigger_interval*2}")
            print(f"  Cartridge passed: {trigger.cartridge is not None}")

print("\n" + "="*70)
