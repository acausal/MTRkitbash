"""
Debug script to isolate failing tests and show full error traces
"""

import traceback

print("\n" + "="*70)
print("DEBUG TEST 1: Grain Activation System")
print("="*70)

try:
    from grain_activation import GrainActivation, Hat
    print("✓ Imported GrainActivation")
    
    activation = GrainActivation(max_cache_mb=1.0)
    print("✓ Initialized GrainActivation")
    
    # Create test grain (as dict, matching what GrainRouter uses)
    test_grain = {
        'grain_id': 'test_001',
        'bit_array_plus': b'\xFF' * 32,
        'bit_array_minus': b'\x00' * 32,
        'num_bits': 256,
    }
    print(f"✓ Created test grain: {test_grain.keys()}")
    
    # Test activation
    print("\nAttempting activation.activate_grains([test_grain])...")
    result = activation.activate_grains([test_grain])
    print(f"✓ Result: {result}")
    
    loaded = result['loaded'] > 0
    print(f"✓ Grains loaded: {loaded}")
    
    # Test lookup
    print("\nAttempting activation.lookup('test_001')...")
    lookup = activation.lookup('test_001')
    print(f"✓ Lookup result: {lookup}")
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    traceback.print_exc()

print("\n" + "="*70)
print("DEBUG TEST 2: Cartridge Phase 1-4 Learning")
print("="*70)

try:
    from cartridge_loader import CartridgeInferenceEngine
    print("✓ Imported CartridgeInferenceEngine")
    
    engine = CartridgeInferenceEngine(cartridges_dir="./cartridges")
    print("✓ Initialized CartridgeInferenceEngine")
    print(f"  Cartridges: {engine.registry.cartridge_count}")
    print(f"  Facts: {engine.registry.total_facts}")
    
    # Check learning infrastructure
    print("\nChecking learning infrastructure...")
    attributes_to_check = [
        'fact_graph',
        'query_anchors', 
        'ctr_tracker',
        'seasonality_tracker',
        'learning_state',
        '_learning_state',
        'phase1_data',
        'phase2_data',
    ]
    
    found_attrs = []
    missing_attrs = []
    
    for attr in attributes_to_check:
        if hasattr(engine, attr):
            found_attrs.append(attr)
            print(f"  ✓ {attr}")
        else:
            missing_attrs.append(attr)
    
    print(f"\nFound {len(found_attrs)} learning attributes: {found_attrs}")
    print(f"Missing {len(missing_attrs)} learning attributes: {missing_attrs}")
    
    # List actual attributes
    print("\nActual CartridgeInferenceEngine attributes (first 20):")
    engine_attrs = [a for a in dir(engine) if not a.startswith('_')][:20]
    for attr in engine_attrs:
        print(f"  - {attr}")
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    traceback.print_exc()

print("\n" + "="*70)
