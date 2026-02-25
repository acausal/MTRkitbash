#!/usr/bin/env python3
"""
Quick end-to-end test: orchestrator + cartridges (no grains yet)
"""
import torch
from pathlib import Path
from phase3e_orchestrator import Phase3EOrchestrator, QueryContext

def main():
    print("Initializing Phase3E with test cartridges...\n")
    
    orch = Phase3EOrchestrator(
        cartridges_dir="./cartridges",
        vocab_size=50257,
        d_model=256,
        d_state=144,
        device="cpu"
    )
    
    test_queries = [
        "What is physics?",
        "Tell me about energy",
        "Explain force and motion",
    ]
    
    print("\n" + "="*70)
    print("Running test queries")
    print("="*70)
    
    for i, query_text in enumerate(test_queries, 1):
        print(f"\n[Query {i}] {query_text}")
        
        context = QueryContext(
            query_text=query_text,
            session_id="test_run",
            project_context="testing"
        )
        
        try:
            result = orch.process_query(context)
            
            print(f"  Status: SUCCESS")
            print(f"  Cartridge facts: {len(result.cartridge_facts)}")
            if result.cartridge_facts:
                fact = result.cartridge_facts[0]
                print(f"    Source: {fact['source']}")
                print(f"    Confidence: {fact['confidence']:.3f}")
            
            print(f"  MTR latency: {result.mtr_latency_ms:.1f}ms")
            print(f"  Cartridge latency: {result.cartridge_latency_ms:.1f}ms")
            print(f"  Total: {result.total_latency_ms:.1f}ms")
        
        except Exception as e:
            print(f"  Status: FAILED")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("Saving state...")
    orch.save_state(session_id="test_run", metadata={
        "test_type": "end_to_end",
        "queries": len(test_queries)
    })
    
    print("\nTest complete.")

if __name__ == "__main__":
    main()
