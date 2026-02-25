#!/usr/bin/env python3
"""
MTR Diagnostics Test Harness
Captures and analyzes what the MTR engine learns during inference.
"""
import torch
from pathlib import Path
from phase3e_orchestrator import Phase3EOrchestrator, QueryContext
from mtr_diagnostics import MTRDiagnostics

def main():
    print("Initializing Phase3E with MTR diagnostics...\n")
    
    orch = Phase3EOrchestrator(
        cartridges_dir="./cartridges",
        vocab_size=50257,
        d_model=256,
        d_state=144,
        device="cpu"
    )
    
    # Initialize diagnostics tracker
    diagnostics = MTRDiagnostics()
    
    test_queries = [
        "What is physics?",
        "Tell me about energy",
        "Explain force and motion",
        "What is gravity?",
        "How does light work?",
    ]
    
    print("\n" + "="*70)
    print("Running queries with full diagnostics capture")
    print("="*70)
    
    for i, query_text in enumerate(test_queries, 1):
        print(f"\n[Query {i}] {query_text}")
        
        context = QueryContext(
            query_text=query_text,
            session_id="diagnostics_run",
            project_context="testing"
        )
        
        try:
            # Request diagnostic capture
            result = orch.process_query(context, capture_diagnostics=True)
            
            print(f"  ✓ Query processed successfully")
            print(f"    Cartridge source: {result.cartridge_facts[0]['source'] if result.cartridge_facts else 'None'}")
            print(f"    MTR confidence: {result.mtr_confidence:.4f}")
            print(f"    Latency: {result.total_latency_ms:.1f}ms")
            
            # Log to diagnostics
            if result.error_signal is not None and result.epistemic_snapshot is not None:
                diagnostics.log_inference(
                    result.error_signal,
                    result.epistemic_snapshot,
                    orch.mtr_state
                )
                print(f"    ✓ Diagnostics logged")
        
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate diagnostics report
    print("\n" + diagnostics.report())
    
    # Layer activity summary
    print("\n" + diagnostics.layer_activity_summary())
    
    # Save final state
    print("\n" + "="*70)
    print("Saving final state...")
    orch.save_state(session_id="diagnostics_run", metadata={
        "test_type": "diagnostics",
        "queries": len(test_queries),
        "final_mtr_time": orch.mtr_state['time']
    })
    print("✓ State saved")
    
    print("\n" + "="*70)
    print("Diagnostics test complete")
    print("="*70)

if __name__ == "__main__":
    main()
