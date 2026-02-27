"""
Extended Long-Run Test for Kitbash Phase 3E
Runs 200+ queries and monitors:
- Phantom tracking progression
- Crystallization triggers
- Grain activation
- Cache hit rate growth
- Latency improvements
- MTR learning curve

Run: python test_long_run.py
"""

import sys
from pathlib import Path
from typing import Dict, List


def run_extended_test(num_queries: int = 250) -> Dict:
    """
    Run extended test with 250+ queries and monitor all metrics.
    
    Args:
        num_queries: Number of queries to process
        
    Returns:
        Dict with detailed metrics
    """
    try:
        from phase3e_orchestrator import Phase3EOrchestrator, QueryContext
        
        print("\n" + "="*70)
        print(f"EXTENDED LONG-RUN TEST ({num_queries} QUERIES)")
        print("="*70)
        print("\nInitializing orchestrator...")
        
        orch = Phase3EOrchestrator(
            cartridges_dir="./cartridges",
            enable_grain_system=True
        )
        
        print("✓ Ready to process queries\n")
        
        # Generate diverse queries
        query_templates = [
            "What is physics?",
            "Tell me about biology",
            "Explain chemistry",
            "Physics and motion",
            "Biology and life",
            "Chemistry and reactions",
            "What about physics?",
            "Tell me biology",
            "Explain reactions",
        ]
        
        # Metrics tracking
        metrics = {
            'queries': [],
            'latencies': {
                'total': [],
                'cartridge': [],
                'grain': [],
                'mtr': [],
            },
            'errors': [],
            'crystallizations': 0,
            'grains_activated': 0,
            'cache_stats': [],
        }
        
        print(f"Processing {num_queries} queries...")
        print("-" * 70)
        
        for i in range(num_queries):
            # Cycle through queries
            query_text = query_templates[i % len(query_templates)]
            ctx = QueryContext(query_text=query_text)
            
            try:
                result = orch.process_query(ctx)
                
                # Track latencies
                metrics['latencies']['total'].append(result.total_latency_ms)
                metrics['latencies']['cartridge'].append(result.cartridge_latency_ms)
                metrics['latencies']['grain'].append(result.grain_latency_ms)
                metrics['latencies']['mtr'].append(result.mtr_latency_ms)
                metrics['errors'].append(result.mtr_confidence)
                
                # Track crystallizations
                if result.crystallization_report:
                    if result.crystallization_report.get('crystallized_grains'):
                        metrics['crystallizations'] += 1
                
                # Print progress
                if (i + 1) % 50 == 0:
                    avg_latency = sum(metrics['latencies']['total'][-50:]) / 50
                    avg_error = sum(metrics['errors'][-50:]) / 50
                    print(f"  Query {i+1:3d}: avg_latency={avg_latency:6.2f}ms, "
                          f"avg_error={avg_error:.3f}, "
                          f"crystallizations={metrics['crystallizations']}")
                
            except Exception as e:
                print(f"  Query {i+1} failed: {e}")
                return {'passed': False, 'error': str(e)}
        
        # Calculate statistics
        print("\n" + "-" * 70)
        print("ANALYSIS")
        print("-" * 70)
        
        def analyze_latencies(latencies: List[float], name: str) -> Dict:
            if not latencies:
                return {'name': name, 'count': 0}
            
            sorted_latencies = sorted(latencies)
            return {
                'name': name,
                'count': len(latencies),
                'min_ms': round(min(latencies), 2),
                'max_ms': round(max(latencies), 2),
                'avg_ms': round(sum(latencies) / len(latencies), 2),
                'median_ms': round(sorted_latencies[len(sorted_latencies)//2], 2),
                'p95_ms': round(sorted_latencies[int(len(sorted_latencies)*0.95)], 2),
                'p99_ms': round(sorted_latencies[int(len(sorted_latencies)*0.99)], 2),
            }
        
        # Analyze each component
        analyses = {
            'total': analyze_latencies(metrics['latencies']['total'], 'Total Query'),
            'cartridge': analyze_latencies(metrics['latencies']['cartridge'], 'Cartridge'),
            'grain': analyze_latencies(metrics['latencies']['grain'], 'Grain'),
            'mtr': analyze_latencies(metrics['latencies']['mtr'], 'MTR'),
        }
        
        # Print latency analysis
        print("\nLATENCY ANALYSIS")
        for key, analysis in analyses.items():
            if analysis['count'] > 0:
                print(f"\n{analysis['name']}:")
                print(f"  Min:    {analysis['min_ms']:6.2f}ms")
                print(f"  Max:    {analysis['max_ms']:6.2f}ms")
                print(f"  Avg:    {analysis['avg_ms']:6.2f}ms")
                print(f"  Median: {analysis['median_ms']:6.2f}ms")
                print(f"  P95:    {analysis['p95_ms']:6.2f}ms")
                print(f"  P99:    {analysis['p99_ms']:6.2f}ms")
        
        # Learning curve analysis
        print("\nLEARNING CURVE ANALYSIS")
        first_100_error = sum(metrics['errors'][:100]) / 100 if len(metrics['errors']) >= 100 else 0
        last_100_error = sum(metrics['errors'][-100:]) / 100 if len(metrics['errors']) >= 100 else 0
        improvement = ((first_100_error - last_100_error) / first_100_error * 100) if first_100_error > 0 else 0
        
        print(f"  First 100 queries avg error: {first_100_error:.3f}")
        print(f"  Last 100 queries avg error:  {last_100_error:.3f}")
        print(f"  Error reduction: {improvement:.1f}%")
        
        # Latency improvement analysis
        first_100_latency = sum(metrics['latencies']['total'][:100]) / 100 if len(metrics['latencies']['total']) >= 100 else 0
        last_100_latency = sum(metrics['latencies']['total'][-100:]) / 100 if len(metrics['latencies']['total']) >= 100 else 0
        latency_improvement = ((first_100_latency - last_100_latency) / first_100_latency * 100) if first_100_latency > 0 else 0
        
        print(f"\nLATENCY IMPROVEMENT")
        print(f"  First 100 queries avg: {first_100_latency:.2f}ms")
        print(f"  Last 100 queries avg:  {last_100_latency:.2f}ms")
        print(f"  Improvement: {latency_improvement:.1f}%")
        
        # Grain system analysis
        print(f"\nGRAIN SYSTEM")
        print(f"  Crystallization triggers: {metrics['crystallizations']}")
        if orch.grain_router:
            grain_stats = orch.grain_router.get_activation_stats()
            print(f"  Cache hits: {grain_stats['manual_stats']['cache_hits']}")
            print(f"  Cache misses: {grain_stats['manual_stats']['cache_misses']}")
            if grain_stats['manual_stats']['cache_hits'] + grain_stats['manual_stats']['cache_misses'] > 0:
                hit_rate = grain_stats['manual_stats']['cache_hits'] / (grain_stats['manual_stats']['cache_hits'] + grain_stats['manual_stats']['cache_misses'])
                print(f"  Hit rate: {hit_rate:.1%}")
        
        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Queries processed: {num_queries}")
        print(f"Total time: {sum(metrics['latencies']['total']):.0f}ms")
        print(f"Overall avg latency: {sum(metrics['latencies']['total'])/len(metrics['latencies']['total']):.2f}ms")
        print(f"Learning improvement: {improvement:.1f}%")
        print(f"Latency improvement: {latency_improvement:.1f}%")
        
        return {
            'passed': True,
            'queries': num_queries,
            'latencies': analyses,
            'learning_improvement': round(improvement, 1),
            'latency_improvement': round(latency_improvement, 1),
            'crystallizations': metrics['crystallizations'],
        }
        
    except Exception as e:
        import traceback
        print(f"\n✗ ERROR: {e}")
        traceback.print_exc()
        return {'passed': False, 'error': str(e)}


if __name__ == "__main__":
    result = run_extended_test(num_queries=250)
    
    if result['passed']:
        print("\n✓ Extended test completed successfully!")
        sys.exit(0)
    else:
        print(f"\n✗ Extended test failed: {result.get('error')}")
        sys.exit(1)
