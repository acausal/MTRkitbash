"""
GrainRouter Phase 1-2 Enhancement
Adds co-occurrence learning, CTR tracking, and false positive detection.

Mirrors CartridgeLoader Phase 1 implementation but tuned for grain specificity.
- Phase 1: Co-occurrence graph (edges higher than cartridges: +0.20 vs +0.15)
- Phase 1.5: CTR tracking (success rate measurement)
- Phase 2: False positive detection (feeds Dream Bucket)
- PATCH_3: Grain usage logging for unified learning with cartridges

Author: Kitbash Team
Date: February 23, 2026
"""

import json
import time
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple
from collections import defaultdict
from datetime import datetime
from grain_activation import GrainActivation


class GrainRouter:
    """
    Routes queries to crystallized grains for Layer 0 reflex responses.
    
    Now with Phase 1-2 learning:
    - Phase 1: Grain co-occurrence graph (edges when grains activate together)
    - Phase 1.5: CTR tracking (success rate of each grain)
    - Phase 2: False positive detection (high conf grain, high MTR error)
    - PATCH_3: Unified learning with cartridge engine (grain usage logs back to facts)
    
    Responsibilities:
    - Load all crystallized grains from disk
    - Index grains by fact_id, cartridge, and concepts
    - Provide O(1) grain lookup by fact_id
    - Return routing decisions with confidence
    - Track grain performance and learn from MTR feedback
    - Log grain usage back to cartridge engine for fact_graph/anchor/CTR/seasonality
    
    Performance:
    - Load time: ~1-2 seconds (all grains at startup)
    - Lookup time: <1ms (hash table)
    - Learning: Non-blocking append to graph/CTR
    """
    
    def __init__(self, cartridges_dir: str = "./cartridges", cartridge_engine=None, dream_bucket_writer=None):
        """
        Initialize GrainRouter with Phase 1-2 learning infrastructure.
        
        Args:
            cartridges_dir: Path to cartridges directory
            cartridge_engine: Optional CartridgeInferenceEngine for unified learning
            dream_bucket_writer: Optional DreamBucketWriter for false positive logging
        """
        self.cartridges_dir = Path(cartridges_dir)
        self.cartridge_engine = cartridge_engine  # PATCH_3: Reference to cartridge engine
        self.dream_bucket_writer = dream_bucket_writer
        
        # Indices (original functionality)
        self.grains: Dict[str, Dict[str, Any]] = {}  # grain_id -> grain data
        self.grain_by_fact: Dict[int, str] = {}  # fact_id -> grain_id
        self.grain_by_cartridge: Dict[str, List[str]] = defaultdict(list)  # cartridge -> [grain_ids]
        self.grain_by_confidence: List[tuple] = []  # [(confidence, grain_id), ...] sorted desc
        
        # Phase 1: Fact co-occurrence graph
        self.grain_graph: Dict[str, set] = defaultdict(set)  # grain_id -> set of related grain_ids
        self.grain_graph_updated = datetime.now()
        
        # Phase 1.5: CTR (Click-Through Rate) tracking
        self.grain_ctr: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"selected": 0, "good": 0}
        )  # grain_id -> {selected: N, good: M}
        self.good_error_threshold = 0.25  # MTR error < this = "good"
        
        # Phase 2: False positive detection
        self.false_positive_candidates: List[Dict] = []  # For Dream Bucket
        self.collision_clusters: Dict[tuple, Dict] = {}  # (grain_ids) -> {count, queries, first_seen}
        
        # Statistics
        self.load_time_ms = 0
        self.total_grains = 0
        self.total_size_bytes = 0
        
        # Phase 3E.3: L3 cache for hot grains
        self.grain_activation = GrainActivation(max_cache_mb=1.0)
        
        # Load all grains
        self._load_grains()
    
    def _load_grains(self) -> None:
        """Load all crystallized grains from disk.
        
        Supports two storage layouts:
        1. Flat: grains/ directory at root level (grain_*.json, sg_*.json, etc.)
        2. By-cartridge: cartridges/*/grains/ directories
        
        Tries flat first, then falls back to by-cartridge.
        """
        start_time = time.perf_counter()
        
        # Find all grain files
        grain_count = 0
        duplicates = []
        
        # Strategy 1: Try flat grain storage (grains/ at root)
        # This is the current structure: B:\...\grains\grain_*.json
        grains_root = Path(self.cartridges_dir).parent / "grains"
        
        if grains_root.exists():
            # Load from flat structure
            for grain_file in grains_root.glob("*.json"):
                # Skip system files like phantom_tracker.json
                if grain_file.stem == "phantom_tracker":
                    continue
                
                try:
                    with open(grain_file, 'r') as f:
                        grain = json.load(f)
                    
                    grain_id = grain.get('grain_id') or grain_file.stem
                    if not grain_id:
                        continue
                    
                    # Check for duplicate grain_id
                    if grain_id in self.grains:
                        duplicates.append((grain_id, "flat", grain_file.name))
                        continue
                    
                    # Store grain
                    self.grains[grain_id] = grain
                    
                    # Index by fact_id
                    fact_id = grain.get('fact_id')
                    if fact_id is not None:
                        self.grain_by_fact[fact_id] = grain_id
                    
                    # Index by cartridge (use source cartridge if available, else 'unknown')
                    cartridge_id = grain.get('cartridge_id', 'unknown')
                    self.grain_by_cartridge[cartridge_id].append(grain_id)
                    
                    # Track confidence
                    confidence = grain.get('confidence', 0.0)
                    self.grain_by_confidence.append((confidence, grain_id))
                    
                    # Statistics
                    grain_count += 1
                    self.total_size_bytes += grain_file.stat().st_size
                
                except Exception as e:
                    print(f"Warning: Could not load grain {grain_file}: {e}")
        
        # Strategy 2: If no flat structure, try by-cartridge structure
        # This is the fallback: cartridges/*/grains/ directories
        if grain_count == 0:
            for cartridge_dir in self.cartridges_dir.glob("*.kbc"):
                grains_dir = cartridge_dir / "grains"
                
                if not grains_dir.exists():
                    continue
                
                cartridge_id = cartridge_dir.name.replace('.kbc', '')
                
                for grain_file in grains_dir.glob("*.json"):
                    try:
                        with open(grain_file, 'r') as f:
                            grain = json.load(f)
                        
                        grain_id = grain.get('grain_id')
                        if not grain_id:
                            continue
                        
                        # Check for duplicate grain_id
                        if grain_id in self.grains:
                            duplicates.append((grain_id, cartridge_id, grain_file.name))
                            continue
                        
                        # Store grain
                        self.grains[grain_id] = grain
                        
                        # Index by fact_id
                        fact_id = grain.get('fact_id')
                        if fact_id is not None:
                            self.grain_by_fact[fact_id] = grain_id
                        
                        # Index by cartridge
                        self.grain_by_cartridge[cartridge_id].append(grain_id)
                        
                        # Track confidence
                        confidence = grain.get('confidence', 0.0)
                        self.grain_by_confidence.append((confidence, grain_id))
                        
                        # Statistics
                        grain_count += 1
                        self.total_size_bytes += grain_file.stat().st_size
                    
                    except Exception as e:
                        print(f"Warning: Could not load grain {grain_file}: {e}")
        
        # Report duplicates
        if duplicates:
            print(f"\nWarning: Found {len(duplicates)} duplicate grain_ids (skipped):")
            for grain_id, location, filename in duplicates[:10]:
                print(f"  - {grain_id} in {location}/{filename}")
            if len(duplicates) > 10:
                print(f"  ... and {len(duplicates) - 10} more")
        
        # Sort by confidence (descending)
        self.grain_by_confidence.sort(reverse=True, key=lambda x: x[0])
        
        # Calculate statistics
        self.load_time_ms = (time.perf_counter() - start_time) * 1000
        self.total_grains = grain_count
    
    # ========================================================================
    # Phase 3E.3: L3 CACHE METHODS
    # ========================================================================
    
    def activate_grains(self, grain_ids: List[str]) -> Dict[str, bool]:
        """
        Activate grains into L3 cache after crystallization.
        
        Args:
            grain_ids: List of grain IDs to activate
        
        Returns:
            Dict mapping grain_id -> activation_success
        """
        results = {}
        for grain_id in grain_ids:
            grain = self.grains.get(grain_id)
            if grain:
                success = self.grain_activation.load_grain(grain)
                results[grain_id] = success
            else:
                results[grain_id] = False
        
        return results
    
    def lookup_cached(self, grain_id: str) -> Optional[Dict[str, Any]]:
        """
        Look up grain in cache with fallback to disk.
        
        Fast path (L3 cache): <0.1ms on hit
        Slow path (disk): ~1-2ms on miss
        
        Args:
            grain_id: ID of grain to find
        
        Returns:
            Grain dictionary or None if not found
        """
        # Try cache first (fast)
        grain = self.grain_activation.lookup(grain_id)
        if grain:
            return grain
        
        # Fall back to disk (slow)
        return self.grains.get(grain_id)
    
    def batch_activate(self, grain_ids: List[str]) -> None:
        """
        Batch activate multiple grains.
        
        Args:
            grain_ids: List of grain IDs to activate
        """
        for grain_id in grain_ids:
            grain = self.grains.get(grain_id)
            if grain:
                self.grain_activation.load_grain(grain)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get L3 cache statistics."""
        return self.grain_activation.get_stats()
    
    def print_cache_stats(self) -> None:
        """Print formatted cache statistics."""
        self.grain_activation.print_stats()
    
    # ========================================================================
    # ORIGINAL METHODS (existing)
    # ========================================================================

        """
        Look up a grain by fact_id.
        
        Args:
            fact_id: Fact identifier
        
        Returns:
            Grain data if found, None otherwise
        """
        grain_id = self.grain_by_fact.get(fact_id)
        if grain_id:
            return self.grains.get(grain_id)
        return None
    
    def lookup_by_grain_id(self, grain_id: str) -> Optional[Dict[str, Any]]:
        """
        Look up a grain by grain_id.
        
        Args:
            grain_id: Grain identifier (sg_XXXXXXXX)
        
        Returns:
            Grain data if found, None otherwise
        """
        return self.grains.get(grain_id)
    
    def search_grains(self, query_concepts: List[str], 
                     recent_grains: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """
        Search for grains matching query concepts WITH Phase 1-2 boosting.
        
        Args:
            query_concepts: Keywords from the query
            recent_grains: Grains from previous queries in this session (Phase 1 boost)
        
        Returns:
            List of (grain_id, boosted_score) sorted by score descending
        """
        if recent_grains is None:
            recent_grains = []
        
        results = []
        
        for grain_id, grain in self.grains.items():
            # Base score: confidence
            score = grain.get('confidence', 0.0)
            
            # Bonus if grain has derivations
            delta = grain.get('delta', {})
            derivation_count = (
                len(delta.get('positive', [])) +
                len(delta.get('negative', [])) +
                len(delta.get('void', []))
            )
            if derivation_count > 0:
                score += 0.05
            
            # Phase 1: Graph boost (grain is adjacent to recent grains)
            for recent_gid in recent_grains:
                if recent_gid in self.grain_graph.get(grain_id, set()):
                    score += 0.20  # Higher boost for grains (+0.20 vs cartridge +0.15)
                    break
            
            # Phase 1.5: CTR boost (grains with high success rate)
            ctr_data = self.grain_ctr.get(grain_id, {})
            if ctr_data['selected'] > 5:  # Threshold for meaningful data
                success_rate = ctr_data['good'] / ctr_data['selected']
                score += success_rate * 0.10  # Up to +0.10 boost
            
            if score > 0:
                results.append((grain_id, score))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    # ========================================================================
    # PHASE 1: GRAIN CO-OCCURRENCE GRAPH
    # ========================================================================
    
    def log_grain_co_occurrence(self, activated_grains: List[str]) -> None:
        """
        Log co-occurrence: which grains activated together in this query.
        Builds a graph of grain relationships for future boosting.
        
        Args:
            activated_grains: List of grain_ids that activated together
        """
        if len(activated_grains) < 2:
            return
        
        # Add edges between all pairs
        for i, gid1 in enumerate(activated_grains):
            for gid2 in activated_grains[i+1:]:
                self.grain_graph[gid1].add(gid2)
                self.grain_graph[gid2].add(gid1)
        
        self.grain_graph_updated = datetime.now()
    
    def get_graph_density(self) -> Dict[str, Any]:
        """Get statistics about grain co-occurrence graph."""
        nodes = len(self.grain_graph)
        edges = sum(len(neighbors) for neighbors in self.grain_graph.values()) // 2
        avg_neighbors = sum(len(neighbors) for neighbors in self.grain_graph.values()) / max(nodes, 1)
        density = edges / (nodes * (nodes - 1) / 2) if nodes > 1 else 0
        
        return {
            'nodes': nodes,
            'edges': edges,
            'avg_neighbors': avg_neighbors,
            'density': density,
        }
    
    # ========================================================================
    # PHASE 1.5: CTR TRACKING
    # ========================================================================
    
    def log_grain_outcome(self, grain_id: str, mtr_error: float) -> None:
        """
        Log the outcome of using a grain (was it successful?).
        
        Args:
            grain_id: Grain that was used
            mtr_error: MTR error signal
        """
        if grain_id not in self.grains:
            return
        
        if grain_id not in self.grain_ctr:
            self.grain_ctr[grain_id] = {"selected": 0, "good": 0}
        
        self.grain_ctr[grain_id]["selected"] += 1
        
        if mtr_error < self.good_error_threshold:
            self.grain_ctr[grain_id]["good"] += 1
    
    def get_ctr_stats(self) -> Dict[str, Any]:
        """Get CTR statistics across all grains."""
        total_selections = sum(ctr['selected'] for ctr in self.grain_ctr.values())
        total_good = sum(ctr['good'] for ctr in self.grain_ctr.values())
        avg_ctr = total_good / total_selections if total_selections > 0 else 0
        
        # High performers
        high_performers = []
        for gid, ctr_data in self.grain_ctr.items():
            if ctr_data['selected'] >= 3:  # Threshold
                ctr_val = ctr_data['good'] / ctr_data['selected']
                if ctr_val >= 0.8:
                    high_performers.append((gid, ctr_val, ctr_data['selected']))
        
        high_performers.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'total_grains_tracked': len(self.grain_ctr),
            'total_selections': total_selections,
            'avg_ctr': avg_ctr,
            'high_performers': high_performers,
        }
    
    # ========================================================================
    # PHASE 2: FALSE POSITIVE DETECTION
    # ========================================================================
    
    def detect_false_positive(self, grain_id: str, query: str, 
                            grain_confidence: float, mtr_error: float) -> bool:
        """
        Detect false positives: high-confidence grain, but MTR struggled.
        
        Args:
            grain_id: Grain that was returned
            query: Original query
            grain_confidence: Grain's confidence score
            mtr_error: MTR error signal
        
        Returns:
            True if false positive detected, False otherwise
        """
        # Threshold: confident grain + high error = false positive
        if grain_confidence > 0.75 and mtr_error > 0.5:
            self.false_positive_candidates.append({
                'grain_id': grain_id,
                'query': query,
                'grain_confidence': grain_confidence,
                'mtr_error': mtr_error,
                'timestamp': datetime.now().isoformat(),
            })
            return True
        return False
    
    def detect_collision(self, returned_grain_id: str, query: str) -> None:
        """
        Log a collision: when high-confidence alternatives exist for same query.
        Used to detect confusable grains (for Dream Bucket).
        
        Args:
            returned_grain_id: Grain we selected
            query: Query that triggered selection
        """
        # Find other high-confidence grains that could have matched
        other_high_conf_grains = []
        returned_grain = self.grains.get(returned_grain_id, {})
        returned_conf = returned_grain.get('confidence', 0.0)
        
        for gid, grain in self.grains.items():
            if gid == returned_grain_id:
                continue
            
            conf = grain.get('confidence', 0.0)
            # Consider it a collision if confidence is close
            if conf > 0.7 and abs(conf - returned_conf) < 0.15:
                other_high_conf_grains.append(gid)
        
        if not other_high_conf_grains:
            return
        
        cluster_key = tuple(sorted([returned_grain_id] + other_high_conf_grains))
        
        if cluster_key not in self.collision_clusters:
            self.collision_clusters[cluster_key] = {
                'count': 0,
                'queries': [],
                'first_seen': datetime.now().isoformat(),
                'last_seen': datetime.now().isoformat(),
            }
        
        self.collision_clusters[cluster_key]['count'] += 1
        self.collision_clusters[cluster_key]['queries'].append(query)
        self.collision_clusters[cluster_key]['last_seen'] = datetime.now().isoformat()
    
    def get_false_positive_report(self) -> List[Dict]:
        """Get all detected false positives (for Dream Bucket logging)."""
        return self.false_positive_candidates.copy()
    
    def get_collision_cluster_report(self) -> Dict:
        """Get all detected collision clusters (for Dream Bucket logging)."""
        return dict(self.collision_clusters)
    
    def clear_false_positive_log(self) -> None:
        """Clear false positive candidates after logging to Dream Bucket."""
        self.false_positive_candidates.clear()
    
    # ========================================================================
    # PHASE 1.75: CONFIDENCE UPDATES (from MTR feedback)
    # ========================================================================
    
    def update_grain_confidence(self, grain_id: str, mtr_error: float) -> None:
        """
        Dynamically adjust grain confidence based on MTR feedback.
        
        Args:
            grain_id: Grain to update
            mtr_error: MTR error signal
        """
        if grain_id not in self.grains:
            return
        
        grain = self.grains[grain_id]
        current_conf = grain.get('confidence', 0.75)
        
        # Reward: MTR likes this grain (low error)
        if mtr_error < 0.2:
            new_conf = min(current_conf + 0.02, 0.99)
            grain['confidence'] = new_conf
        
        # Penalty: MTR struggled (high error)
        elif mtr_error > 0.5:
            new_conf = max(current_conf - 0.03, 0.5)
            grain['confidence'] = new_conf
        
        # Else: no change for moderate error (0.2 - 0.5)
    
    # ========================================================================
    # PATCH_3: LOG GRAIN USAGE BACK TO CARTRIDGE ENGINE FOR UNIFIED LEARNING
    # ========================================================================
    
    def log_grain_usage(self, grain_id: str, success: bool = True, 
                        mtr_error: float = 0.0, context: str = "general") -> None:
        """
        Log that a grain was used (success or failure).
        
        Maps grain usage back to source fact_id and logs to cartridge_engine.
        This implements Option A: grains contribute to their source facts' learning.
        
        PATCH_3: This enables unified learning where:
        - fact_graph edges track both cartridge + grain usage
        - query anchors accumulate across both layers
        - CTR reflects combined success rate
        - seasonality captures all access patterns
        
        Args:
            grain_id: ID of grain that was used
            success: Whether it contributed to good inference
            mtr_error: Error signal from MTR
            context: Usage context (project name, etc.)
        """
        # If no cartridge_engine connected, skip logging
        if self.cartridge_engine is None:
            return
        
        # Look up source fact_id from grain metadata
        grain = self.grains.get(grain_id)
        if grain is None:
            return
        
        fact_id = grain.get('fact_id')
        if fact_id is None:
            return
        
        # Log to cartridge_engine - same fact_id means same learning
        # This way:
        # - fact_graph edges track both cartridge + grain usage
        # - query anchors accumulate across both layers
        # - CTR reflects combined success rate
        # - seasonality captures all access patterns
        self.cartridge_engine.log_fact_usage(
            fact_id=fact_id,
            success=success,
            mtr_error=mtr_error,
            context=context
        )
    
    def log_mtr_feedback(self, query_text: str, returned_grain_id: str,
                         error_signal: float, session_id: str = None) -> None:
        """
        Log false positive to dream bucket when MTR disagrees with grain lookup.
        
        Called after MTR provides error_signal feedback.
        Logs if: grain confidence is high (>0.8) but MTR error is high (>0.3).
        
        Args:
            query_text: User query text
            returned_grain_id: Grain ID we returned
            error_signal: MTR error signal (higher = more confused)
            session_id: Optional session identifier
        """
        if self.dream_bucket_writer is None:
            return
        
        grain = self.grains.get(returned_grain_id)
        if grain is None:
            return
        
        grain_confidence = grain.get('confidence', 0.5)
        fact_id = grain.get('fact_id')
        
        # Log if confidence/error mismatch
        if grain_confidence > 0.8 and error_signal > 0.3:
            from dream_bucket import log_false_positive
            log_false_positive(
                self.dream_bucket_writer,
                source_layer="grain",
                query_text=query_text,
                returned_id=fact_id if fact_id else int(returned_grain_id),
                returned_confidence=grain_confidence,
                error_signal=error_signal,
                session_id=session_id
            )
    
    # ========================================================================
    # DIAGNOSTICS & STATISTICS
    # ========================================================================
    
    def print_statistics(self) -> None:
        """Print router statistics including Phase 1-2 metrics."""
        print("\n" + "="*70)
        print("GRAIN ROUTER STATISTICS (with Phase 1-2 Learning)")
        print("="*70)
        
        print("\nBasic Stats:")
        print(f"  Grains loaded: {self.total_grains}")
        print(f"  Total storage: {self.total_size_bytes:,} bytes")
        print(f"  Load time: {self.load_time_ms:.1f}ms")
        
        # Graph stats
        graph_density = self.get_graph_density()
        print(f"\nPhase 1 Graph Learning:")
        print(f"  Nodes (grains with edges): {graph_density['nodes']}")
        print(f"  Edges (relationships): {graph_density['edges']}")
        print(f"  Avg neighbors per grain: {graph_density['avg_neighbors']:.2f}")
        print(f"  Graph density: {graph_density['density']:.4f}")
        
        # CTR stats
        ctr_stats = self.get_ctr_stats()
        print(f"\nPhase 1.5 CTR Tracking:")
        print(f"  Grains tracked: {ctr_stats['total_grains_tracked']}")
        print(f"  Total selections: {ctr_stats['total_selections']}")
        print(f"  Overall CTR: {ctr_stats['avg_ctr']:.2%}")
        
        if ctr_stats['high_performers']:
            print(f"  High performers (80%+ CTR):")
            for gid, ctr, uses in ctr_stats['high_performers'][:5]:
                print(f"    - {gid}: {ctr:.1%} ({uses} uses)")
        
        # False positive stats
        print(f"\nPhase 2 False Positive Detection:")
        print(f"  Detected this session: {len(self.false_positive_candidates)}")
        print(f"  Collision clusters: {len(self.collision_clusters)}")
        
        if self.collision_clusters:
            print(f"  Top clusters:")
            sorted_clusters = sorted(
                self.collision_clusters.items(),
                key=lambda x: x[1]['count'],
                reverse=True
            )
            for cluster_ids, data in sorted_clusters[:3]:
                print(f"    - {cluster_ids}: {data['count']} activations")
        
        # Confidence distribution
        confidences = [g.get('confidence', 0.0) for g in self.grains.values()]
        if confidences:
            print(f"\nConfidence Distribution:")
            print(f"  Min: {min(confidences):.4f}")
            print(f"  Avg: {sum(confidences) / len(confidences):.4f}")
            print(f"  Max: {max(confidences):.4f}")
        
        print("="*70 + "\n")
    
    def print_graph_report(self) -> None:
        """Print detailed Phase 1 graph report."""
        density = self.get_graph_density()
        
        print("\n" + "="*70)
        print("PHASE 1: GRAIN CO-OCCURRENCE GRAPH REPORT")
        print("="*70)
        print(f"Nodes (facts with edges): {density['nodes']}")
        print(f"Edges (relationships): {density['edges']}")
        print(f"Avg neighbors per fact: {density['avg_neighbors']:.2f}")
        print(f"Graph density: {density['density']:.4f}")
        print(f"Last updated: {self.grain_graph_updated.isoformat()}")
        
        print("\nInterpretation:")
        print(f"  - {density['nodes']} grains have semantic relationships")
        if density['avg_neighbors'] > 0:
            print(f"  - Each grain has ~{density['avg_neighbors']:.1f} related grains on average")
        else:
            print("  - Graph is still being populated (run more queries)")
        
        # Show some sample edges
        if self.grain_graph:
            print("\nSample relationships:")
            for i, (gid, neighbors) in enumerate(list(self.grain_graph.items())[:3]):
                neighbor_list = ', '.join(list(neighbors)[:3])
                print(f"  {gid} → [{neighbor_list}]")
        
        print("="*70 + "\n")


# Example usage and testing
if __name__ == "__main__":
    print("Initializing GrainRouter with Phase 1-2 Learning...")
    router = GrainRouter('./cartridges')
    
    print(f"✓ Loaded {router.total_grains} grains in {router.load_time_ms:.1f}ms")
    
    # Print statistics
    router.print_statistics()
    
    # Simulate some queries with Phase 1-2 learning
    print("\nSimulating 5 queries with Phase 1-2 feedback loop:")
    print("-" * 70)
    
    recent_grains = []
    
    for turn in range(1, 6):
        print(f"\nQuery {turn}:")
        
        # Simulate grain selection
        results = router.search_grains(['test', 'query'], recent_grains=recent_grains)
        if results:
            grain_id, score = results[0]
            print(f"  Selected: {grain_id} (score: {score:.3f})")
            
            # Simulate MTR feedback (random-like pattern)
            mtr_error = 0.15 + (turn * 0.05)  # Decreasing error over time
            print(f"  MTR error: {mtr_error:.3f}")
            
            # Log outcomes
            router.log_grain_co_occurrence([grain_id])
            router.log_grain_outcome(grain_id, mtr_error)
            router.update_grain_confidence(grain_id, mtr_error)
            
            # Check for false positive
            grain = router.grains[grain_id]
            is_fp = router.detect_false_positive(grain_id, "test query", 
                                                grain.get('confidence', 0.75), mtr_error)
            if is_fp:
                print(f"  ⚠ FALSE POSITIVE DETECTED")
            
            recent_grains.append(grain_id)
        else:
            print("  No grains available")
    
    print("\n" + "="*70)
    print("Final Statistics After Learning:")
    router.print_statistics()
    router.print_graph_report()
    
    print("\nFalse Positives Detected:")
    fp_report = router.get_false_positive_report()
    if fp_report:
        for fp in fp_report:
            print(f"  - {fp['grain_id']}: error={fp['mtr_error']:.3f}")
    else:
        print("  (None)")
    
    print("\n✓ GrainRouter Phase 1-2 ready for integration with Phase3EOrchestrator")
