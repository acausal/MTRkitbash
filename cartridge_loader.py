#!/usr/bin/env python3
"""
CartridgeLoader - Phase 3E MVP + Google Search Integration

Integrates four proven abandoned Google algorithms:
1. Phase 1: Fact Co-occurrence Graph (2006) - semantic clustering
2. Phase 2: Query Anchor Profiles (2005) - query→fact mappings
3. Phase 3: CTR + Freshness (2003-2010) - feedback learning
4. Phase 4: Temporal Seasonality (2007) - project cycle detection

All techniques feed on honest signals (no adversarial gaming):
- Co-occurrence: facts used together
- Anchors: queries that successfully retrieved facts
- CTR: user feedback on fact usefulness
- Seasonality: temporal access patterns

Author: Kitbash Team
Date: February 23, 2026
"""

import sqlite3
import json
import time
import logging
import re
import math
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime, timedelta


logger = logging.getLogger(__name__)


@dataclass
class Fact:
    """Single fact from cartridge"""
    fact_id: int
    content: str
    confidence: float = 0.75
    domain: str = ""
    cartridge: str = ""


# ============================================================================
# PHASE 4: TEMPORAL SEASONALITY
# ============================================================================

class TemporalSeasonality:
    """
    Track seasonal usage patterns for facts.
    
    Detects when certain facts are predominantly accessed in specific months,
    enabling relevance ranking that matches project/interest cycles.
    """
    
    def __init__(self):
        """Initialize seasonality tracker."""
        self.fact_season_matrix = defaultdict(lambda: defaultdict(int))
        # fact_id → {month_name: count, ...}
    
    def record_usage(self, fact_id: int, timestamp: datetime = None) -> None:
        """
        Log when fact was used.
        
        Args:
            fact_id: Fact that was accessed
            timestamp: When it was accessed (default: now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        month_name = timestamp.strftime("%B")  # "January", "February", etc.
        self.fact_season_matrix[fact_id][month_name] += 1
    
    def is_seasonal(self, fact_id: int, threshold: float = 0.5) -> bool:
        """
        Check if fact has strong seasonal signal.
        
        Args:
            fact_id: Fact to check
            threshold: Concentration threshold (0.5 = 50% in one month)
        
        Returns:
            True if fact is concentrated in specific months
        """
        if fact_id not in self.fact_season_matrix:
            return False
        
        seasons = self.fact_season_matrix[fact_id]
        if not seasons:
            return False
        
        total = sum(seasons.values())
        max_concentration = max(seasons.values()) / total if total > 0 else 0.0
        
        return max_concentration >= threshold
    
    def find_peak_season(self, fact_id: int) -> Optional[str]:
        """Find month with highest usage for a fact."""
        if fact_id not in self.fact_season_matrix:
            return None
        
        seasons = self.fact_season_matrix[fact_id]
        if not seasons:
            return None
        
        peak_month = max(seasons.items(), key=lambda x: x[1])[0]
        return peak_month
    
    def get_seasonality_score(self, fact_id: int, month: str) -> float:
        """Get concentration score for fact in specific month."""
        if fact_id not in self.fact_season_matrix:
            return 0.0
        
        seasons = self.fact_season_matrix[fact_id]
        total = sum(seasons.values())
        
        if total == 0:
            return 0.0
        
        return seasons.get(month, 0) / total


# ============================================================================
# CARTRIDGE LOADER
# ============================================================================

class CartridgeLoader:
    """
    Loads and manages a single cartridge.
    
    Responsibilities:
    - Load facts from facts.db
    - Load annotations from annotations.jsonl
    - Build keyword index for search
    - Provide O(log N) fact lookup
    """
    
    def __init__(self, cartridge_path: Path):
        """
        Initialize loader for a single cartridge.
        
        Args:
            cartridge_path: Path to {name}_{faction}.kbc directory
        """
        self.cartridge_path = cartridge_path
        self.cartridge_id = cartridge_path.name.replace('.kbc', '')
        self.domain = self.cartridge_id.split('_')[0]  # Extract domain from name
        self.faction = self.cartridge_id.split('_')[1] if '_' in self.cartridge_id else 'general'
        
        # Storage
        self.facts: Dict[int, Fact] = {}
        self.annotations: Dict[int, Dict] = {}
        self.keyword_index: Dict[str, Set[int]] = {}  # word -> set of fact_ids
        
        self.is_loaded = False
        self.load_time_ms = 0.0
        
        # Load cartridge
        self._load()
    
    def _load(self) -> None:
        """Load all cartridge data from disk."""
        start_time = time.perf_counter()
        
        try:
            # Load facts from SQLite
            self._load_facts_from_db()
            
            # Load annotations from JSONL
            self._load_annotations()
            
            # Build keyword index
            self._build_keyword_index()
            
            self.is_loaded = True
            self.load_time_ms = (time.perf_counter() - start_time) * 1000
            
            logger.debug(
                f"✓ Loaded {self.cartridge_id}: {len(self.facts)} facts, "
                f"{len(self.keyword_index)} keywords in {self.load_time_ms:.1f}ms"
            )
        
        except Exception as e:
            logger.error(f"✗ Failed to load {self.cartridge_id}: {e}")
            raise
    
    def _load_facts_from_db(self) -> None:
        """Load facts from facts.db SQLite database."""
        db_path = self.cartridge_path / "facts.db"
        
        if not db_path.exists():
            raise FileNotFoundError(f"facts.db not found in {self.cartridge_path}")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT id, content_hash, content, created_at, access_count, status
                FROM facts
                WHERE status = 'active'
                ORDER BY id
            """)
            
            for row in cursor.fetchall():
                fact_id, content_hash, content, created_at, access_count, status = row
                
                # Default confidence from cartridge metadata (will be overridden by annotations)
                self.facts[fact_id] = Fact(
                    fact_id=fact_id,
                    content=content,
                    confidence=0.75,  # Default
                    domain=self.domain,
                    cartridge=self.cartridge_id
                )
        
        finally:
            conn.close()
    
    def _load_annotations(self) -> None:
        """Load annotations from annotations.jsonl."""
        annotations_path = self.cartridge_path / "annotations.jsonl"
        
        if not annotations_path.exists():
            logger.warning(f"No annotations.jsonl found in {self.cartridge_path}")
            return
        
        try:
            with open(annotations_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    ann = json.loads(line)
                    fact_id = ann.get('fact_id')
                    
                    if fact_id not in self.annotations:
                        self.annotations[fact_id] = ann
                        
                        # Update fact confidence from annotation
                        if fact_id in self.facts:
                            metadata = ann.get('metadata', {})
                            confidence = metadata.get('confidence', 0.75)
                            self.facts[fact_id].confidence = confidence
        
        except Exception as e:
            logger.warning(f"Error loading annotations: {e}")
    
    def _build_keyword_index(self) -> None:
        """Build inverted keyword index from fact content."""
        for fact_id, fact in self.facts.items():
            # Extract words (3+ chars, alphanumeric)
            words = set()
            
            content_lower = fact.content.lower()
            
            # Split on whitespace and punctuation
            tokens = re.findall(r'\b\w{3,}\b', content_lower)
            
            for token in tokens:
                words.add(token)
            
            # Add to index
            for word in words:
                if word not in self.keyword_index:
                    self.keyword_index[word] = set()
                self.keyword_index[word].add(fact_id)
    
    def search(self, query: str, limit: int = 10) -> List[Fact]:
        """
        Search facts by keywords.
        
        Args:
            query: Search query (natural language)
            limit: Maximum results
        
        Returns:
            List of matching facts, sorted by relevance
        """
        # Extract keywords from query
        query_tokens = set(re.findall(r'\b\w{3,}\b', query.lower()))
        
        if not query_tokens:
            return []
        
        # Find facts matching all keywords (AND search)
        matching_fact_ids = None
        
        for token in query_tokens:
            if token in self.keyword_index:
                token_facts = self.keyword_index[token]
                
                if matching_fact_ids is None:
                    matching_fact_ids = token_facts.copy()
                else:
                    matching_fact_ids &= token_facts  # Intersection (AND)
        
        if not matching_fact_ids:
            return []
        
        # Convert to facts and sort by confidence
        results = [self.facts[fid] for fid in matching_fact_ids if fid in self.facts]
        results.sort(key=lambda f: f.confidence, reverse=True)
        
        return results[:limit]
    
    def get_fact(self, fact_id: int) -> Optional[Fact]:
        """Get a specific fact by ID."""
        return self.facts.get(fact_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cartridge statistics."""
        return {
            'cartridge_id': self.cartridge_id,
            'domain': self.domain,
            'faction': self.faction,
            'fact_count': len(self.facts),
            'keyword_count': len(self.keyword_index),
            'annotation_count': len(self.annotations),
            'avg_confidence': sum(f.confidence for f in self.facts.values()) / len(self.facts) if self.facts else 0.0,
            'load_time_ms': self.load_time_ms,
        }


# ============================================================================
# CARTRIDGE REGISTRY: ALL 4 TECHNIQUES INTEGRATED
# ============================================================================

class CartridgeRegistry:
    """
    Manages loading and querying multiple cartridges with 4 Google techniques.
    
    Provides:
    - Bulk loading of all cartridges
    - Cross-cartridge search with learned ranking
    - Per-cartridge lookup
    - Learning from MTR feedback (co-occurrence, anchors, CTR, seasonality)
    """
    
    def __init__(self, cartridges_dir: str = "./cartridges"):
        """
        Initialize registry with all learning infrastructure.
        
        Args:
            cartridges_dir: Path to cartridges directory
        """
        self.cartridges_dir = Path(cartridges_dir)
        self.cartridges: Dict[str, CartridgeLoader] = {}
        self.load_time_ms = 0.0
        
        # ====================================================================
        # PHASE 1: FACT CO-OCCURRENCE GRAPH (2006)
        # ====================================================================
        self.fact_graph = defaultdict(set)  # fact_id → set of related fact_ids
        self.fact_graph_updated = datetime.now()
        
        # ====================================================================
        # PHASE 2: QUERY ANCHOR PROFILES (2005)
        # ====================================================================
        self.query_anchor_profile = defaultdict(set)  # fact_id → set of query terms
        self.anchor_updated = datetime.now()
        
        # ====================================================================
        # PHASE 3: CTR + FRESHNESS (2003-2010)
        # ====================================================================
        self.fact_usage_stats = {}  # fact_id → {used, last_used, success_count, contexts}
        self.ctr_threshold = 0.25  # MTR error < this = successful use
        self.freshness_decay_days = 30  # Facts older than this get freshness penalty
        
        # ====================================================================
        # PHASE 4: TEMPORAL SEASONALITY (2007)
        # ====================================================================
        self.temporality = TemporalSeasonality()
        self.seasonality_updated = datetime.now()
        
        self._load_all()
    
    def _load_all(self) -> None:
        """Load all cartridges from directory."""
        if not self.cartridges_dir.exists():
            raise FileNotFoundError(f"Cartridges directory not found: {self.cartridges_dir}")
        
        start_time = time.perf_counter()
        
        # Find all .kbc directories
        kbc_paths = sorted(self.cartridges_dir.glob("*.kbc"))
        
        if not kbc_paths:
            raise RuntimeError(f"No .kbc cartridges found in {self.cartridges_dir}")
        
        logger.info(f"Loading {len(kbc_paths)} cartridges from {self.cartridges_dir}...")
        
        for kbc_path in kbc_paths:
            try:
                loader = CartridgeLoader(kbc_path)
                self.cartridges[loader.cartridge_id] = loader
            except Exception as e:
                logger.warning(f"Failed to load {kbc_path.name}: {e}")
        
        self.load_time_ms = (time.perf_counter() - start_time) * 1000
        
        if not self.cartridges:
            raise RuntimeError("No cartridges loaded successfully")
        
        total_facts = sum(len(c.facts) for c in self.cartridges.values())
        logger.info(
            f"✓ Loaded {len(self.cartridges)} cartridges with {total_facts} facts "
            f"in {self.load_time_ms:.1f}ms"
        )
    
    def search_all(self, query: str, limit: int = 10, recent_facts: List[int] = None) -> List[Tuple[str, Fact]]:
        """
        Search across all cartridges with all 4 techniques applied.
        
        Args:
            query: Search query
            limit: Maximum total results
            recent_facts: Facts from previous queries (for co-occurrence boost)
        
        Returns:
            List of (cartridge_id, fact) tuples, sorted by learned ranking
        """
        if recent_facts is None:
            recent_facts = []
        
        # Extract query terms for Phase 2 (anchors) and Phase 3 (CTR)
        query_terms = set(re.findall(r'\b\w{3,}\b', query.lower()))
        current_month = datetime.now().strftime("%B")
        
        all_results = []
        
        # Search all cartridges
        for cart_id, cartridge in self.cartridges.items():
            facts = cartridge.search(query, limit=100)  # Get more than needed per cartridge
            for fact in facts:
                all_results.append((cart_id, fact))
        
        # ====================================================================
        # APPLY ALL 4 TECHNIQUES TO SCORE
        # ====================================================================
        
        scored_results = []
        
        for cart_id, fact in all_results:
            # Base score (from keyword search)
            base_score = fact.confidence
            
            # PHASE 1: Graph boost (co-occurrence)
            graph_boost = self._graph_boost(fact.fact_id, recent_facts)
            
            # PHASE 2: Anchor boost (query history)
            anchor_boost = self._anchor_boost(fact.fact_id, query_terms)
            
            # PHASE 3: CTR boost + freshness penalty
            ctr_boost = self._ctr_boost(fact.fact_id)
            freshness_decay = self._freshness_decay(fact.fact_id)
            
            # PHASE 4: Seasonality boost (temporal relevance)
            seasonality_boost = self._seasonality_boost(fact.fact_id, current_month)
            
            # Combine scores with learned weights
            # Weights determined by importance of signal
            final_score = (
                base_score * 0.60 +           # Keyword matching (baseline)
                graph_boost * 0.15 +          # Co-occurrence (strong signal)
                anchor_boost * 0.10 +         # Query anchors (moderate signal)
                ctr_boost * 0.08 +            # CTR (moderate signal)
                freshness_decay * 0.04 +      # Freshness (weak signal)
                seasonality_boost * 0.03      # Seasonality (weak signal)
            )
            
            scored_results.append((final_score, cart_id, fact))
        
        # Sort by final score (descending)
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        # Return top results with scores stripped
        return [(cart_id, fact) for _, cart_id, fact in scored_results[:limit]]
    
    def search_cartridge(self, cartridge_id: str, query: str, limit: int = 10) -> List[Fact]:
        """Search within a specific cartridge."""
        if cartridge_id not in self.cartridges:
            return []
        
        return self.cartridges[cartridge_id].search(query, limit=limit)
    
    def get_cartridge(self, cartridge_id: str) -> Optional[CartridgeLoader]:
        """Get a specific cartridge loader."""
        return self.cartridges.get(cartridge_id)
    
    def get_fact_confidence(self, fact_id: int) -> float:
        """
        Get confidence score for a fact across all cartridges.
        
        Returns the confidence from whichever cartridge owns this fact.
        
        Args:
            fact_id: Fact to look up
        
        Returns:
            Confidence float (0.0-1.0), or 0.5 if not found (neutral)
        """
        for cartridge in self.cartridges.values():
            fact = cartridge.get_fact(fact_id)
            if fact:
                return fact.confidence
        return 0.5  # Unknown fact, neutral confidence
    
    # ========================================================================
    # PHASE 1: FACT CO-OCCURRENCE GRAPH
    # ========================================================================
    
    def log_fact_co_occurrence(self, facts_used: List[int]) -> None:
        """
        Log facts that co-occurred in same MTR inference epoch.
        
        Creates edges between facts that activate together, enabling
        semantic clustering discovery without manual annotation.
        
        Args:
            facts_used: List of fact IDs that were active in same epoch
        """
        if not facts_used or len(facts_used) < 2:
            return
        
        # Create bidirectional edges
        for fid in facts_used:
            for other_fid in facts_used:
                if fid != other_fid:
                    self.fact_graph[fid].add(other_fid)
        
        self.fact_graph_updated = datetime.now()
    
    def _graph_boost(self, fact_id: int, recent_facts: List[int]) -> float:
        """
        Boost score if this fact is adjacent in co-occurrence graph.
        
        Logic: Facts that co-activate are semantically related.
        Max boost: 0.5
        """
        if not recent_facts or fact_id not in self.fact_graph:
            return 0.0
        
        boost = 0.0
        neighbors = self.fact_graph.get(fact_id, set())
        
        for recent_fid in recent_facts:
            if recent_fid in neighbors:
                boost += 0.15
        
        return min(boost, 0.5)
    
    def get_graph_density(self) -> Dict[str, float]:
        """Get co-occurrence graph statistics."""
        if not self.fact_graph:
            return {
                'nodes': 0,
                'edges': 0,
                'avg_neighbors': 0.0,
                'density': 0.0,
            }
        
        total_edges = sum(len(neighbors) for neighbors in self.fact_graph.values())
        avg_neighbors = total_edges / max(len(self.fact_graph), 1)
        
        # Density: actual edges / possible edges
        n = len(self.fact_graph)
        possible_edges = n * (n - 1) if n > 0 else 1
        density = total_edges / possible_edges
        
        return {
            'nodes': len(self.fact_graph),
            'edges': total_edges,
            'avg_neighbors': avg_neighbors,
            'density': density,
        }
    
    def print_graph_report(self) -> None:
        """Print human-readable graph diagnostics"""
        density = self.get_graph_density()
        
        print("\n" + "="*70)
        print("FACT CO-OCCURRENCE GRAPH REPORT")
        print("="*70)
        print(f"Nodes (facts with edges): {density['nodes']}")
        print(f"Edges (relationships): {density['edges']}")
        print(f"Avg neighbors per fact: {density['avg_neighbors']:.2f}")
        print(f"Graph density: {density['density']:.4f}")
        print("\nInterpretation:")
        print(f"  - {density['nodes']} facts have semantic relationships")
        if density['avg_neighbors'] > 0:
            print(f"  - Each fact has ~{density['avg_neighbors']:.1f} related facts on average")
        else:
            print("  - Graph is still being populated (run more queries)")
        print("="*70 + "\n")
    
    # ========================================================================
    # PHASE 2: QUERY ANCHOR PROFILES
    # ========================================================================
    
    def log_query_anchor(self, fact_id: int, query_terms: List[str]) -> None:
        """
        Log a successful query-to-fact retrieval.
        
        Over time, builds a profile of which query terms successfully
        retrieved this fact, enabling synonym discovery.
        
        Args:
            fact_id: Fact that was retrieved
            query_terms: Query terms that matched it
        """
        if fact_id is None or not query_terms:
            return
        
        self.query_anchor_profile[fact_id].update(query_terms)
        self.anchor_updated = datetime.now()
    
    def _anchor_boost(self, fact_id: int, query_terms: Set[str]) -> float:
        """
        Boost if query terms match this fact's successful query history.
        
        Max boost: 0.3
        """
        if fact_id not in self.query_anchor_profile or not query_terms:
            return 0.0
        
        anchor_terms = self.query_anchor_profile[fact_id]
        matching = query_terms & anchor_terms
        
        if not matching:
            return 0.0
        
        # Log scale: more matches = stronger boost
        overlap_ratio = len(matching) / max(len(query_terms), 1)
        return min(0.3, math.log(1 + len(matching)) * overlap_ratio * 0.1)
    
    def get_anchor_report(self, fact_id: int = None) -> Dict:
        """Diagnostic: show query anchor profiles."""
        if fact_id is not None:
            return {
                'fact_id': fact_id,
                'anchor_terms': sorted(list(self.query_anchor_profile.get(fact_id, []))),
                'term_count': len(self.query_anchor_profile.get(fact_id, [])),
            }
        
        report = {}
        for fid, terms in self.query_anchor_profile.items():
            report[fid] = sorted(list(terms))
        return report
    
    # ========================================================================
    # PHASE 3: CTR + FRESHNESS
    # ========================================================================
    
    def log_fact_usage(self, fact_id: int, success: bool = True, 
                       mtr_error: float = 0.0, context: str = "general") -> None:
        """
        Log that a fact was used (success or failure).
        
        Called from orchestrator after MTR processes facts.
        
        Args:
            fact_id: Fact that was used
            success: Whether it was useful (determined by mtr_error < threshold)
            mtr_error: Error signal from MTR
            context: Usage context (project name, etc.)
        """
        if fact_id is None:
            return
        
        if fact_id not in self.fact_usage_stats:
            self.fact_usage_stats[fact_id] = {
                'used': 0,
                'last_used': None,
                'success_count': 0,
                'contexts': []
            }
        
        stats = self.fact_usage_stats[fact_id]
        stats['used'] += 1
        stats['last_used'] = datetime.now()
        
        if success or mtr_error < self.ctr_threshold:
            stats['success_count'] += 1
        
        stats['contexts'].append(context)
        
        # Also record for seasonality
        self.temporality.record_usage(fact_id)
        self.seasonality_updated = datetime.now()
    
    def _ctr_boost(self, fact_id: int) -> float:
        """
        Boost based on click-through rate (success ratio).
        
        Max boost: 0.25
        """
        if fact_id not in self.fact_usage_stats:
            return 0.0
        
        stats = self.fact_usage_stats[fact_id]
        used = stats['used']
        successful = stats['success_count']
        
        if used == 0:
            return 0.0
        
        ctr = successful / used
        # Boost scales with CTR: 0.0 (0% success) to 0.25 (100% success)
        return min(0.25, ctr * 0.25)
    
    def _freshness_decay(self, fact_id: int) -> float:
        """
        Penalty for stale facts (not recently used).
        
        Returns value between 0.0 (very stale) and 0.1 (fresh)
        """
        if fact_id not in self.fact_usage_stats:
            return 0.0
        
        last_used = self.fact_usage_stats[fact_id]['last_used']
        if last_used is None:
            return 0.0
        
        age_days = (datetime.now() - last_used).days
        
        # Decay from 0.1 (fresh) to 0.0 (very stale)
        if age_days <= self.freshness_decay_days:
            # Linear decay over 30 days
            return 0.1 * (1.0 - age_days / self.freshness_decay_days)
        else:
            return 0.0  # Very old facts get no boost
    
    # ========================================================================
    # PHASE 4: TEMPORAL SEASONALITY
    # ========================================================================
    
    def _seasonality_boost(self, fact_id: int, current_month: str) -> float:
        """
        Boost if fact is seasonally relevant (used frequently in this month).
        
        Max boost: 0.15
        """
        if fact_id not in self.temporality.fact_season_matrix:
            return 0.0
        
        seasonality_score = self.temporality.get_seasonality_score(fact_id, current_month)
        
        # Boost scales with seasonality concentration
        # 50% = 0.075, 75% = 0.1125, 90% = 0.135
        return min(0.15, seasonality_score * 0.15)
    
    def get_seasonal_report(self) -> Dict:
        """Diagnostic: show which facts are seasonal."""
        seasonal_facts = {}
        
        for fact_id in range(1, 10000):  # Scan all possible fact IDs
            if self.temporality.is_seasonal(fact_id):
                peak = self.temporality.find_peak_season(fact_id)
                seasonal_facts[fact_id] = {
                    'peak_season': peak,
                    'concentration': self.temporality.get_seasonality_score(fact_id, peak),
                }
        
        return seasonal_facts
    
    # ========================================================================
    # DIAGNOSTICS & PERSISTENCE
    # ========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        total_facts = sum(len(c.facts) for c in self.cartridges.values())
        avg_confidence = sum(
            sum(f.confidence for f in c.facts.values()) 
            for c in self.cartridges.values()
        ) / total_facts if total_facts else 0.0
        
        return {
            'cartridge_count': len(self.cartridges),
            'total_facts': total_facts,
            'total_keywords': sum(len(c.keyword_index) for c in self.cartridges.values()),
            'avg_confidence': avg_confidence,
            'load_time_ms': self.load_time_ms,
            'learning': {
                'fact_graph_nodes': len(self.fact_graph),
                'fact_graph_edges': sum(len(n) for n in self.fact_graph.values()),
                'query_anchors_tracked': len(self.query_anchor_profile),
                'facts_with_usage_stats': len(self.fact_usage_stats),
                'facts_with_seasonality': len(self.temporality.fact_season_matrix),
            },
            'cartridges': {
                cart_id: cart.get_stats()
                for cart_id, cart in self.cartridges.items()
            }
        }
    
    def get_learning_state(self) -> Dict[str, Any]:
        """
        Get all learning state for persistence.
        
        Returns format compatible with mtr_state_manager.py save/load.
        """
        return {
            'fact_graph': {
                int(k): sorted(list(v)) for k, v in self.fact_graph.items()
            },
            'query_anchor_profile': {
                int(k): sorted(list(v)) for k, v in self.query_anchor_profile.items()
            },
            'fact_usage_stats': {
                int(k): {
                    'used': v['used'],
                    'last_used': v['last_used'].isoformat() if v['last_used'] else None,
                    'success_count': v['success_count'],
                    'contexts': v['contexts'],
                } for k, v in self.fact_usage_stats.items()
            },
            'temporality_matrix': {
                int(k): dict(v) for k, v in self.temporality.fact_season_matrix.items()
            },
        }
    
    def restore_learning_state(self, state: Dict[str, Any]) -> None:
        """
        Restore all learning state from persistence.
        
        Called by mtr_state_manager.py after loading.
        """
        # Restore fact graph
        self.fact_graph = defaultdict(set)
        for k, v in state.get('fact_graph', {}).items():
            self.fact_graph[int(k)] = set(v)
        
        # Restore query anchors
        self.query_anchor_profile = defaultdict(set)
        for k, v in state.get('query_anchor_profile', {}).items():
            self.query_anchor_profile[int(k)] = set(v)
        
        # Restore usage stats
        self.fact_usage_stats = {}
        for k, v in state.get('fact_usage_stats', {}).items():
            self.fact_usage_stats[int(k)] = {
                'used': v['used'],
                'last_used': datetime.fromisoformat(v['last_used']) if v['last_used'] else None,
                'success_count': v['success_count'],
                'contexts': v['contexts'],
            }
        
        # Restore temporality
        self.temporality.fact_season_matrix = defaultdict(lambda: defaultdict(int))
        for k, v in state.get('temporality_matrix', {}).items():
            self.temporality.fact_season_matrix[int(k)] = defaultdict(int, v)
        
        logger.info(
            f"✓ Restored search learning state: "
            f"{len(self.fact_graph)} graph nodes, "
            f"{len(self.query_anchor_profile)} anchors, "
            f"{len(self.fact_usage_stats)} usage stats"
        )


# ============================================================================
# INFERENCE ENGINE INTERFACE
# ============================================================================

class CartridgeInferenceRequest:
    """Query request for inference engine"""
    def __init__(self, user_query: str):
        self.user_query = user_query


class CartridgeInferenceResponse:
    """Query response from inference engine"""
    def __init__(self, answer: str, confidence: float, cartridge: str, 
                 fact_id: int, latency_ms: float, sources: List[str]):
        self.answer = answer
        self.confidence = confidence
        self.cartridge = cartridge
        self.fact_id = fact_id
        self.latency_ms = latency_ms
        self.sources = sources
        self.metadata = {
            'cartridge': cartridge,
            'fact_id': fact_id,
        }


class CartridgeInferenceEngine:
    """
    Phase 3E Cartridge Inference Engine.
    
    Wraps CartridgeRegistry to provide query interface for MTR integration.
    Handles all learning feedback: co-occurrence, anchors, CTR, seasonality.
    
    Now with dream bucket integration: logs false positives when MTR feedback
    indicates confidence mismatch (high confidence search, high MTR error).
    """
    
    def __init__(self, cartridges_dir: str = "./cartridges", dream_bucket_writer=None):
        """
        Initialize engine and load all cartridges.
        
        Args:
            cartridges_dir: Path to cartridges directory
            dream_bucket_writer: Optional DreamBucketWriter for false positive logging
        """
        self.registry = CartridgeRegistry(cartridges_dir)
        self.query_count = 0
        self.hits = 0
        self.misses = 0
        self.dream_bucket_writer = dream_bucket_writer
        
        # Track recent facts for graph boost
        self._recent_facts_in_session = []
    
    def query(self, request: CartridgeInferenceRequest, limit: int = 5) -> Optional[CartridgeInferenceResponse]:
        """
        Execute a query across all cartridges with learned ranking.
        
        Applies all 4 techniques: co-occurrence, anchors, CTR, seasonality.
        
        Args:
            request: CartridgeInferenceRequest with user query
            limit: Maximum facts to search
        
        Returns:
            CartridgeInferenceResponse with best match, or None if no match
        """
        start_time = time.perf_counter()
        self.query_count += 1
        
        # Search with all learned techniques
        results = self.registry.search_all(
            request.user_query,
            limit=limit,
            recent_facts=self._recent_facts_in_session
        )
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        if results:
            # Return best match (highest learned score)
            best_cart_id, best_fact = results[0]
            self.hits += 1
            
            # Log query anchors (Phase 2)
            query_terms = set(re.findall(r'\b\w{3,}\b', request.user_query.lower()))
            self.registry.log_query_anchor(best_fact.fact_id, list(query_terms))
            
            # Log co-occurrence if we had recent facts (Phase 1)
            if self._recent_facts_in_session and best_fact.fact_id not in self._recent_facts_in_session:
                all_facts_used = self._recent_facts_in_session + [best_fact.fact_id]
                self.registry.log_fact_co_occurrence(all_facts_used)
            
            # Update recent facts for next query
            self._recent_facts_in_session = [best_fact.fact_id]
            
            return CartridgeInferenceResponse(
                answer=best_fact.content,
                confidence=best_fact.confidence,
                cartridge=best_cart_id,
                fact_id=best_fact.fact_id,
                latency_ms=latency_ms,
                sources=[f"fact_{best_fact.fact_id}_{best_cart_id}"]
            )
        else:
            self.misses += 1
            self._recent_facts_in_session = []  # Clear on miss
            return None
    
    def log_fact_usage(self, fact_id: int, success: bool = True, 
                      mtr_error: float = 0.0, context: str = "general") -> None:
        """
        Log that MTR successfully used a fact.
        
        Called from orchestrator after MTR inference completes.
        
        Args:
            fact_id: Fact that was used
            success: Whether it contributed to good inference
            mtr_error: Error signal from MTR (< 0.25 = success)
            context: What project/context was this in
        """
        self.registry.log_fact_usage(fact_id, success, mtr_error, context)
    
    def log_mtr_feedback(self, query_text: str, returned_id: int, 
                         error_signal: float, session_id: str = None) -> None:
        """
        Log false positive to dream bucket when MTR disagrees with search.
        
        Called after MTR provides error_signal feedback.
        Logs if: we were confident (>0.8) but MTR error is high (>0.3).
        
        Args:
            query_text: User query text
            returned_id: Fact ID we returned
            error_signal: MTR error signal (higher = more confused)
            session_id: Optional session identifier
        """
        if self.dream_bucket_writer is None:
            return
        
        returned_confidence = self.registry.get_fact_confidence(returned_id)
        
        # Log if confidence/error mismatch
        if returned_confidence > 0.8 and error_signal > 0.3:
            from dream_bucket import log_false_positive
            log_false_positive(
                self.dream_bucket_writer,
                source_layer="cartridge",
                query_text=query_text,
                returned_id=returned_id,
                returned_confidence=returned_confidence,
                error_signal=error_signal,
                session_id=session_id
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        hit_rate = (self.hits / self.query_count * 100) if self.query_count > 0 else 0.0
        
        return {
            'query_count': self.query_count,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate_percent': hit_rate,
            'registry': self.registry.get_stats(),
        }
    
    def get_learning_state(self) -> Dict[str, Any]:
        """Get all search learning state for persistence."""
        return self.registry.get_learning_state()
    
    def restore_learning_state(self, state: Dict[str, Any]) -> None:
        """Restore search learning state from persistence."""
        self.registry.restore_learning_state(state)
    
    # Delegation methods for diagnostics
    def log_fact_co_occurrence(self, facts_used: List[int]) -> None:
        """Log co-occurring facts (Phase 1)."""
        self.registry.log_fact_co_occurrence(facts_used)
    
    def get_graph_density(self) -> Dict[str, float]:
        """Get co-occurrence graph statistics (Phase 1)."""
        return self.registry.get_graph_density()
    
    def print_graph_report(self) -> None:
        """Print co-occurrence graph diagnostics (Phase 1)."""
        self.registry.print_graph_report()
    
    def get_anchor_report(self, fact_id: int = None) -> Dict:
        """Get query anchor diagnostics (Phase 2)."""
        return self.registry.get_anchor_report(fact_id)
    
    def get_seasonal_report(self) -> Dict:
        """Get seasonality diagnostics (Phase 4)."""
        return self.registry.get_seasonal_report()


# ============================================================================
# EXAMPLE USAGE & TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("CARTRIDGE LOADER - PHASE 3E WITH GOOGLE SEARCH INTEGRATION")
    print("="*70)
    
    try:
        # Initialize registry
        print("\nInitializing CartridgeRegistry with all 4 techniques...")
        registry = CartridgeRegistry('./inbox/cartridges')
        
        print(f"\n✓ Registry initialized")
        stats = registry.get_stats()
        print(f"  Cartridges: {stats['cartridge_count']}")
        print(f"  Total facts: {stats['total_facts']}")
        print(f"  Total keywords: {stats['total_keywords']}")
        print(f"  Avg confidence: {stats['avg_confidence']:.4f}")
        print(f"  Load time: {stats['load_time_ms']:.1f}ms")
        
        # Test queries
        print("\n" + "="*70)
        print("QUERY EXAMPLES")
        print("="*70)
        
        test_queries = [
            "What is physics?",
            "Tell me about biology",
            "What is mathematics?",
            "Explain chemistry",
            "Something about logic and reasoning",
        ]
        
        engine = CartridgeInferenceEngine('./inbox/cartridges')
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            
            request = CartridgeInferenceRequest(query)
            response = engine.query(request, limit=5)
            
            if response:
                print(f"  Status: MATCH")
                print(f"  Cartridge: {response.cartridge}")
                print(f"  Confidence: {response.confidence:.4f}")
                print(f"  Latency: {response.latency_ms:.2f}ms")
                print(f"  Answer: {response.answer[:70]}...")
                
                # Simulate MTR feedback
                engine.log_fact_usage(response.fact_id, success=True, mtr_error=0.15, context="test")
            else:
                print(f"  Status: NO MATCH")
        
        # Print learning diagnostics
        print("\n" + "="*70)
        print("LEARNING DIAGNOSTICS")
        print("="*70)
        
        # Phase 1: Graph
        engine.print_graph_report()
        
        # Phase 2: Anchors
        anchor_report = engine.get_anchor_report()
        if anchor_report:
            print("QUERY ANCHOR PROFILES (Phase 2):")
            for fact_id, terms in sorted(anchor_report.items())[:5]:
                print(f"  Fact {fact_id}: {terms}")
        
        # Phase 4: Seasonality
        seasonal = engine.get_seasonal_report()
        if seasonal:
            print("\nSEASONAL FACTS (Phase 4):")
            for fact_id, info in list(seasonal.items())[:5]:
                print(f"  Fact {fact_id}: Peak {info['peak_season']} ({info['concentration']:.1%})")
        
        # Print final stats
        print("\n" + "="*70)
        print("ENGINE STATISTICS")
        print("="*70)
        
        engine_stats = engine.get_stats()
        print(f"Queries: {engine_stats['query_count']}")
        print(f"Hits: {engine_stats['hits']}")
        print(f"Misses: {engine_stats['misses']}")
        print(f"Hit rate: {engine_stats['hit_rate_percent']:.1f}%")
        
        print("\n✓ CartridgeLoader test complete")
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
