"""
Cartridge Engine - Phase 3B Layer 2 Wrapper

Wraps the existing Cartridge system (10 domain knowledge bases)
in the InferenceEngine interface for use by QueryOrchestrator.

Provides keyword-based fact lookup across all cartridges.
Faster than semantic search, slower than grain lookup but broader coverage.

Used by: QueryOrchestrator (second/third layer)
Wraps: kitbash_cartridge.Cartridge (existing Phase 2C code)
"""

import time
import logging
from typing import Optional, Dict, Any, List, Set
from pathlib import Path

from interfaces.inference_engine import InferenceEngine, InferenceRequest, InferenceResponse
from structured_logger import get_event_logger

logger = logging.getLogger(__name__)


class CartridgeEngine(InferenceEngine):
    """
    Layer 2 Inference Engine - Cartridge-Based Fact Lookup.
    
    Wraps the Cartridge system to provide keyword-based fact search
    across 10 domain knowledge bases.
    
    Performance:
    - Load time: ~50-100ms (all cartridges)
    - Query latency: 15-50ms (keyword search + ranking)
    - Hit rate: ~70-80% on typical queries
    - Confidence: 0.70-0.92 (depends on cartridge quality)
    
    Architecture:
    - Loads all 10 cartridges (physics, chemistry, biology, etc.)
    - Searches by keywords/concepts
    - Returns facts with associated confidence
    - Ranks results by confidence and relevance
    """
    
    engine_name = "CARTRIDGE"
    
    def __init__(self, cartridges_dir: str = "./cartridges"):
        """
        Initialize Cartridge Engine.
        
        Args:
            cartridges_dir: Path to cartridges directory
        
        Raises:
            RuntimeError: If cartridge loading fails
        """
        super().__init__()
        
        self.cartridges_dir = Path(cartridges_dir)
        self.cartridges: Dict[str, Any] = {}
        self.is_loaded = False
        
        # Statistics
        self.query_count = 0
        self.cartridge_hits = 0
        self.multi_cartridge_hits = 0
        self.no_matches = 0
        self.total_latency_ms = 0.0
        self.load_time_ms = 0.0
        
        # NEW: Initialize event logger
        self.logger = get_event_logger("cartridge_engine")
        
        # Try to load cartridges
        self._load_cartridges()
    
    def _load_cartridges(self) -> None:
        """Load all cartridges from disk."""
        try:
            # Import here to avoid circular dependency
            from kitbash_cartridge import Cartridge
            
            logger.info(f"Loading cartridges from {self.cartridges_dir}...")
            
            start_time = time.perf_counter()
            
            if not self.cartridges_dir.exists():
                raise RuntimeError(f"Cartridges directory not found: {self.cartridges_dir}")
            
            # Load all .kbc directories
            for kbc_path in sorted(self.cartridges_dir.glob("*.kbc")):
                cartridge_name = kbc_path.stem
                
                try:
                    cartridge = Cartridge(cartridge_name, str(self.cartridges_dir))
                    cartridge.load()
                    self.cartridges[cartridge_name] = cartridge
                    
                    fact_count = len(cartridge.facts)
                    logger.debug(f"  Loaded {cartridge_name}: {fact_count} facts")
                
                except Exception as e:
                    logger.warning(f"  Failed to load {cartridge_name}: {e}")
            
            self.load_time_ms = (time.perf_counter() - start_time) * 1000
            self.is_loaded = len(self.cartridges) > 0
            
            if self.is_loaded:
                total_facts = sum(len(c.facts) for c in self.cartridges.values())
                logger.info(
                    f"✓ Loaded {len(self.cartridges)} cartridges with {total_facts} facts "
                    f"in {self.load_time_ms:.1f}ms"
                )
            else:
                raise RuntimeError("No cartridges loaded successfully")
        
        except ImportError as e:
            logger.error(f"Could not import Cartridge: {e}")
            raise RuntimeError(f"Cartridge import failed: {e}")
        
        except Exception as e:
            logger.error(f"Failed to load cartridges: {e}")
            raise RuntimeError(f"Cartridge loading failed: {e}")
    
    def is_available(self) -> bool:
        """Check if cartridge engine is ready to use."""
        return self.is_loaded and len(self.cartridges) > 0
    
    def query(self, request: InferenceRequest) -> InferenceResponse:
        """
        Execute a cartridge-based query.
        
        Args:
            request: InferenceRequest with user query
        
        Returns:
            InferenceResponse with answer from best matching fact
        
        Raises:
            RuntimeError: If engine not loaded
        """
        if not self.is_available():
            raise RuntimeError("CartridgeEngine not loaded - check cartridge availability")
        
        start_time = time.perf_counter()
        self.query_count += 1
        
        # Search across all cartridges
        results = self._search_cartridges(request.user_query)
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        self.total_latency_ms += latency_ms
        
        if results['best_match']:
            match = results['best_match']
            
            # Track if single or multi-cartridge hit
            if len(results['cartridges_with_hits']) > 1:
                self.multi_cartridge_hits += 1
            else:
                self.cartridge_hits += 1
            
            # NEW: Log cartridge hit
            self.logger.log(
                event_type="cartridge_hit",
                data={
                    "cartridges_searched": len(self.cartridges),
                    "cartridges_with_hits": len(results['cartridges_with_hits']),
                    "total_hits": len(results['all_hits']),
                    "confidence": match['confidence'],
                    "latency_ms": latency_ms,
                    "query_count": self.query_count
                },
                category="layer_execution"
            )
            
            return InferenceResponse(
                answer=match['answer'],
                confidence=match['confidence'],
                engine_name=self.engine_name,
                sources=[f"fact_{match['fact_id']}"],
                latency_ms=latency_ms,
                metadata={
                    'fact_id': match['fact_id'],
                    'cartridge': match['cartridge'],
                    'total_hits': len(results['all_hits']),
                    'cartridges_searched': len(self.cartridges),
                    'cartridges_with_hits': len(results['cartridges_with_hits']),
                    'query_count': self.query_count,
                }
            )
        
        else:
            # No matches found
            self.no_matches += 1
            
            # NEW: Log cartridge miss
            self.logger.log(
                event_type="cartridge_miss",
                data={
                    "cartridges_searched": len(self.cartridges),
                    "latency_ms": latency_ms,
                    "query_count": self.query_count
                },
                category="layer_execution"
            )
            
            return InferenceResponse(
                answer="[No cartridge matches - escalate to next layer]",
                confidence=0.0,  # Signals to QueryOrchestrator to try next layer
                engine_name=self.engine_name,
                sources=[],
                latency_ms=latency_ms,
                metadata={
                    'reason': 'no_cartridge_matches',
                    'cartridges_searched': len(self.cartridges),
                    'query_count': self.query_count,
                }
            )
    
    def _search_cartridges(self, user_query: str) -> Dict[str, Any]:
        """
        Search across all cartridges using keyword matching.
        
        Args:
            user_query: User's natural language query
        
        Returns:
            Dict with: best_match (best result), all_hits (all results),
                       cartridges_with_hits (which cartridges had matches)
        """
        result = {
            'best_match': None,
            'all_hits': [],
            'cartridges_with_hits': set(),
        }
        
        # Search each cartridge
        for cartridge_name, cartridge in self.cartridges.items():
            try:
                # Use cartridge's built-in query method
                fact_ids = cartridge.query(user_query, log_access=True)
                
                if fact_ids:
                    result['cartridges_with_hits'].add(cartridge_name)
                    
                    # Get confidence for each result
                    for fact_id in fact_ids:
                        confidence = self._get_fact_confidence(
                            cartridge, fact_id
                        )
                        
                        # Get fact text
                        fact_text = cartridge.get_fact(fact_id)
                        if not fact_text:
                            fact_text = f"[Fact {fact_id} from {cartridge_name}]"
                        
                        hit = {
                            'fact_id': fact_id,
                            'cartridge': cartridge_name,
                            'answer': fact_text,
                            'confidence': confidence,
                        }
                        
                        result['all_hits'].append(hit)
            
            except Exception as e:
                logger.warning(f"Error querying {cartridge_name}: {e}")
        
        # Select best match (highest confidence)
        if result['all_hits']:
            result['best_match'] = max(result['all_hits'], key=lambda x: x['confidence'])
        
        return result
    
    def _get_fact_confidence(self, cartridge: Any, fact_id: int) -> float:
        """
        Get confidence for a fact from cartridge annotations.
        
        Args:
            cartridge: Cartridge instance
            fact_id: Fact ID
        
        Returns:
            Confidence score (0.0-1.0)
        """
        try:
            if hasattr(cartridge, 'annotations') and fact_id in cartridge.annotations:
                # Check if annotations has confidence attribute
                annotation = cartridge.annotations[fact_id]
                if hasattr(annotation, 'confidence'):
                    return annotation.confidence
                elif isinstance(annotation, dict) and 'confidence' in annotation:
                    return annotation['confidence']
        except Exception:
            pass
        
        # Default confidence if not found
        return 0.70
    
    def get_stats(self) -> Dict[str, Any]:
        """Return cartridge engine statistics."""
        avg_latency = (
            self.total_latency_ms / self.query_count
            if self.query_count > 0
            else 0.0
        )
        
        hit_rate = (
            ((self.cartridge_hits + self.multi_cartridge_hits) / self.query_count) * 100
            if self.query_count > 0
            else 0.0
        )
        
        total_facts = sum(len(c.facts) for c in self.cartridges.values())
        
        return {
            'query_count': self.query_count,
            'cartridge_hits': self.cartridge_hits,
            'multi_cartridge_hits': self.multi_cartridge_hits,
            'no_matches': self.no_matches,
            'hit_rate_percent': hit_rate,
            'avg_latency_ms': avg_latency,
            'total_latency_ms': self.total_latency_ms,
            'cartridges_loaded': len(self.cartridges),
            'total_facts': total_facts,
            'load_time_ms': self.load_time_ms,
        }
    
    def shutdown(self) -> None:
        """Clean up resources."""
        self.cartridges = {}
        self.is_loaded = False
        logger.info("CartridgeEngine shut down")


# ============================================================================
# EXAMPLE USAGE & TESTING
# ============================================================================

if __name__ == "__main__":
    """
    Example showing how the cartridge engine works.
    """
    
    print("="*70)
    print("CARTRIDGE ENGINE - EXAMPLE USAGE")
    print("="*70)
    
    # Initialize cartridge engine
    try:
        engine = CartridgeEngine('./cartridges')
        print(f"\n✓ Initialized CartridgeEngine")
        print(f"  Engine name: {engine.engine_name}")
        print(f"  Available: {engine.is_available()}")
        print(f"  Cartridges loaded: {len(engine.cartridges)}")
        print(f"  Cartridge names: {', '.join(sorted(engine.cartridges.keys()))}")
        
    except RuntimeError as e:
        print(f"\n✗ Failed to initialize: {e}")
        print("  (Make sure ./cartridges/ directory exists with .kbc directories)")
        exit(1)
    
    # Test queries
    test_queries = [
        "What is photosynthesis?",
        "Explain cellular respiration",
        "Tell me about physics and motion",
        "What are enzymes?",
        "Something completely unrelated to knowledge base",
    ]
    
    print("\n" + "="*70)
    print("CARTRIDGE QUERY EXAMPLES")
    print("="*70)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        
        request = InferenceRequest(user_query=query)
        
        try:
            response = engine.query(request)
            
            if response.confidence > 0.0:
                print(f"  Status: MATCH")
                print(f"  Confidence: {response.confidence:.4f}")
                print(f"  Latency: {response.latency_ms:.2f}ms")
                print(f"  Cartridge: {response.metadata.get('cartridge', 'N/A')}")
                print(f"  Sources: {response.sources}")
                print(f"  Total hits in search: {response.metadata.get('total_hits', 0)}")
                print(f"  Answer: {response.answer[:80]}...")
            else:
                print(f"  Status: NO MATCH (escalate)")
                print(f"  Latency: {response.latency_ms:.2f}ms")
        
        except Exception as e:
            print(f"  Status: ERROR - {e}")
    
    # Print statistics
    print("\n" + "="*70)
    print("CARTRIDGE ENGINE STATISTICS")
    print("="*70)
    
    stats = engine.get_stats()
    print(f"Queries processed: {stats['query_count']}")
    print(f"Single cartridge hits: {stats['cartridge_hits']}")
    print(f"Multi-cartridge hits: {stats['multi_cartridge_hits']}")
    print(f"No matches: {stats['no_matches']}")
    
    if stats['query_count'] > 0:
        print(f"\nOverall hit rate: {stats['hit_rate_percent']:.1f}%")
        print(f"Avg latency: {stats['avg_latency_ms']:.2f}ms")
    
    print(f"\nCartridges loaded: {stats['cartridges_loaded']}")
    print(f"Total facts indexed: {stats['total_facts']}")
    print(f"Load time: {stats['load_time_ms']:.1f}ms")
    
    # Show per-cartridge breakdown
    print(f"\nPer-Cartridge Breakdown:")
    for name, cart in sorted(engine.cartridges.items()):
        fact_count = len(cart.facts)
        print(f"  {name:20} {fact_count:4d} facts")
    
    print("\n✓ CartridgeEngine example complete")
