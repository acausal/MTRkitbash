#!/usr/bin/env python3
"""
Phase 3E Integration: MTR 5.5 + CartridgeLoader + GrainSystem

Wires together:
- MTRHybridModel (inference/learning engine)
- CartridgeInferenceEngine (fact lookup with Phase 1-4 learning)
- GrainSystem (phantom tracking, crystallization pipeline)
- GrainRouter (grain-based Layer 0 reflex responses)
- MTRGrainBridge (unified pipeline coordination)

Full architecture:
1. Query arrives → CartridgeLoader searches facts + GrainRouter searches grains
2. Facts/grains fed to MTR as context
3. MTR processes with learned state (weight matrices, strength, recency)
4. MTR outputs + error signal → phantom tracker (via MTRPhantomBridge)
5. Every N queries → crystallization cycle (validation → crushing → persistence)
6. New grains loaded into GrainRouter's L3 cache
7. State saved periodically (MTR weights + cartridge learning + grain registry)

Performance:
- Cartridge lookup: 15-50ms
- Grain lookup: <1ms (L3 cache)
- MTR inference: 5-20ms
- Phantom tracking: <1ms
- Crystallization (every 100 queries): 200-500ms

Author: Kitbash Team
Date: February 2026
Phase: 3E Consolidated
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Set
from dataclasses import dataclass
import time

from MTR_v5_5_NN import KitbashMTREngine
from mtr_state_manager import MTRStateCheckpoint
from cartridge_loader import CartridgeInferenceEngine, CartridgeInferenceRequest

# Phase 2C Grain Integration
try:
    from grain_system import ShannonGrainOrchestrator
    from grain_router import GrainRouter
    from grain_activation import Hat
    from mtr_grain_bridge import MTRGrainUnifiedPipeline, HatKappaMapper
    GRAIN_SYSTEM_AVAILABLE = True
except ImportError as e:
    GRAIN_SYSTEM_AVAILABLE = False
    print(f"Warning: Grain system not available. Error: {e}")
    print("Running in MTR-only mode.")


@dataclass
class QueryContext:
    """Context for a single user query"""
    query_text: str
    user_id: Optional[str] = None
    project_context: Optional[str] = None
    session_id: Optional[str] = None
    hat: Optional[Any] = None  # Behavioral context for epistemic routing


@dataclass
class QueryResult:
    """Result of query processing"""
    user_query: str
    mtr_response: str
    mtr_confidence: float
    cartridge_facts: list
    grain_facts: list = None  # NEW: grain-based reflex results
    total_latency_ms: float = 0.0
    mtr_latency_ms: float = 0.0
    cartridge_latency_ms: float = 0.0
    grain_latency_ms: float = 0.0  # NEW
    crystallization_report: Optional[Dict] = None  # NEW


class Phase3EOrchestrator:
    """
    Main orchestration loop for Phase 3E with full grain integration.
    
    Steps:
    1. User sends query with context (text, project, hat)
    2. CartridgeLoader searches facts (learns co-occurrence, anchors, CTR, seasonality)
    3. GrainRouter searches crystallized grains (L3 cache lookup)
    4. MTR processes facts + grains with learned state
    5. MTR output fed to phantom tracker (via MTRPhantomBridge)
    6. Every 100 queries: crystallization cycle (phantom→grain)
    7. New grains loaded into GrainRouter cache
    8. State saved: MTR weights + cartridge learning + grain registry
    
    Architecture:
    - Stateful: MTR maintains weight matrices, phantom tracker maintains phantom history
    - Local-first: All learning stored locally, zero cloud dependencies
    - Auditable: Every component explains its existence and decisions
    - Unified: Grain system extends MTR learning, not separate
    """
    
    def __init__(self, 
                 cartridges_dir: str = "./cartridges",
                 vocab_size: int = 50257,
                 d_model: int = 256,
                 d_state: int = 144,
                 state_dir: str = "data/state",
                 grain_storage_dir: str = "./grains",
                 device: str = "cpu",
                 enable_grain_system: bool = True,
                 dream_bucket_dir: str = "data/subconscious/dream_bucket"):
        """
        Initialize Phase 3E orchestrator with full grain integration.
        
        Args:
            cartridges_dir: Path to cartridges (*.kbc directories)
            vocab_size: MTR model vocab size (default: GPT2)
            d_model: Model embedding dimension (256)
            d_state: State space dimension (144 = 12²)
            state_dir: Where to save/load MTR checkpoint state
            grain_storage_dir: Where to store crystallized grains
            device: torch device (cpu, cuda)
            enable_grain_system: Enable grain crystallization pipeline
            dream_bucket_dir: Path to dream bucket research signal archive
        """
        self.device = device
        self.state_dir = state_dir
        self.grain_storage_dir = grain_storage_dir
        self.enable_grain_system = enable_grain_system and GRAIN_SYSTEM_AVAILABLE
        
        print("Phase 3E Initialization (MTR + Cartridges + Grains + Dream Bucket)")
        print("="*70)
        
        # ====================================================================
        # STEP 0: DREAM BUCKET (Research signal archive)
        # ====================================================================
        print("\n0. Initializing Dream Bucket...")
        from dream_bucket import DreamBucketWriter, DreamBucketReader
        try:
            self.dream_bucket_writer = DreamBucketWriter(dream_bucket_dir)
            self.dream_bucket_reader = DreamBucketReader(dream_bucket_dir)
            print(f"   ✓ Dream bucket initialized at {dream_bucket_dir}")
        except Exception as e:
            print(f"   ⚠ Dream bucket initialization failed: {e}")
            print("   ⚠ Continuing without dream bucket logging")
            self.dream_bucket_writer = None
            self.dream_bucket_reader = None
        
        # ====================================================================
        # STEP 1: CARTRIDGE ENGINE (Fact lookup + learning)
        # ====================================================================
        print("\n1. Loading CartridgeInferenceEngine...")
        self.cartridge_engine = CartridgeInferenceEngine(
            cartridges_dir,
            dream_bucket_writer=self.dream_bucket_writer
        )
        cart_stats = self.cartridge_engine.registry.get_stats()
        print(f"   ✓ {cart_stats['cartridge_count']} cartridges loaded")
        print(f"   ✓ {cart_stats['total_facts']} total facts")
        print(f"   ✓ Learning: Phase 1-4 (graph, anchors, CTR, seasonality)")
        
        # ====================================================================
        # STEP 2: MTR ENGINE (Temporal reasoning)
        # ====================================================================
        print("\n2. Initializing MTR 5.5 Engine...")
        self.mtr_engine = KitbashMTREngine(
            vocab_size=vocab_size,
            d_model=d_model,
            d_state=d_state,
            dream_bucket_writer=self.dream_bucket_writer
        )
        self.mtr_engine = self.mtr_engine.to(device)
        print(f"   ✓ MTR initialized (vocab={vocab_size}, d_model={d_model}, d_state={d_state})")
        print(f"   ✓ Learning: Weights (W1, W2), strength, recency, time")
        
        # ====================================================================
        # STEP 3: STATE MANAGER (Persistence)
        # ====================================================================
        print("\n3. Initializing State Manager...")
        self.state_manager = MTRStateCheckpoint(state_dir)
        self.mtr_state = None
        
        if self.state_manager.exists():
            print("   ✓ Previous state found, loading...")
            self.mtr_state, metadata = self.state_manager.load(device=device)
            print(f"     Session: {metadata.get('session_id')}")
            print(f"     MTR time: {metadata.get('time')} (cumulative queries)")
        else:
            print("   ✓ No previous state, starting fresh")
        
        # ====================================================================
        # STEP 4: GRAIN SYSTEM (Phantom tracking + crystallization)
        # ====================================================================
        self.grain_orchestrator = None
        self.grain_router = None
        self.mtr_grain_pipeline = None
        
        if self.enable_grain_system:
            print("\n4. Initializing Grain System...")
            
            # Phantom tracker + crystallization orchestrator
            self.grain_orchestrator = ShannonGrainOrchestrator(
                cartridge_id="default",
                storage_path=grain_storage_dir,
                cartridge_engine=self.cartridge_engine
            )
            print(f"   ✓ ShannonGrainOrchestrator initialized")
            
            # Grain router (loads crystallized grains from disk)
            self.grain_router = GrainRouter(
                cartridges_dir=cartridges_dir,
                cartridge_engine=self.cartridge_engine,
                dream_bucket_writer=self.dream_bucket_writer
            )
            print(f"   ✓ GrainRouter loaded {self.grain_router.total_grains} grains")
            
            # Unified MTR↔Grain bridge
            # Get first cartridge for crushing (TODO: support multiple cartridges)
            first_cartridge = next(iter(self.cartridge_engine.registry.cartridges.values())) if self.cartridge_engine.registry.cartridges else None
            self.mtr_grain_pipeline = MTRGrainUnifiedPipeline(
                mtr_grain_orchestrator=self.grain_orchestrator,
                grain_router=self.grain_router,
                trigger_interval=51,  # Crystallize at query 51 (after 50 cycles for locking)
                cartridge=first_cartridge  # Pass actual CartridgeLoader, not registry
            )
            print(f"   ✓ MTRGrainUnifiedPipeline initialized")
            print(f"   ✓ Phantom tracking + crystallization enabled")
            print(f"   ✓ Trigger interval: 51 queries (after phantom locking)")
        else:
            print("\n4. Grain System: DISABLED (running MTR-only mode)")
        
        # ====================================================================
        # STEP 5: QUERY TRACKING
        # ====================================================================
        self.query_count = 0
        self.recent_facts = []  # For graph learning
        self.recent_grains = []  # For grain co-occurrence
        
        print("\n" + "="*70)
        print("✓ Phase 3E Ready (Full Pipeline)\n")
    
    def process_query(self, context: QueryContext) -> QueryResult:
        """
        Process a single query through full Phase 3E pipeline.
        
        Flow:
        1. Cartridge search (learns co-occurrence, anchors, CTR, seasonality)
        2. Grain search (O(1) L3 cache lookup if available)
        3. MTR inference with learned state
        4. Phantom tracking (high error = low confidence = learning signal)
        5. Optional: crystallization (every 100 queries)
        
        Args:
            context: QueryContext with text, project, session, hat
        
        Returns:
            QueryResult with response, latencies, and diagnostics
        """
        import time
        start_time = time.perf_counter()
        
        self.query_count += 1
        
        # ====================================================================
        # PHASE 1: CARTRIDGE LOOKUP (Facts)
        # ====================================================================
        cartridge_start = time.perf_counter()
        
        cartridge_request = CartridgeInferenceRequest(context.query_text)
        cartridge_response = self.cartridge_engine.query(
            cartridge_request, 
            limit=3
        )
        
        cartridge_latency_ms = (time.perf_counter() - cartridge_start) * 1000
        
        context_facts = []
        fact_ids = set()
        
        if cartridge_response:
            context_facts = [{
                'text': cartridge_response.answer,
                'confidence': cartridge_response.confidence,
                'source': cartridge_response.cartridge,
                'fact_id': cartridge_response.fact_id,
            }]
            fact_ids.add(cartridge_response.fact_id)
            self.recent_facts.append(cartridge_response.fact_id)
        
        # ====================================================================
        # PHASE 2: GRAIN LOOKUP (Crystallized patterns)
        # ====================================================================
        grain_latency_ms = 0.0
        grain_facts = []
        
        if self.grain_router is not None:
            grain_start = time.perf_counter()
            
            # Extract query concepts
            query_concepts = context.query_text.lower().split()[:5]
            
            # Set hat context before grain search
            if context.hat and GRAIN_SYSTEM_AVAILABLE:
                try:
                    self.grain_router.set_context_hat(context.hat)
                except:
                    pass  # Hat may not be a valid value, skip
            
            # Search grains with L3 cache preference (NEW)
            grain_search_results = self.grain_router.search_grains(
                query_concepts=query_concepts,
                recent_grains=self.recent_grains
            )
            
            grain_latency_ms = (time.perf_counter() - grain_start) * 1000
            
            if grain_search_results:
                # Take top result (grain_id, score)
                top_grain_id, top_score = grain_search_results[0]
                top_grain = self.grain_router.lookup_cached(top_grain_id)
                
                if top_grain:
                    grain_facts = [{
                        'grain_id': top_grain_id,
                        'fact_id': top_grain.get('fact_id'),
                        'confidence': top_grain.get('confidence', 0.0),
                        'from_cache': True,
                        'latency_ms': grain_latency_ms,
                        'score': top_score,
                        'source': 'crystallized_grain',
                    }]
                    
                    # Track for co-occurrence learning
                    self.recent_grains.append(top_grain_id)
                    if top_grain.get('fact_id'):
                        fact_ids.add(top_grain['fact_id'])
        
        # ====================================================================
        # PHASE 3: MTR INFERENCE (Temporal reasoning)
        # ====================================================================
        mtr_start = time.perf_counter()
        
        # Tokenize query
        tokens = self._simple_tokenize(context.query_text)
        token_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        
        # Get kappa from hat context (epistemic rigidity)
        kappa = 1.0
        if context.hat and self.enable_grain_system:
            kappa = HatKappaMapper.get_kappa(context.hat)
        
        # Run MTR inference
        with torch.no_grad():
            logits, error_signal, new_state = self.mtr_engine(
                token_ids,
                state=self.mtr_state,
                kappa=kappa
            )
        
        # Update state for next query
        self.mtr_state = new_state
        
        mtr_latency_ms = (time.perf_counter() - mtr_start) * 1000
        
        # ====================================================================
        # PHASE 4: FEEDBACK LOGGING (Learning)
        # ====================================================================
        
        # Log to cartridge engine (Phase 3 learning)
        if cartridge_response and cartridge_response.fact_id is not None:
            mtr_error = float(error_signal.mean())
            success = mtr_error < 0.25
            context_str = context.project_context or "general"
            
            self.cartridge_engine.log_fact_usage(
                fact_id=cartridge_response.fact_id,
                success=success,
                mtr_error=mtr_error,
                context=context_str
            )
        
        # Log to grain router (Phase 1.5 CTR tracking)
        if self.grain_router and grain_facts:
            grain_id = grain_facts[0]['grain_id']
            mtr_error = float(error_signal.mean())
            success = mtr_error < 0.25
            
            self.grain_router.log_grain_outcome(grain_id, mtr_error)
            self.grain_router.log_grain_usage(grain_id, success, mtr_error, context.project_context or "general")
        
        # ====================================================================
        # PHASE 5: PHANTOM TRACKING & CRYSTALLIZATION (NEW)
        # ====================================================================
        crystallization_report = None
        
        if self.mtr_grain_pipeline is not None:
            # Get epistemic snapshot for phantom tracking
            epistemic_snapshot = self.mtr_engine.get_epistemic_snapshot(
                token_ids, self.mtr_state, kappa
            )
            
            # Advance phantom cycle from PREVIOUS query (so we're always one cycle ahead)
            if self.query_count > 0:  # Don't advance on first query
                self.mtr_grain_pipeline.advance_phantom_cycle()
            
            # Log to phantom tracker + check for crystallization
            pipeline_metadata = self.mtr_grain_pipeline.process_mtr_query(
                fact_ids=fact_ids,
                query_tokens=tokens,
                error_signal=error_signal,
                epistemic_snapshot=epistemic_snapshot,
                hat=context.hat,
                dissonance_result=None  # Optional, can compute if needed
            )
            
            crystallization_report = pipeline_metadata.get('crystallization')
            
            # Activate newly crystallized grains into L3 cache (NEW)
            if crystallization_report and self.grain_router:
                newly_crystallized = crystallization_report.get('crystallized_grains', [])
                if newly_crystallized:
                    grain_ids = [g.get('grain_id') for g in newly_crystallized if g.get('grain_id')]
                    if grain_ids:
                        activation_result = self.grain_router.activate_grains(grain_ids)
                        # Optionally log activation
                        # print(f"  ✓ Activated {activation_result['loaded']} grains")
            
            # Advance phantom cycle after each query (needed for locking)
            self.mtr_grain_pipeline.advance_phantom_cycle()
        
        # ====================================================================
        # PHASE 6: RESPONSE GENERATION
        # ====================================================================
        predictions = torch.argmax(logits, dim=-1)
        response_tokens = predictions[0].cpu().tolist()
        mtr_response = self._decode_tokens(response_tokens)
        
        total_latency_ms = (time.perf_counter() - start_time) * 1000
        
        # ====================================================================
        # RETURN RESULT
        # ====================================================================
        result = QueryResult(
            user_query=context.query_text,
            mtr_response=mtr_response,
            mtr_confidence=float(error_signal.mean()),
            cartridge_facts=context_facts,
            grain_facts=grain_facts if grain_facts else None,
            total_latency_ms=total_latency_ms,
            mtr_latency_ms=mtr_latency_ms,
            cartridge_latency_ms=cartridge_latency_ms,
            grain_latency_ms=grain_latency_ms,
            crystallization_report=crystallization_report,
        )
        
        return result
    
    def _simple_tokenize(self, text: str) -> list:
        """
        Simple tokenization for demo.
        
        Args:
            text: Text to tokenize
        
        Returns:
            List of token IDs (0-49256)
        """
        tokens = []
        words = text.lower().split()
        
        for word in words:
            token_id = hash(word) % 49256
            tokens.append(token_id)
        
        return tokens
    
    def _decode_tokens(self, token_ids: list) -> str:
        """
        Decode token IDs to text (placeholder for MVP).
        
        Args:
            token_ids: List of token IDs
        
        Returns:
            Decoded text
        """
        return f"[MTR Response from {len(token_ids)} tokens]"
    
    def save_state(self, session_id: str = "default", metadata: Optional[Dict] = None) -> None:
        """
        Save all state to disk.
        
        Saves:
        - MTR weights (W1, W2, strength, last_seen, time)
        - CartridgeEngine learning (fact_graph, anchors, CTR, seasonality)
        - GrainRouter learning (grain_graph, grain_ctr)
        - Phantom tracker state (cycles, phantoms)
        
        Args:
            session_id: Session identifier
            metadata: Optional metadata dict
        """
        if self.mtr_state is None:
            print("No state to save")
            return
        
        if metadata is None:
            metadata = {}
        
        metadata['query_count'] = self.query_count
        
        # Save MTR state
        self.state_manager.save(
            self.mtr_state,
            d_model=self.mtr_engine.d_model,
            d_state=self.mtr_engine.d_state,
            session_id=session_id,
            metadata=metadata
        )
        
        print(f"✓ MTR state saved")
        print(f"  Session: {session_id}")
        print(f"  Query count: {self.query_count}")
        print(f"  MTR time: {self.mtr_state['time']}")
        
        # Save grain system state if enabled
        if self.grain_orchestrator:
            self.grain_orchestrator.phantom_tracker.save(
                f"{self.grain_storage_dir}/phantom_tracker.json"
            )
            print(f"✓ Phantom tracker state saved")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get complete orchestrator statistics.
        
        Returns:
            Dict with stats from all components
        """
        stats = {
            'query_count': self.query_count,
            'cartridge_engine': self.cartridge_engine.get_stats(),
            'mtr_state_time': self.mtr_state['time'] if self.mtr_state else 0,
        }
        
        if self.grain_router:
            stats['grain_router'] = {
                'total_grains': self.grain_router.total_grains,
                'grain_graph': self.grain_router.get_graph_density(),
                'ctr_stats': self.grain_router.get_ctr_stats(),
                'activation': self.grain_router.get_activation_stats()['manual_stats'],  # NEW
            }
        
        if self.mtr_grain_pipeline:
            stats['mtr_grain_pipeline'] = self.mtr_grain_pipeline.get_full_stats()
        
        return stats
    
    def print_summary(self) -> None:
        """Print human-readable orchestrator summary."""
        stats = self.get_stats()
        
        print("\n" + "="*70)
        print("PHASE 3E ORCHESTRATOR SUMMARY")
        print("="*70)
        
        print(f"\nQueries Processed: {stats['query_count']}")
        
        if 'cartridge_engine' in stats:
            cart = stats['cartridge_engine']['registry']
            print(f"\nCartridge Engine:")
            print(f"  Cartridges: {cart['cartridge_count']}")
            print(f"  Facts: {cart['total_facts']}")
        
        print(f"\nMTR Engine:")
        print(f"  Cumulative time: {stats['mtr_state_time']} queries")
        
        if 'grain_router' in stats:
            print(f"\nGrain Router:")
            print(f"  Total grains: {stats['grain_router']['total_grains']}")
            print(f"  Graph nodes: {stats['grain_router']['grain_graph']['nodes']}")
            print(f"  Graph edges: {stats['grain_router']['grain_graph']['edges']}")
            
            # NEW: Grain activation stats
            if 'activation' in stats['grain_router']:
                activation = stats['grain_router']['activation']
                print(f"\nGrain Activation (L3 Cache):")
                print(f"  Total activations: {activation['total_activations']}")
                print(f"  Grains activated: {activation['grains_activated']}")
                print(f"  Cache hits: {activation['cache_hits']}")
                print(f"  Cache misses: {activation['cache_misses']}")
                if activation['cache_hits'] + activation['cache_misses'] > 0:
                    hit_rate = activation['cache_hits'] / (activation['cache_hits'] + activation['cache_misses'])
                    print(f"  Hit rate: {hit_rate:.1%}")
        
        if 'mtr_grain_pipeline' in stats:
            pipeline = stats['mtr_grain_pipeline']
            print(f"\nMTR-Grain Pipeline:")
            print(f"  Phantom hits: {pipeline['phantom_bridge']['total_hits_logged']}")
            print(f"  Crystallizations: {pipeline['crystallization_trigger']['crystallization_count']}")
        
        print("="*70 + "\n")


# ============================================================================
# EXAMPLE USAGE & TESTING
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("PHASE 3E INTEGRATION TEST - Full MTR + Cartridges + Grains Pipeline")
    print("="*70 + "\n")
    
    try:
        # Initialize orchestrator with grain system
        orchestrator = Phase3EOrchestrator(
            cartridges_dir="./cartridges",
            grain_storage_dir="./grains",
            device="cpu",
            enable_grain_system=True
        )
        
        # Test queries
        test_queries = [
            QueryContext(
                query_text="What is physics and motion?",
                session_id="test_session",
                project_context="physics"
            ),
            QueryContext(
                query_text="Tell me about biology and life",
                session_id="test_session",
                project_context="biology"
            ),
            QueryContext(
                query_text="Explain chemistry and reactions",
                session_id="test_session",
                project_context="chemistry"
            ),
        ]
        
        print("Processing queries:")
        print("-"*70)
        
        for i, query_ctx in enumerate(test_queries, 1):
            print(f"\n{i}. Query: '{query_ctx.query_text}'")
            
            result = orchestrator.process_query(query_ctx)
            
            print(f"   Cartridge: {len(result.cartridge_facts)} facts")
            if result.cartridge_facts:
                print(f"     - {result.cartridge_facts[0]['source']}")
            
            if result.grain_facts:
                print(f"   Grain: {len(result.grain_facts)} grains")
                print(f"     - {result.grain_facts[0]['source']}")
            
            print(f"   MTR: {result.mtr_response}")
            print(f"   Latency: {result.total_latency_ms:.1f}ms "
                  f"(cart: {result.cartridge_latency_ms:.1f}ms, "
                  f"grain: {result.grain_latency_ms:.1f}ms, "
                  f"mtr: {result.mtr_latency_ms:.1f}ms)")
            
            if result.crystallization_report:
                print(f"   Crystallization: {result.crystallization_report.get('verdict')}")
        
        # Save state
        print("\n" + "-"*70)
        print("Saving state...")
        orchestrator.save_state(session_id="test_session")
        
        # Print summary
        print("\nOrchestrator Summary:")
        orchestrator.print_summary()
        
        print("✓ Phase 3E integration test complete")
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
