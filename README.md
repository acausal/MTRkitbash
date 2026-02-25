# KitbashAI

A local-first cognitive architecture for continuous learning without catastrophic forgetting.

**Status**: Early-stage research project, work-in-progress. Not production-ready. Validation in progress.

## What Is This?

Kitbash is an experimental system combining:

- **MTR-Ebbinghaus Engine** (Phase 5.5): A temporal reasoning layer implementing Monarch matrix structures with Ebbinghaus-inspired memory decay. Learns weight matrices (W1, W2), memory strength, and recency tracking across sequential queries.

- **CartridgeLoader** (Phase 5.5 Extensions): Fact retrieval with four proven-but-abandoned Google search techniques: fact co-occurrence graphs, query anchor profiles, CTR+freshness weighting, and temporal seasonality detection.

- **GrainSystem** (Phase 2C): A crystallization pipeline that identifies procedural memories and persistent query patterns (phantoms), validates them against axiom rules, compresses them to ternary representations, and persists them as fast-lookup grains for reflex-layer responses.

- **GrainRouter** (Phase 2C): Sub-millisecond grain lookup via L3 cache, with its own co-occurrence learning and CTR tracking parallel to the cartridge engine.

- **Phase 3E Integration**: All three engines orchestrated through a unified query pipeline with state persistence across sessions.

The core innovation is treating **anomalies and false positives as learning signals** rather than errors to be filtered. A background sleep process analyzes these signals to discover structural patterns in how knowledge is organized.

## Current State

**What works:**
- Full 6-phase query pipeline (cartridge search → grain lookup → MTR inference → phantom tracking → crystallization trigger → response)
- MTR state persistence with checkpoint/restore
- Cartridge engine with Phase 1-4 learning integrated
- Grain system phantom tracking and crystallization orchestration
- Integration bridge connecting all components

**What still needs validation:**
- End-to-end integration testing (does state persist correctly across sessions?)
- Performance benchmarking (are latencies acceptable?)
- Crystallization quality (do new grains actually improve future queries?)
- Dream bucket signal capture (are false positives being logged?)
- Sleep orchestrator (can we actually learn from the anomalies?)

**What's intentionally incomplete:**
- Multi-session management (single session only)
- Distributed state (all in-memory)
- Production error handling
- Comprehensive test coverage
- Documentation beyond code comments

## Immediate Roadmap (2-3 weeks)

### Phase 1: Validation (Days 1-3)
- Fix grain_system.py (fact_id field mismatch, one-line change)
- Run integration test suite against live pipeline
- Benchmark latencies and memory usage
- Validate state save/load round-trip

### Phase 2: Signal Capture (Days 3-5)
- Implement dream_bucket.py (archive false positives, collisions, anomalies)
- Hook into grain_router and phantom_bridge for logging
- Verify signals are being captured in the right format

### Phase 3: Analysis Loop (Days 5-10)
- Implement sleep_orchestrator.py (analyze dream bucket logs)
- Build basic recommendation engine
- Test end-to-end: query → anomaly → sleep analysis → recommendation

### Future
- Multi-session support (different projects/characters isolated)
- Episodic memory with narrative indexing (TVTropes framework)
- Gradient-based grain refinement (use MTR error to improve grains)
- Production hardening (error recovery, observability, configuration)

## Problems We're Hoping To Solve

**Catastrophic Forgetting**: Traditional ML systems degrade when trained on new data. We're testing whether a local-first architecture with explicit memory decay and periodic crystallization can learn continuously without regression.

**Knowledge Organization Blindness**: Most systems treat knowledge as flat. We're experimenting with using pattern collision and false positive clustering to discover the actual topology of how concepts relate.

**Speed vs. Scale Tradeoff**: Cartridge lookups are expensive (50ms), grain lookups are fast (<1ms), but grain quality is unknown. We're exploring whether we can bootstrap a fast grain layer from cartridge misses.

**Interpretability**: Every component explains its existence. We know why each fact is ranked, why each grain was crystallized, why each phantom was locked. No black boxes.

**Local-First Design**: Zero cloud dependencies, complete auditability, full control of learning signals. Everything runs on the local machine.

## Why "Kitbash"?

In filmmaking, kitbash refers to combining found objects and miniatures to create something new. This system combines Phase 5.5 temporal reasoning, abandoned-but-proven search algorithms, Phase 2C crystallization machinery, and a local-first learning architecture into something experimental. We're assembling working pieces to see what emerges.

## Technical Notes

- **Language**: Python 3.9+
- **Dependencies**: PyTorch, standard library (no ML frameworks beyond PyTorch)
- **Architecture**: Modular, dependency-free components (grain_system.py is self-contained, mtr_grain_bridge.py is a pure integration layer)
- **State Size**: MTR weights ~10MB, cartridge learning artifacts ~5MB, grain registry ~variable, phantom tracker ~1MB per 1000 queries
- **Latency Targets**: Cartridge <50ms, grain <1ms, MTR 5-20ms, crystallization <500ms

## What's Here

- `MTR_Phase55_Engine.py` — Temporal reasoning with Ebbinghaus decay
- `cartridge_loader.py` — Fact retrieval with Phase 1-4 learning
- `grain_system.py` — Consolidated grain crystallization pipeline (1350 lines, 5 files merged)
- `grain_router.py` — Fast grain lookup with learning
- `mtr_grain_bridge.py` — Integration layer (phantom tracking, hat→kappa mapping, crystallization triggers)
- `phase3e_orchestrator.py` — Main query orchestration (550 lines, fully integrated)
- `mtr_state_manager.py` — Checkpoint/restore for learning artifacts

## Running It

```python
from phase3e_orchestrator import Phase3EOrchestrator, QueryContext

orchestrator = Phase3EOrchestrator(
    cartridges_dir="./cartridges",
    grain_storage_dir="./grains",
    device="cpu",
    enable_grain_system=True
)

result = orchestrator.process_query(
    QueryContext(
        query_text="What is photosynthesis?",
        session_id="my_session",
        project_context="biology"
    )
)

print(result.mtr_response)
orchestrator.save_state()
```

**Current state**: Test harness exists, but integration tests not yet run against actual cartridges. Expect issues.

## Caveats

- This is **not a finished product**. It's a research exploration of whether an architecture based on explicit memory decay, collision analysis, and periodic crystallization can achieve continuous learning without forgetting.
- **Validation is incomplete**. We don't yet know if phantoms crystallize into useful grains, if sleep analysis actually recommends good improvements, or if the system's learning capacity is meaningful at production scale.
- **Known gaps**: Multi-session isolation, distributed state, comprehensive error handling, unit test coverage, performance under load.
- **Abandoned techniques matter**: This explicitly builds on search algorithms dropped by Google (co-occurrence graphs, query anchors, CTR+freshness, seasonality) because they worked but became obsolete under adversarial optimization. We're testing whether they work better in a non-adversarial, local-first context.

## Contributing

This is a solo research project at the moment. Issues and PRs welcome, but expect the codebase to shift significantly as validation progresses.

## Philosophy

- **Spec first, code second**: Features are designed in markdown before implementation
- **Incremental validation**: Each phase is tested before the next is built
- **100% backward compatibility**: Learning continues from previous sessions without reset
- **Local-first always**: No cloud calls, no external APIs (except at boundaries user explicitly chooses)
- **Noise as signal at different scales**: Anomalies, collisions, and false positives are archived as research data for the sleep process

## What Success Looks Like

- Query N+100 outperforms query N because grains crystallized from queries 1-100
- Sleep analysis identifies which grains are collision-prone and recommends axiom refinements
- System continues learning across 10,000+ queries without degradation
- New queries are answered 10x faster (via grain cache) while maintaining coherence

## What Failure Looks Like

- Grains don't actually improve query quality (crystallization is wasted compute)
- Sleep analysis produces noisy recommendations that don't help
- Learning caps out after a few thousand queries (plateau effect)
- System requires manual tuning to prevent drift

---

**Last Updated**: February 25, 2026  
**Current Focus**: Validation phase (integration tests, benchmarking)
