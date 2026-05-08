# Kitbash: Local-First Continuous Learning without Catastrophic Forgetting

**Version:** Phase 4 (Complete) → Phase 5 (Ready)  
**Status:** Production-ready core; actively developing next generation  
**Last Updated:** May 2026

Kitbash is a deterministic cognitive architecture for continuous learning on local hardware. It combines symbolic knowledge organization (epistemological stack), procedural memory (sleep pipeline), and temporal reasoning (MTR) into a unified system that grows with hardware capabilities while maintaining zero external dependencies.

## Philosophy

Instead of asking LLMs to orchestrate and remember, Kitbash delegates orchestration to purpose-built deterministic components and uses LLMs as **generation peripherals**. All state management, routing, and knowledge topology lives in symbolic, inspectable systems.

**Core principles:**
- **Local-first:** Zero cloud dependencies. All computation happens on consumer hardware.
- **Deterministic routing:** Explicit logic replaces LLM-based choreography.
- **Hardware-aware:** Architecture anticipates evolving local capabilities (learned neural components replacing friction points).
- **Preservation-first:** Compress rather than delete. Data is treated as an archive.
- **Sleep as first-class:** The sleep pipeline isn't an afterthought—it's the primary learning mechanism.

## Architecture Overview

```
Kitbash (Phase 4)
├── Query Orchestrator (POSIX-compliant, pluggable)
│   ├── Triage Agent (rule-based MVP; BitNet learned agent in Phase 5)
│   ├── Inference Engines
│   │   ├── Grain Engine (crystallized facts, L3 cache)
│   │   ├── Cartridge Engine (full fact search, Phase 1-4 learning stats)
│   │   └── BitNet Engine (learned specialist models)
│   └── Mamba Context Service (temporal context windows)
│
├── Epistemological Stack (L0–L5)
│   ├── L0: Hardwired facts (undefined layer, to be specified)
│   ├── L1: Axioms (≥95% confidence) & observations (70-95%)
│   ├── L2: Working theory (heuristics, learning stats, MTR weights)
│   ├── L3: Context (grain state, co-occurrence edges)
│   ├── L4: Hat/attitude system (inference-licensing, mode markers)
│   └── L5: Scenario state (user model, session continuity)
│
├── Grain System (Knowledge Crystallization)
│   ├── Phantom Tracker (ephemeral hypotheses → locks → grains)
│   ├── Axiom Validator (Sicherman rules: persistence, resistance, independence)
│   ├── Grain Crystallization (harmonic lock → ternary crush → 1.58-bit encoding)
│   ├── Grain Activation Cache (L3, ~261 grains active)
│   └── Spreading Activation (Hopfield-style energy minimization)
│
├── Sleep Pipeline (6 stages, ~0.1s end-to-end on synthetic data)
│   ├── Stage 1: Log consolidation
│   ├── Stage 2: Pattern recognition
│   ├── Stage 3: Hypothesis generation
│   ├── Stage 4: Question generation
│   ├── Stage 5: Observation promotion
│   └── Stage 6: Cleanup & archive
│
├── Temporal Reasoning (MTR v5.5)
│   ├── Monarch-TTT architecture with Ebbinghaus decay
│   ├── CoPENt positional encoding
│   ├── Token-level reasoning, sequence processing
│   └── Integrated with grain confidence tracking
│
└── Redis Blackboard (Shared State Bus)
    ├── Layer state (L0–L5)
    ├── Coupling validation (layer integrity checks)
    ├── Dream bucket logs (anomalies, hypotheses, traces)
    └── Resonance weights (Tier 5 ephemeral patterns)
```

## Phase 4: Complete ✅

**What's finished:**

1. **POSIX-Compliant Architecture**
   - Clean interface contracts (TriageAgent, InferenceEngine, MambaContextService)
   - Dependency injection; adapters bridge Phase 3E components
   - Factory function wires everything; supports multiple configurations
   - Zero breaking changes; 100% backward compatible

2. **Sleep Pipeline (All 6 Stages)**
   - End-to-end consolidation, pattern recognition, hypothesis generation
   - Question formation, observation promotion, archival
   - Dream bucket logs anomalies as navigational data, not noise
   - Preservation-first strategy: compress, don't delete

3. **Epistemological Stack Specification**
   - 6-layer model with mode-aware semantics (fiction vs. non-fiction)
   - Clear query flows (downward for answering, upward for learning)
   - Layer mapping to grain system, sleep pipeline, MTR
   - Canonical spec in EPISTEMOLOGICAL_STACK_SPEC.md

4. **Learned State Preserved**
   - 39 cartridges with 606+ facts
   - 261 crystallized grains in L3 cache
   - MTR weights and resonance patterns intact
   - End-to-end latency: 6–10 ms

## Phase 5: Ready 🚀

**What's planned (design complete, ready to implement):**

### 5A: Mutation Cleanup (High clarity, unblocks Phase 5)
- **Mutation 1:** Separate L1 axioms from L2 heuristics (GrainMetadata → axiom/observation distinction)
- **Mutation 2:** Audit L2 working theory (expose MTR, phantoms, learning stats via L2WorkingTheoryService)
- **Mutation 3:** Enforce L3/L4 separation (context/hat independence in MTR tuning)
- **Mutation 4:** Clarify L5 persistence (session-to-session state boundaries)
- **Mutation 5:** Dream bucket recalibration (confidence updates from anomalies)
- **Completion:** ~3 weeks; all mutations fixed before Phase 5 begins

### 5B: Core Features
- **BitNetTriageAgent:** Learned routing agent using active resonance patterns
- **RealMambaService:** Temporal context windows (1h, 1d, 72h, 1w from conversation history)
- **MetabolismScheduler:** Background work scheduling (sleep cycles, crystallization triggers)
- **Coupling Validation:** Enforce epistemological layer integrity

### 5C: Specialist Models
- Query intent classification
- Confidence prediction
- MTR kappa estimation
- All as ternary BitNet models (straight-through estimator quantization)

## Key Features

### Knowledge Organization
- **Cartridges:** Domain-specific fact stores with co-occurrence graphs
- **Grains:** Crystallized, cached facts with confidence tracking
- **Phantoms:** Ephemeral hypotheses tracked until harmonic lock
- **Axioms:** High-confidence (≥95%) persistent facts
- **Observations:** Lower-confidence (70–95%) facts, mutable by dream bucket

### Memory & Learning
- **Declarative memory:** Facts in epistemological stack (L1–L3)
- **Procedural memory:** Query chains and navigational links (Stages 1.5, 2.5 incoming)
- **Episodic memory:** Dreams logged to bucket (false positives, collisions, violations, hypotheses, anomalies, traces)
- **Sleep consolidation:** Multi-stage processing treats anomalies as navigational signals

### Temporal Reasoning
- **MTR v5.5:** Token-level reasoning with Ebbinghaus decay
- **Resonance weights:** Ephemeral pattern tracking (Tier 5)
- **Layer salience:** Dynamic weighting based on inference context
- **Positional encoding:** CoPENt for sequence understanding

### Integration Points
- **Redis blackboard:** Shared state bus for layer coordination
- **Dream bucket:** 6 event categories, all preserved for post-hoc analysis
- **Epistemic settler (EDGE):** L0–L5 integration before LLM prompting
- **CartridgeLoader:** Google search algorithm re-implementations (fact co-occurrence, CTR + freshness, temporal seasonality)

## Installation & Setup

### Requirements
- Python 3.9+
- Redis (local or network-accessible)
- SQLite (bundled)
- ~500 MB free disk (for learned state, cartridges, grains)

### Quick Start
```bash
# Clone or copy project files
cd /path/to/kitbash

# Initialize Redis (if not running)
redis-server &

# Create basic orchestrator
python3 -c "
from query_orchestrator_factory import create_query_orchestrator
orch = create_query_orchestrator()
result = orch.process_query('What is ATP?')
print(result)
"
```

### Configuration
- **Redis:** Set `REDIS_HOST` / `REDIS_PORT` (default: localhost:6379)
- **Cartridge path:** Set `CARTRIDGE_DIR` (default: ./cartridges/)
- **Grain cache size:** Edit `GrainActivationCache.max_size` (default: 261)
- **Sleep cycle timing:** Edit `SleepOrchestrator.sleep_interval` (default: 300s)

## File Structure

### Core Components
- `query_orchestrator.py` — Original Phase 3E (still works)
- `query_orchestrator_posix.py` — POSIX version (dependency injection)
- `query_orchestrator_factory.py` — Factory function (recommended entry point)

### Knowledge Systems
- `cartridge_loader.py` — Fact lookup, Phase 1-4 learning stats
- `grain_system.py` — Phantom tracking, crystallization, confidence
- `grain_router.py` — L3 cache, spreading activation
- `axiom_validator.py` — Sicherman rules (persistence, resistance, independence)

### Temporal & Resonance
- `MTR_v5_5_NN.py` — Temporal reasoning engine (Monarch-TTT + Ebbinghaus decay)
- `resonance_weights.py` — Tier 5 ephemeral pattern tracking
- `mtr_grain_bridge.py` — MTR ↔ grain integration

### Sleep & Consolidation
- `sleep_orchestrator.py` — Sleep pipeline coordinator
- `sleep_consolidator.py` — Stage 1 (log consolidation)
- `sleep_pattern_recognition.py` — Stage 2 (pattern mining)
- `sleep_hypothesis_generation.py` — Stage 3 (hypothesis formation)
- `sleep_question_generation.py` — Stage 4 (question formation)
- `sleep_observation_promotion.py` — Stage 5 (observation to L1)
- `sleep_cleanup_archive.py` — Stage 6 (cleanup & preservation)
- `dream_bucket.py` — Event logging (anomalies as navigational data)

### Interfaces & Adapters (Phase 4)
- `query_orchestrator_posix.py` — QueryOrchestrator with dependency injection
- `query_orchestrator_factory.py` — Wires all components
- `grain_engine.py` — InferenceEngine adapter for GrainRouter
- `cartridge_engine.py` — InferenceEngine adapter for CartridgeLoader
- `bitnet_engine.py` — InferenceEngine adapter for BitNet HTTP server
- `rule_based_triage.py` — MVP TriageAgent
- `mock_mamba_service.py` — MVP MambaContextService

### Redis & State Management
- `redis_blackboard.py` — Shared state bus (L0–L5)
- `redis_coupling.py` — Layer coupling validation
- `heartbeat_service.py` — Background work control (pause/resume)

### Support
- `structured_logger.py` — JSONL logging with rotation
- `kitbash_cartridge.py` — Cartridge data structure
- `kitbash_registry.py` — Cartridge registry
- `batch_cartridge_builder.py` — Bulk cartridge creation

### Specifications & Designs
- `EPISTEMOLOGICAL_STACK_SPEC.md` — 6-layer model (definitive)
- `EPISTEMOLOGICAL_STACK_MUTATIONS.md` — Problem diagnosis (Phase 5A)
- `THEORETICAL_SPATIAL_STACK_SPEC.md` — Long-term vision
- `SLEEP_METABOLISM_SPEC.md` — Sleep pipeline design
- `EDGE_EPISTEMICINTEGRATION_DESIGN.md` — Settler integration
- `DREAM_BUCKET_DESIGN.md` — Logging philosophy
- `KITBASH_ROADMAP_MAY_2026.md` — Complete roadmap (Phase 5A/5/5+)
- `MAGPIE_PILE.md` — External research synthesis

## Quick Reference

### Basic Query
```python
from query_orchestrator_factory import create_query_orchestrator

orch = create_query_orchestrator()
result = orch.process_query("What is cellular respiration?")
print(result['answer'])
print(f"Layer: {result['layer_hit']}")
print(f"Confidence: {result['confidence']}")
```

### Check Working Theory
```python
working_theory = orch.get_working_theory_snapshot(brief=True)
print(f"Active phantoms: {len(working_theory.active_phantoms)}")
print(f"Hot edges: {working_theory.co_occurrence_edges}")
```

### Run Sleep Cycle
```python
from sleep_orchestrator import SleepOrchestrator

sleep = SleepOrchestrator(orch.cartridge_loader, orch.grain_system)
sleep.run_full_cycle()  # All 6 stages
```

### Access Learned State
```python
# Cartridges (domains)
domains = orch.cartridge_loader.list_domains()
facts = orch.cartridge_loader.search(query="ATP", domain="biology")

# Grains (crystallized)
grains = orch.grain_router.get_active_grains()
grain = orch.grain_router.lookup_grain("grain_xyz")

# MTR state
epistemic_snapshot = orch.mtr_engine.get_epistemic_snapshot()

# Resonance patterns
active = orch.resonance.get_active_patterns()
```

## Research Influences

Kitbash converges on patterns identified independently in:
- **Complementary Learning Systems (Derosiaux):** Sleep pipeline mirrors hippocampus-neocortex consolidation
- **Continuous Thought Machine (Wei et al.):** Token-level reasoning embedded in MTR
- **Differentiable Neural Computers (Graves et al.):** Memory + attention patterns
- **Karpathy's LLM knowledge base:** Deterministic indexing + curation before inference
- **Google search algorithms:** Fact co-occurrence graphs, CTR + freshness, temporal seasonality
- **Lakoff & Johnson image schemas:** Epistemological topology constrained by embodied cognition
- **Resonance Web (Brans):** Polarity-weighted crystallization, death shockwaves

See `MAGPIE_PILE.md` for detailed synthesis.

## Roadmap (2026+)

### Immediate (Next 3-4 weeks)
- [ ] Complete Phase 5A mutation cleanup
- [ ] BitNetTriageAgent integration
- [ ] RealMambaService temporal windows
- [ ] MetabolismScheduler background work

### Medium Term
- [ ] Procedural memory (Stages 1.5, 2.5): query chains, navigational links
- [ ] TVTropes embedding space (1000-d, narrative pattern search)
- [ ] Specialist NN models (query classifier, confidence predictor, MTR kappa)
- [ ] Coupling validation enforcement

### Long Term
- [ ] Spatial topology framework (density bands, fractal navigation)
- [ ] Learned neural components replacing symbolic friction points
- [ ] Autoresearch (Kitbash improving itself)
- [ ] Modal semantics unification (fiction + non-fiction same substrate)

## Testing & Validation

### Existing Tests
- `test_sleep_consolidator.py` — Sleep pipeline validation
- `generate_synthetic_dream_bucket.py` — Synthetic data generation

### Running Tests
```bash
python3 test_sleep_consolidator.py
python3 generate_synthetic_dream_bucket.py  # Creates TEST-dream_bucket.jsonl
```

### Performance Benchmarks
- End-to-end query: 6–10 ms
- Cartridge lookup: 15–50 ms
- Grain lookup: <1 ms (L3 cache)
- MTR inference: 5–20 ms
- Sleep cycle: ~100 ms (synthetic)
- Pattern reinforcement: <1 ms

## Design Philosophy Deep Dives

### Why Sleep Consolidation?
The sleep pipeline isn't a post-processing step—it's where learning happens. Awake query-answering generates raw logs (queries, false positives, collisions). Sleep transforms these into:
1. Patterns (Stage 2): "ATP appeared with energy and ATP synthase"
2. Hypotheses (Stage 3): "ATP might be core to energy"
3. Questions (Stage 4): "How does ATP relate to metabolism?"
4. Promotions (Stage 5): Validate and move observations → axioms

This mirrors neuroscience: consolidation during sleep is when the hippocampus replays experiences to the neocortex, strengthening synaptic weights. Kitbash's sleep pipeline does the same with symbolic facts.

### Why Deterministic Routing?
LLMs excel at generation but are expensive and unreliable for orchestration. Asking "which knowledge engine should I query?" is a task for a small, fast, deterministic decision tree, not a 70B parameter model. Once the decision is made, the LLM can generate fluent text. This inversion—**determinism first, fluency second**—is central to Kitbash's cost profile and reliability.

### Why Preservation-First?
In traditional systems, "cleaning up" data means deletion. Kitbash compresses instead. A dream bucket entry marked "false positive" isn't garbage—it's a navigation signal. It says "this path was a dead end." Over time, recurring false positives become observations (in L1), which guide future routing. Data preservation is epistemologically useful.

### Why Anomalies Matter?
The dream bucket logs:
- **False positives:** Queries that returned wrong answers
- **Collisions:** Fact conflicts
- **Violations:** Axiom violations
- **Hypotheses:** Sleep-generated candidate facts
- **Anomalies:** Unexpected patterns
- **Traces:** Query chains

None of these are noise. They're all navigational—they reveal the topology of the knowledge space. A spike in false positives in domain X might indicate a systematic gap in X's cartridge. The sleep pipeline surfaces these to guide learning.

## Contributing

This is a single-developer project. Contributions should align with:
- **Local-first:** No external dependencies
- **Deterministic:** Logic is explicit, not delegated to LLMs
- **Preservation-first:** Data is archival
- **Testable:** Changes should include tests
- **Documented:** Specs come before code

See `KITBASH_ROADMAP_MAY_2026.md` for priority ordering.

## Known Limitations & Gaps

### Phase 5A: Mutations to Fix
- L0 definition unclear (what counts as hardwired facts?)
- L1/L2 boundary fuzzy in code (axioms mixed with heuristics)
- L5 persistence undefined (session boundaries)
- Fiction mode not yet specified
- Dream bucket doesn't yet recalibrate grain confidence

### Phase 5: Features Not Yet Implemented
- BitNetTriageAgent (design ready)
- RealMambaService (design ready)
- MetabolismScheduler (design ready)
- Procedural memory (Stages 1.5, 2.5)

### Research Frontiers
- Modal semantics: Can fiction and non-fiction use the same epistemological stack?
- Causal reasoning: How do you extract causal chains from correlations?
- Specialist NNs: Which operations decompose cleanly into learned components?

## Performance & Hardware

### Tested On
- CPU: Intel i7 (12-core, 3.6 GHz)
- Memory: 32 GB
- Disk: SSD, 1 TB
- Redis: Local, single-threaded

### Scalability Notes
- Cartridges scale linearly (lookup: O(log n) with indexing)
- Grains scale with L3 cache size (current: 261; adjustable)
- Sleep pipeline stages parallelize (dependency analysis pending)
- MTR state grows with query history (currently ~100k tokens preserved)

### Future Hardware Targets
- **Near term (2 years):** Local LLM inference (3-7B models with LoRA)
- **Medium term (3-5 years):** Learned neural components replacing symbolic layers
- **Long term (5+ years):** Full neuro-symbolic hybrid with hardware acceleration

## License

TBD (currently undecided; contact author)

## Acknowledgments

Kitbash emerged from independent study of sleep research, embodied cognition (Lakoff & Johnson), and neuroscience, combined with reverse-engineering of published AI systems (Karpathy, Graves, Wei, Derosiaux, and others). The architecture reflects a layman's interpretation, which is sometimes its strength (unconstrained by convention) and sometimes its weakness (missing established terminology).

Special thanks to the research community for publishing work that inspired the convergences documented in `MAGPIE_PILE.md`.

## Questions?

For architecture questions, see `EPISTEMOLOGICAL_STACK_SPEC.md` and `KITBASH_ROADMAP_MAY_2026.md`. For implementation questions, read the component files directly—the code is documented. For research questions, start with `MAGPIE_PILE.md` and follow citations outward.

---

**Status:** Phase 4 complete. Phase 5 ready to begin.  
**Next major milestone:** Phase 5A mutation cleanup + BitNetTriageAgent integration  
**Estimated timeline:** 4–6 weeks to Phase 5 operational
