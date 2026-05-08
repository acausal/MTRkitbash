# Architecture: Kitbash Query Flow & Component Design

## Overview

Kitbash processes user queries through a **POSIX-based orchestrator** that coordinates deterministic routing (triage), parallel inference engines, and background metabolism. All state is managed through interfaces and a Redis blackboard, enabling clean separation of concerns and testability.

**Current Phase:** Phase 4 Refactoring (query_orchestrator_posix.py and POSIX interfaces)  
**Ready for:** Phase 5 deployment (BitNet + RealMambaService) **after** three pre-Phase 5 refactorings  
**Future Phases:** Phase 5 adds BitNet and specialist models; Phase 5+ adds TVTropes embedding space and neuro-symbolic consolidation.

---

## ⚠️ Pre-Phase 5 Refactoring Required

The architecture below describes **current state** but three refactorings must complete before Phase 5 development can begin. See `PHASE_5_READINESS_ANALYSIS.md` for detailed specs.

### Blocker 1: L1/L2 Separation (Mutation 1)
- **Issue:** Current `GrainMetadata` doesn't distinguish axioms (L1, ≥95% confidence, immutable) from observations (L2, 70-95%, mutable)
- **Impact:** Phase 5 agents can't tell which grains are safe to reuse vs. which can be recalibrated
- **Fix:** Add `grain_type: Literal["axiom", "observation"]` field to GrainMetadata (~4 hours)
- **Status:** Not yet implemented

### Blocker 2: L2 Auditability (Mutation 2)
- **Issue:** MTR weights, phantom tracker, and co-occurrence graph are opaque; no way to inspect active heuristics
- **Impact:** Phase 5 debugging will be blind; can't see why agent made routing decisions
- **Fix:** Create L2WorkingTheoryService to expose locked phantoms, hot edges, MTR snapshots (~3 hours)
- **Status:** Not yet implemented

### Blocker 3: Redis Bus Integration (Critical Path)
- **Issue:** `redis_blackboard.py` exists but orchestrator doesn't write query state to it; coupling validation is drafted but not wired
- **Impact:** Phase 5 requires bus for inter-component coordination; sleep pipeline can't build procedural edges without query chains on bus
- **Fix:** Wire orchestrator to write query state; sleep pipeline to read query chains; enable coupling validation (~2-3 hours)
- **Status:** redis_blackboard.py exists but is not actively used

**These three blockers must complete before Phase 5 work begins** (estimated 9-10 hours total). Other mutations (3, 4, 5) are lower priority and can be deferred.

---

---

## Query Flow (8-Step Pipeline)

```
User Query
    ↓
[PHASE 1: Metabolism Check]
    ├─ If background work is due (sleep cycle), run it
    └─ Advance turn counter
    ↓
[PHASE 2: Context Retrieval]
    ├─ MambaContextService (Phase 4 placeholder; currently no-op)
    └─ Returns temporal window context
    ↓
[PHASE 3: Triage Decision]
    ├─ RuleBasedTriageAgent analyzes query
    ├─ Routes to layer sequence: [GRAIN, CARTRIDGE, ...]
    └─ Sets confidence thresholds per layer
    ↓
[PHASE 4: Pause Background Work]
    └─ Heartbeat.pause() stops sleep pipeline during query
    ↓
[PHASE 5: Engine Cascade (Complexity Sieve)]
    ├─ For each layer in sequence:
    │  ├─ Attempt inference (layer → engine)
    │  ├─ Check confidence vs. threshold
    │  ├─ If passed: use response, exit cascade
    │  └─ If failed: try next layer
    └─ Result: winning_response or None
    ↓
[PHASE 6: Finalize Response]
    ├─ If answer: record resonance pattern, tag phantom hits
    ├─ Compile metrics (latency, confidence, sources)
    └─ Log to dream bucket for sleep pipeline
    ↓
[PHASE 7-8: Resume & Advance Clock]
    ├─ Heartbeat.resume() restarts background work
    ├─ Advance turn counter (global clock)
    └─ Sync turn across stateful services (triage, resonance)
    ↓
Query Result
```

---

## Component Interactions

### 1. **QueryOrchestrator** (Coordinator)
**File:** `query_orchestrator_posix.py`

The main user-facing entry point. Implements the 8-phase pipeline above.

**Responsibilities:**
- Coordinate triage → cascade → result flow
- Pause/resume heartbeat around query execution
- Record metrics and resonance patterns
- Sync turn counter across services
- Handle phase 3B.3 coupling validation (L0-L5 constraint checking)

**Key methods:**
```python
orchestrator.process_query(user_query: str) -> QueryResult
```

---

### 2. **TriageAgent** (Router)
**Interface:** `interfaces/triage_agent.py`  
**Implementation:** `rule_based_triage.py`

Decides which inference layers to attempt and in what order.

**Rules (Phase 3B MVP):**
- **Very short queries** (<3 words): GRAIN → CARTRIDGE
- **Short queries** (3–10 words): GRAIN → CARTRIDGE → ESCALATE
- **Long queries** (>10 words): CARTRIDGE → ESCALATE
- **Explicit fact reference** ("fact 42"): GRAIN only
- **Default:** Multi-layer sieve

**Output:** `TriageDecision`
```python
{
    "layer_sequence": ["GRAIN", "CARTRIDGE"],
    "confidence_thresholds": {"GRAIN": 0.90, "CARTRIDGE": 0.70},
    "reasoning": "Short query with explicit lookup pattern"
}
```

---

### 3. **InferenceEngines** (Workers)
**Interface:** `interfaces/inference_engine.py`

Engines implement deterministic inference. Phase 3B MVP has two:

#### **GrainEngine** (L3 cache hit)
**File:** `grain_engine.py`

Queries the grain activation cache (L3, epistemological stack).
- **Input:** User query, context
- **Process:** Embedding lookup, resonance-weighted retrieval, topological spreading activation
- **Output:** Confidence score, source grains, answer text

**Threshold:** 0.90 (high confidence required)  
**Latency:** ~2–4ms (in-memory)

---

#### **CartridgeEngine** (Full search)
**File:** `cartridge_engine.py`

Scans loaded cartridges for matching facts using co-occurrence graphs.
- **Input:** User query, mamba context
- **Process:** Phase 3E CartridgeInferenceEngine adaptor; Google search algorithms (CTR, co-occurrence, freshness)
- **Output:** Confidence score, source facts, answer text

**Threshold:** 0.70 (lower bar, broader search)  
**Latency:** ~8–15ms (disk I/O + graph traversal)

---

#### **BitNetEngine** (Phase 4 stub)
**File:** `bitnet_engine.py`

Queues learned inference via external server. Currently disabled in Phase 3B MVP.

---

### 4. **MambaContextService** (Temporal Context)
**Interface:** `interfaces/mamba_context_service.py`  
**Implementation:** `mock_mamba_service.py` (Phase 4 placeholder)

Retrieves temporal context windows (recent facts, recent queries). Will implement sliding window retrieval in Phase 4.

---

### 5. **ResonanceWeightService** (L5 pattern memory)
**File:** `resonance_weights.py`

Tracks query patterns (embeddings + metadata) and reinforces successful ones.

**Responsibilities:**
- Record new query patterns after successful inference
- Reinforce existing patterns on repeat queries
- Decay low-weight patterns (cleanup)
- Provide weights to triage agent for ranking

---

### 6. **HeartbeatService** (Clock & Pause)
**File:** `heartbeat_service.py`

Global clock coordinating turn counter and pause/resume signals. ✅ Implemented and working.

**Responsibilities:**
- Advance turn counter per query
- Pause background work (sleep pipeline) during query execution
- Resume background work after response is ready
- Sync turn with triage agent and resonance service

**Status:** Fully functional. MetabolismScheduler (Phase 5) will extend this to coordinate background work scheduling.

---

### 6.5 **MetabolismScheduler** (Background Work Coordinator)
**File:** Not yet implemented (Phase 5 work)

Will coordinate background work (sleep pipeline, crystallization) with heartbeat pause/resume. Currently a placeholder; Phase 5 will implement this to manage resource allocation across sleep cycles.

**When complete will:**
- Schedule sleep pipeline execution between queries
- Coordinate pause/resume signals with heartbeat
- Manage resource budgets (compute time, memory) per sleep cycle

---

### 7. **RedisBlackboard** (Shared State)
**File:** `redis_blackboard.py`

Redis-backed message bus. **⚠️ Status: Drafted but not integrated into query orchestrator yet (Blocker 3 in pre-Phase 5 refactoring).**

Will support:
- Query state storage (phases, engines, latency)
- Dream bucket event logging
- Query chain reconstruction (for sleep pipeline procedural edge extraction)
- Health and metrics tracking

**Intended Keyspace:**
```
kitbash:grains:<fact_id>        → JSON grain data
kitbash:queries:queue           → Pending query IDs
kitbash:queries:state:<query_id> → Query state (phase, engine, latency)
kitbash:diagnostic:feed         → Event log (triage, cascades, errors)
kitbash:health:<worker>         → Worker heartbeat
kitbash:metrics:<name>          → Performance metrics (sorted sets)
```

**Integration work (Phase 3B.3 / pre-Phase 5):**
- Wire QueryOrchestrator to write query state to bus after each query
- Wire SleepOrchestrator to read query chains from bus instead of isolated dream events
- Enable `redis_coupling.py` validation (currently drafted but not actively used)

**Current status:** redis_blackboard.py exists and has layer read/write API; orchestrator doesn't call it yet.

---

### 8. **SleepOrchestrator** (Background Consolidation)
**File:** `sleep_orchestrator.py`

Runs offline consolidation (background grain crystallization, pattern analysis, cleanup). Paused during query execution (heartbeat.pause).

**Stages (6-stage pipeline, all ✅ implemented):**
1. **Log consolidation** – Move recent query logs to dream bucket
2. **Pattern recognition** – Detect phantom associations, grain collisions
3. **Hypothesis generation** – Generate refinement questions
4. **Question generation** – Prioritize learning opportunities
5. **Observation promotion** – Elevate high-confidence patterns to axioms (L1)
6. **Cleanup & archive** – Decay old patterns, compress cold data

**Planned additions (Phase 5+):**
- **Stage 1.5 (Trace consolidation):** Extract intra-cartridge sequential relationships
- **Stage 2.5 (Procedural edge extraction):** Extract inter-cartridge navigational links and query chains
- These require Blocker 3 (Redis bus) to be integrated first (so sleep pipeline can read query chains from bus)

---

## Layer Mapping (Epistemological Stack → Components)

| Layer | Name | Storage | Components | Purpose | Status |
|-------|------|---------|------------|---------|--------|
| **L0** | Hardwired facts | Code + RAM | GrainRegistry, grain cache | Bootstrap facts (sleep structure, memory consolidation rules) | ✅ Exists |
| **L1** | Axioms / Crystallized | SQLite + Redis | AxiomValidator, grain crystallizer | Persistent observations (facts passed multiple validation gates) | ⚠️ Exists but not distinguished from L2 yet |
| **L2** | Heuristics (mutable) | Redis | Epistemological settler (L2 weighting) | Query-response rules (if user asks about X, try Y engine) | ⚠️ Embedded in CartridgeLoader; no separate API (Blocker 2) |
| **L3** | Context (transient) | GrainRegistry cache | GrainEngine, spreading activation | Recent facts + topological neighbors (active during consolidation) | ✅ Works |
| **L4** | Hat/attitude (inference-licensing) | TorchState (MTR checkpoints) | MTRGrainBridge, HatKappamapper | Stance system (e.g., "analyze strictly biologically") | ✅ Works |
| **L5** | Resonance patterns (episodic) | Redis + cartridges | ResonanceWeightService, CartridgeEngine | Query success patterns; probabilistic inference licensing | ⚠️ Service exists but not explicitly exposed as L5 scenario state |

**Key Limitations (being addressed in pre-Phase 5 refactoring):**
- **L1/L2 not distinguished:** GrainMetadata lacks `grain_type` field (axiom vs. observation). All grains treated equally in crystallization. [Blocker 1]
- **L2 opaque:** Learning stats (co-occurrence, CTR, seasonality) are embedded in CartridgeLoader with no inspection API. [Blocker 2]
- **L5 not explicit:** Resonance weights exist but aren't formally part of scenario state abstraction. [Mutation 4, lower priority]

---

## Latency Profile

Typical query latencies (Phase 3B MVP):

| Phase | Component | Latency | Bottleneck |
|-------|-----------|---------|------------|
| Metabolism | MetabolismScheduler | 0–10ms | Conditional; often skipped |
| Context | MambaContextService | <1ms | No-op in Phase 3B |
| Triage | RuleBasedTriageAgent | 1–3ms | Hardcoded rules; O(1) |
| Pause | Heartbeat.pause | <1ms | Lock acquisition |
| Engine | **GRAIN** | 2–4ms | Memory lookup + spreading |
| Engine | **CARTRIDGE** | 8–15ms | Disk I/O + graph traversal |
| Finalize | Resonance + logging | 2–5ms | Redis writes |
| Resume | Heartbeat.resume | <1ms | Lock release |
| **Total** | **P50 (GRAIN hit)** | **~8–12ms** | Triage dominates |
| **Total** | **P95 (CARTRIDGE)** | **~20–30ms** | Engine dominates |

**Key insight:** Triage-only latency is ~2–4ms. Engine choice dominates total time. Phase 4 specialist models will add 10–50ms but offload expensive LLM inference.

---

## Factory Pattern (Dependencies)

**File:** `query_orchestrator_factory.py`

The `create_query_orchestrator()` function wires all dependencies:

```python
from query_orchestrator_factory import create_query_orchestrator

orchestrator = create_query_orchestrator(
    cartridges_dir="./cartridges",
    device="cpu",
    enable_grain_system=True,
    enable_bitnet=False,
)

result = orchestrator.process_query("What is sleep consolidation?")
```

**What it does:**
1. Initialize Phase 3E components (CartridgeLoader, MTR, grain system)
2. Wrap them in POSIX interfaces (GrainEngine, CartridgeEngine adapters)
3. Create ResonanceWeightService, RuleBasedTriageAgent, HeartbeatService
4. Wire into QueryOrchestrator
5. Return ready-to-use orchestrator

---

## Which Orchestrator to Use?

| Orchestrator | Location | When to Use |
|--------------|----------|------------|
| **POSIX (Phase 3B)** | `query_orchestrator_posix.py` | **Default.** Clean interfaces, testable, extensible. Use this. |
| **Phase 3E (legacy)** | `phase3e_orchestrator.py` | Historical reference only. Monolithic, harder to test. Don't use for new work. |
| **Factory** | `query_orchestrator_factory.py` | Use to bootstrap the POSIX orchestrator with dependency wiring. |

---

## Cascade Thresholds

Confidence must exceed threshold to pass a layer:

```python
FALLBACK_THRESHOLDS = {
    "GRAIN":     0.90,      # High bar; likely explicit lookup
    "CARTRIDGE": 0.70,      # Medium bar; factual search
    # Phase 4+:
    # "BITNET":    0.75,    # Learned inference
    # "SPECIALIST": 0.65,   # Domain-specific models
    # "LLM":       0.0,     # Fallback (always passes)
}
```

If confidence < threshold, triage moves to next layer. If all layers fail, answer is "I don't know."

---

## State Coordination (Coupling & Phases)

**Phase 3B.3:** Coupling validation layer ensures L0–L5 constraints are satisfied.

```python
# During cascade, after each engine:
if self.coupling_validator:
    self.coupling_validator.validate_and_record(query_id, "L2", "L4")
```

This prevents invalid state transitions (e.g., L2 heuristic applied before L0 bootstrap is ready).

---

## Next: Query → Answer Walkthrough

**Example: "What is ATP?"**

1. **Query created.** ID = `abc123`, turn = 45
2. **Metabolism check.** Sleep cycle not due; skip
3. **Triage.** 2 words → "very short" → GRAIN → CARTRIDGE
4. **Pause.** Heartbeat blocks sleep pipeline
5. **GRAIN attempt.** GrainEngine searches activation cache
   - Finds "ATP_energy_currency_grain" (0.95 confidence)
   - **Passes threshold (0.90)** ✓
6. **Finalize.** Record resonance, emit answer
7. **Resume.** Heartbeat unblocks sleep, advances turn to 46
8. **Result returned.** Confidence 0.95, latency 4ms

---

## Debugging & Inspection

**View orchestrator state:**
```python
metrics = orchestrator.get_metrics()
print(f"Queries answered: {metrics['queries_answered']}")
print(f"Success rate: {metrics['success_rate']:.1%}")
print(f"P95 latency: {metrics['latency_p95_ms']:.1f}ms")
```

**Enable verbose triage:**
```python
triage = RuleBasedTriageAgent(verbose=True)
orchestrator = QueryOrchestrator(..., triage_agent=triage)
```

**Check Redis state:**
```bash
redis-cli KEYS "kitbash:*"
redis-cli GET "kitbash:queries:state:<query_id>"
```

---

## Phase 4+ Extensions

Phase 4 will add:

- **Mamba context service:** Temporal sliding window (current placeholder is no-op)
- **BitNet engine:** Learned inference via ternary quantization
- **Specialist models:** Fine-tuned models for domain-specific queries
- **LLM fallback:** OpenAI/Claude for open-ended reasoning

Phase 5 will add:

- **TVTropes embedding space:** Narrative pattern indexing for episodic retrieval
- **Meta-cartridges:** Navigational indexes across domain cartridges
- **Procedural edge extraction:** Sleep stage 1.5 + 2.5 for query chain consolidation

All phases maintain the same POSIX interface. Engines are pluggable.
