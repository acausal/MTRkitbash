"""
kitbash/orchestration/query_orchestrator.py

QueryOrchestrator - the single external entry point for user queries.

Coordinates:
  1. Background work scheduling (via MetabolismScheduler)
  2. Mamba context retrieval (temporal windows)
  3. Triage routing decision (also routes background work)
  4. PAUSE background work (heartbeat.pause())
  5. Serial engine cascade (Complexity Sieve)
  6. RESUME background work (heartbeat.resume())
  7. Resonance pattern recording & Turn Sync
  8. Advance turn counter

Phase 3B MVP: GRAIN → CARTRIDGE only
Phase 4+: Add BITNET, LLM, specialists

Standardized for Phase 3B MVP.
"""

import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from interfaces.triage_agent import TriageAgent, TriageRequest, TriageDecision
from interfaces.inference_engine import InferenceEngine, InferenceRequest, InferenceResponse
from interfaces.mamba_context_service import MambaContextService, MambaContextRequest
from memory.resonance_weights import ResonanceWeightService
from heartbeat_service import HeartbeatService
from metabolism_scheduler import MetabolismScheduler

# Phase 3B.3: Coupling Geometry Validation
try:
    from redis_coupling import CouplingValidator
except ImportError:
    CouplingValidator = None  # Graceful degradation if not available

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    """Final result returned to the caller."""
    query_id: str
    answer: Optional[str]
    confidence: float
    engine_name: str
    layer_results: List["LayerAttempt"]
    triage_reasoning: str
    triage_latency_ms: float
    total_latency_ms: float
    resonance_pattern_recorded: bool
    coupling_deltas: List[Dict[str, Any]] = field(default_factory=list)  # Phase 3B.3


@dataclass
class LayerAttempt:
    """Record of a single engine attempt during cascade."""
    engine_name: str
    confidence: float
    threshold: float
    passed: bool
    latency_ms: float
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# No-op diagnostic feed
# ---------------------------------------------------------------------------

class _NoOpDiagnosticFeed:
    """Silent stand-in for DiagnosticFeed when Redis is unavailable."""
    def log_query_created(self, *a, **kw): pass
    def log_query_started(self, *a, **kw): pass
    def log_layer_attempt(self, *a, **kw): pass
    def log_layer_hit(self, *a, **kw): pass
    def log_layer_miss(self, *a, **kw): pass
    def log_escalation(self, *a, **kw): pass
    def log_error(self, *a, **kw): pass
    def log_query_completed(self, *a, **kw): pass
    def log_metric(self, *a, **kw): pass


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class QueryOrchestrator:
    """
    Main coordinator for user-facing queries.
    
    Phase 3B MVP: Cascades through GRAIN → CARTRIDGE
    Phase 4+: Will add BITNET, LLM, specialists
    """

    FALLBACK_THRESHOLDS: Dict[str, float] = {
        "GRAIN":     0.90,
        "CARTRIDGE": 0.70,
        # Phase 4+:
        # "BITNET":    0.75,
        # "SPECIALIST": 0.65,
        # "LLM":       0.0,
    }

    ESCALATE_SENTINEL = "ESCALATE"

    def __init__(
        self,
        triage_agent: TriageAgent,
        engines: Dict[str, InferenceEngine],
        mamba_service: MambaContextService,
        resonance: ResonanceWeightService,
        heartbeat: Optional[HeartbeatService] = None,
        metabolism_scheduler: Optional[MetabolismScheduler] = None,
        shannon=None,
        diagnostic_feed=None,
        redis_client=None,  # Phase 3B.3: For coupling validation
    ) -> None:
        self.triage_agent = triage_agent
        self.engines = engines
        self.mamba_service = mamba_service
        self.resonance = resonance
        self.shannon = shannon

        # Week 3 Metabolism components
        self.heartbeat = heartbeat or HeartbeatService(initial_turn=0)
        self.metabolism_scheduler = metabolism_scheduler

        # Phase 3B.3: Coupling validation
        self.coupling_validator = None
        if redis_client and CouplingValidator:
            try:
                self.coupling_validator = CouplingValidator(redis_client)
            except Exception as e:
                logger.warning(f"Could not initialize coupling validator: {e}")

        self.feed = self._init_feed(diagnostic_feed)

        self._metrics: Dict[str, Any] = {
            "queries_total": 0,
            "queries_answered": 0,
            "queries_exhausted": 0,
            "layer_hits": {},
            "layer_attempts": {},
            "triage_latencies_ms": [],
            "total_latencies_ms": [],
            "heartbeat_pauses": 0,
            "metabolism_cycles_run": 0,
        }

    def process_query(
        self,
        user_query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> QueryResult:
        """
        Process a user query through the full orchestration pipeline.
        """
        query_id = str(uuid.uuid4())
        total_start = time.perf_counter()
        context = context or {}

        self.feed.log_query_created(query_id, user_query)
        self.feed.log_query_started(query_id)

        # PHASE 1: Metabolism check
        if self.metabolism_scheduler:
            try:
                # Sync turn to scheduler before checking if work is due
                self.metabolism_scheduler.current_turn = self.heartbeat.turn_number
                bg_status = self.metabolism_scheduler.step()
                if bg_status.get("executed"):
                    self._metrics["metabolism_cycles_run"] += 1
            except Exception as e:
                logger.warning(f"Metabolism scheduler failed: {e}")
                self.feed.log_error(query_id, "METABOLISM_SCHEDULER", str(e))

        # PHASE 2: Context retrieval
        mamba_context = self._get_mamba_context(user_query, context)
        context["mamba_context"] = mamba_context

        # PHASE 3: Triage
        triage_start = time.perf_counter()
        decision = self._get_triage_decision(user_query, context, query_id)
        triage_latency = (time.perf_counter() - triage_start) * 1000

        # PHASE 4: PAUSE background work
        if self.heartbeat:
            try:
                self.heartbeat.pause(priority="query_exec")
                self._metrics["heartbeat_pauses"] += 1
            except Exception as e:
                logger.warning(f"Heartbeat pause failed: {e}")
                self.feed.log_error(query_id, "HEARTBEAT_PAUSE", str(e))

        try:
            # PHASE 5: Engine cascade
            layer_results: List[LayerAttempt] = []
            winning_response: Optional[InferenceResponse] = None

            for layer_name in decision.layer_sequence:
                if layer_name == self.ESCALATE_SENTINEL:
                    break

                if layer_name not in self.engines:
                    logger.warning(f"Layer {layer_name} missing from engines.")
                    continue

                threshold = decision.confidence_thresholds.get(
                    layer_name, self.FALLBACK_THRESHOLDS.get(layer_name, 0.5)
                )

                attempt, response = self._attempt_layer(
                    layer_name, threshold, user_query, context, decision, query_id
                )
                layer_results.append(attempt)
                self._record_layer_metric(layer_name, attempt)

                # Phase 3B.3: Coupling validation after layer processes
                if self.coupling_validator:
                    try:
                        # Validate this layer against appropriate previous layers
                        if "L2" in layer_name:
                            self.coupling_validator.validate_and_record(query_id, "L1", "L2")
                        elif "L4" in layer_name:
                            self.coupling_validator.validate_and_record(query_id, "L2", "L4")
                            self.coupling_validator.validate_and_record(query_id, "L4", "L3")
                            self.coupling_validator.validate_and_record(query_id, "L4", "L5")
                    except Exception as e:
                        logger.debug(f"Coupling validation failed: {e}")
                        # Graceful degradation - continue query processing

                if attempt.passed:
                    winning_response = response
                    break

            # PHASE 6: Finalize response
            total_latency = (time.perf_counter() - total_start) * 1000
            pattern_recorded = False

            if winning_response and winning_response.answer:
                answer = winning_response.answer
                confidence = winning_response.confidence
                engine_name = winning_response.engine_name

                # Record pattern to resonance
                pattern_hash = self._hash_query(user_query)
                if pattern_hash in self.resonance.weights:
                    self.resonance.reinforce_pattern(pattern_hash)
                else:
                    self.resonance.record_pattern(
                        pattern_hash,
                        metadata={
                            "query": user_query[:200],
                            "engine": engine_name,
                            "query_id": query_id,
                        },
                    )
                pattern_recorded = True

                if self.shannon:
                    self._record_phantom_hit(winning_response, user_query)

                self.feed.log_query_completed(query_id, engine_name, confidence, total_latency)
                self._metrics["queries_answered"] += 1
            else:
                answer = "I don't know."
                confidence = 0.0
                engine_name = "NONE"
                self.feed.log_query_completed(query_id, "NONE", 0.0, total_latency)
                self._metrics["queries_exhausted"] += 1

            self._metrics["queries_total"] += 1
            self._metrics["triage_latencies_ms"].append(triage_latency)
            self._metrics["total_latencies_ms"].append(total_latency)

            # Phase 3B.3: Collect coupling deltas
            coupling_deltas = []
            if self.coupling_validator:
                try:
                    coupling_deltas = self.coupling_validator.get_deltas_for_query(query_id)
                except Exception as e:
                    logger.debug(f"Could not retrieve coupling deltas: {e}")

            return QueryResult(
                query_id=query_id,
                answer=answer,
                confidence=confidence,
                engine_name=engine_name,
                layer_results=layer_results,
                triage_reasoning=decision.reasoning,
                triage_latency_ms=triage_latency,
                total_latency_ms=total_latency,
                resonance_pattern_recorded=pattern_recorded,
                coupling_deltas=coupling_deltas,  # Phase 3B.3
            )

        finally:
            # PHASE 7 & 8: Resume & Advance Clock
            if self.heartbeat:
                try:
                    self.heartbeat.resume()
                    # Advance the master clock
                    new_turn = self.heartbeat.advance_turn()
                    
                    # Sync turn across stateful services
                    if hasattr(self.resonance, 'current_turn'):
                        self.resonance.current_turn = new_turn
                    if hasattr(self.triage_agent, 'current_turn'):
                        self.triage_agent.current_turn = new_turn
                        
                except Exception as e:
                    logger.warning(f"Turn advancement failed: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Return session-level performance metrics."""
        lats = self._metrics["total_latencies_ms"]
        
        def percentile(data, p):
            if not data: return 0.0
            sorted_data = sorted(data)
            return sorted_data[min(int(len(sorted_data) * p / 100), len(sorted_data) - 1)]
        
        return {
            "queries_total": self._metrics["queries_total"],
            "queries_answered": self._metrics["queries_answered"],
            "queries_exhausted": self._metrics["queries_exhausted"],
            "success_rate": (
                self._metrics["queries_answered"] / self._metrics["queries_total"]
                if self._metrics["queries_total"] > 0 else 0.0
            ),
            "latency_p50_ms": percentile(lats, 50),
            "latency_p95_ms": percentile(lats, 95),
            "latency_p99_ms": percentile(lats, 99),
            "avg_latency_ms": sum(lats) / len(lats) if lats else 0.0,
            "heartbeat_pauses": self._metrics["heartbeat_pauses"],
            "metabolism_cycles": self._metrics["metabolism_cycles_run"],
        }

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _init_feed(self, diagnostic_feed):
        """Initialize diagnostic feed (or no-op if unavailable)."""
        if diagnostic_feed:
            return diagnostic_feed
        return _NoOpDiagnosticFeed()

    def _get_mamba_context(self, user_query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve mamba context for query."""
        try:
            request = MambaContextRequest(
                query=user_query,
                context=context.get("mamba_context", {}),
            )
            return self.mamba_service.get_context(request)
        except Exception as e:
            logger.warning(f"Mamba context retrieval failed: {e}")
            return {}

    def _get_triage_decision(
        self,
        user_query: str,
        context: Dict[str, Any],
        query_id: str,
    ) -> TriageDecision:
        """Get triage decision for layer sequence."""
        try:
            request = TriageRequest(
                query=user_query,
                context=context,
                query_id=query_id,
            )
            return self.triage_agent.decide(request)
        except Exception as e:
            logger.warning(f"Triage decision failed: {e}")
            # Fallback: Try GRAIN only
            return TriageDecision(
                layer_sequence=["GRAIN"],
                confidence_thresholds={"GRAIN": 0.90},
                reasoning=f"Fallback to GRAIN due to triage error: {e}",
            )

    def _attempt_layer(
        self,
        layer_name: str,
        threshold: float,
        user_query: str,
        context: Dict[str, Any],
        decision: TriageDecision,
        query_id: str,
    ) -> tuple:
        """Attempt a single inference layer."""
        engine = self.engines[layer_name]
        attempt_start = time.perf_counter()

        try:
            request = InferenceRequest(
                query=user_query,
                context=context,
                query_id=query_id,
            )
            response = engine.infer(request)
            latency = (time.perf_counter() - attempt_start) * 1000

            passed = response.confidence >= threshold
            
            attempt = LayerAttempt(
                engine_name=layer_name,
                confidence=response.confidence,
                threshold=threshold,
                passed=passed,
                latency_ms=latency,
            )

            if passed:
                self.feed.log_layer_hit(query_id, layer_name, response.confidence)
            else:
                self.feed.log_layer_miss(query_id, layer_name, response.confidence, threshold)

            return attempt, response

        except Exception as e:
            latency = (time.perf_counter() - attempt_start) * 1000
            logger.warning(f"Layer {layer_name} failed: {e}")
            
            attempt = LayerAttempt(
                engine_name=layer_name,
                confidence=0.0,
                threshold=threshold,
                passed=False,
                latency_ms=latency,
                error=str(e),
            )
            
            self.feed.log_error(query_id, layer_name, str(e))
            return attempt, None

    def _record_layer_metric(self, layer_name: str, attempt: LayerAttempt) -> None:
        """Record metrics for layer attempt."""
        if layer_name not in self._metrics["layer_attempts"]:
            self._metrics["layer_attempts"][layer_name] = 0
        self._metrics["layer_attempts"][layer_name] += 1

        if attempt.passed:
            if layer_name not in self._metrics["layer_hits"]:
                self._metrics["layer_hits"][layer_name] = 0
            self._metrics["layer_hits"][layer_name] += 1

    def _hash_query(self, user_query: str) -> str:
        """Hash query for resonance pattern matching."""
        return hashlib.sha256(user_query.encode()).hexdigest()[:16]

    def _record_phantom_hit(self, response: InferenceResponse, user_query: str) -> None:
        """Record to Shannon grain orchestrator if available."""
        if self.shannon:
            try:
                self.shannon.record_phantom_hit(
                    query=user_query,
                    engine=response.engine_name,
                    confidence=response.confidence,
                )
            except Exception as e:
                logger.debug(f"Shannon recording failed: {e}")
