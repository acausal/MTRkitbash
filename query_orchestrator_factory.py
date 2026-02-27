"""
query_orchestrator_factory.py

Factory for creating a POSIX-compliant QueryOrchestrator with all dependencies
properly wired.

This module bridges the old Phase 3E implementation (which builds everything
from scratch) with the new POSIX interface layer (which accepts dependency
objects). It creates adapters and wires them together.

Usage:
    orchestrator = create_query_orchestrator(
        cartridges_dir="./cartridges",
        enable_grain_system=True,
        device="cpu"
    )
    
    result = orchestrator.process_query("What is ATP?")
"""

import logging
from typing import Optional, Dict, Any

from interfaces.triage_agent import TriageAgent
from interfaces.inference_engine import InferenceEngine
from interfaces.mamba_context_service import MambaContextService
from memory.resonance_weights import ResonanceWeightService
from heartbeat_service import HeartbeatService

from grain_engine import GrainEngine
from cartridge_engine import CartridgeEngine
from bitnet_engine import BitNetEngine
from rule_based_triage import RuleBasedTriageAgent
from mock_mamba_service import MockMambaService

# Import existing Phase 3E components (unchanged)
try:
    from cartridge_loader import CartridgeInferenceEngine
    from grain_router import GrainRouter
    from MTR_v5_5_NN import KitbashMTREngine
    from grain_system import ShannonGrainOrchestrator
    from mtr_state_manager import MTRStateCheckpoint
    from mtr_grain_bridge import MTRGrainUnifiedPipeline, HatKappaMapper
    GRAIN_SYSTEM_AVAILABLE = True
except ImportError as e:
    GRAIN_SYSTEM_AVAILABLE = False
    logging.warning(f"Grain system not available: {e}")

logger = logging.getLogger(__name__)


def create_query_orchestrator(
    cartridges_dir: str = "./cartridges",
    vocab_size: int = 50257,
    d_model: int = 256,
    d_state: int = 144,
    state_dir: str = "data/state",
    grain_storage_dir: str = "./grains",
    device: str = "cpu",
    enable_grain_system: bool = True,
    dream_bucket_dir: str = "data/subconscious/dream_bucket",
    enable_bitnet: bool = False,
    bitnet_url: str = "http://127.0.0.1:8080",
    verbose: bool = False,
) -> Any:
    """
    Factory function to create a fully-wired QueryOrchestrator.
    
    This creates all the POSIX interface implementations, wires them to the
    existing Phase 3E components, and returns a QueryOrchestrator ready to use.
    
    Args:
        cartridges_dir: Path to cartridge files
        vocab_size: MTR vocabulary size
        d_model: MTR embedding dimension
        d_state: MTR state space dimension
        state_dir: MTR checkpoint directory
        grain_storage_dir: Grain registry directory
        device: torch device (cpu, cuda)
        enable_grain_system: Enable grain system
        dream_bucket_dir: Dream bucket directory
        enable_bitnet: Include BitNet in engine cascade
        bitnet_url: BitNet server URL
        verbose: Enable verbose logging
        
    Returns:
        QueryOrchestrator instance (from uploaded version, with POSIX interface)
        
    Raises:
        RuntimeError: If required components cannot be initialized
    """
    
    logger.info("Creating QueryOrchestrator with POSIX interfaces...")
    
    # ========================================================================
    # STEP 1: Build Phase 3E components (unchanged logic)
    # ========================================================================
    
    logger.debug("Step 1: Initializing Phase 3E components...")
    
    # CartridgeLoader
    try:
        cartridge_engine_phase3e = CartridgeInferenceEngine(cartridges_dir)
        logger.info(f"  ✓ CartridgeLoader: {cartridge_engine_phase3e.registry.get_stats()['total_facts']} facts")
    except Exception as e:
        logger.error(f"  ✗ Failed to initialize CartridgeLoader: {e}")
        raise
    
    # MTR Engine
    try:
        mtr_engine = KitbashMTREngine(
            vocab_size=vocab_size,
            d_model=d_model,
            d_state=d_state,
        )
        mtr_engine = mtr_engine.to(device)
        logger.info(f"  ✓ MTREngine initialized")
    except Exception as e:
        logger.error(f"  ✗ Failed to initialize MTREngine: {e}")
        raise
    
    # State Manager
    try:
        state_manager = MTRStateCheckpoint(state_dir)
        logger.info(f"  ✓ StateManager initialized")
    except Exception as e:
        logger.error(f"  ✗ Failed to initialize StateManager: {e}")
        raise
    
    # Grain System (if enabled)
    grain_router = None
    grain_orchestrator = None
    mtr_grain_pipeline = None
    
    if enable_grain_system and GRAIN_SYSTEM_AVAILABLE:
        try:
            grain_orchestrator = ShannonGrainOrchestrator(
                cartridge_id="default",
                storage_path=grain_storage_dir,
                cartridge_engine=cartridge_engine_phase3e
            )
            
            grain_router = GrainRouter(
                cartridges_dir=cartridges_dir,
                cartridge_engine=cartridge_engine_phase3e
            )
            
            first_cartridge = next(
                iter(cartridge_engine_phase3e.registry.cartridges.values())
            ) if cartridge_engine_phase3e.registry.cartridges else None
            
            mtr_grain_pipeline = MTRGrainUnifiedPipeline(
                mtr_grain_orchestrator=grain_orchestrator,
                grain_router=grain_router,
                trigger_interval=51,
                cartridge=first_cartridge
            )
            
            logger.info(f"  ✓ Grain system initialized ({grain_router.total_grains} grains)")
        except Exception as e:
            logger.warning(f"  ⚠ Grain system initialization failed: {e}")
            enable_grain_system = False
    
    # ========================================================================
    # STEP 2: Create POSIX interface implementations (adapters)
    # ========================================================================
    
    logger.debug("Step 2: Creating POSIX interface implementations...")
    
    # ResonanceWeightService (Tier 5)
    try:
        resonance_service = ResonanceWeightService(
            initial_stability=3.0,
            stability_growth=2.0,
            cleanup_threshold=0.001,
            spacing_sensitive=False
        )
        logger.info(f"  ✓ ResonanceWeightService initialized")
    except Exception as e:
        logger.error(f"  ✗ Failed to initialize ResonanceWeightService: {e}")
        raise
    
    # TriageAgent (routing)
    try:
        triage_agent = RuleBasedTriageAgent(verbose=verbose)
        logger.info(f"  ✓ RuleBasedTriageAgent initialized")
    except Exception as e:
        logger.error(f"  ✗ Failed to initialize TriageAgent: {e}")
        raise
    
    # InferenceEngines (adapters to Phase 3E components)
    engines: Dict[str, InferenceEngine] = {}
    
    try:
        # Grain Engine
        if grain_router:
            grain_engine = GrainEngine(grain_router)
            engines["GRAIN"] = grain_engine
            logger.info(f"  ✓ GrainEngine created")
    except Exception as e:
        logger.warning(f"  ⚠ Failed to create GrainEngine: {e}")
    
    try:
        # Cartridge Engine
        cartridge_engine = CartridgeEngine(
            {name: cart for name, cart in cartridge_engine_phase3e.registry.cartridges.items()}
        )
        engines["CARTRIDGE"] = cartridge_engine
        logger.info(f"  ✓ CartridgeEngine created")
    except Exception as e:
        logger.warning(f"  ⚠ Failed to create CartridgeEngine: {e}")
    
    try:
        # BitNet Engine (optional)
        if enable_bitnet:
            bitnet_engine = BitNetEngine(server_url=bitnet_url)
            engines["BITNET"] = bitnet_engine
            logger.info(f"  ✓ BitNetEngine created (server: {bitnet_url})")
    except Exception as e:
        logger.warning(f"  ⚠ BitNet initialization skipped: {e}")
    
    if not engines:
        logger.error("No inference engines available!")
        raise RuntimeError("Failed to create any inference engines")
    
    # MambaContextService (MVP: no-op)
    try:
        mamba_service = MockMambaService()
        logger.info(f"  ✓ MockMambaService initialized (Phase 4 placeholder)")
    except Exception as e:
        logger.error(f"  ✗ Failed to initialize MambaContextService: {e}")
        raise
    
    # HeartbeatService (pause/resume background work)
    try:
        heartbeat = HeartbeatService(initial_turn=0)
        logger.info(f"  ✓ HeartbeatService initialized")
    except Exception as e:
        logger.error(f"  ✗ Failed to initialize HeartbeatService: {e}")
        raise
    
    # ========================================================================
    # STEP 3: Create POSIX QueryOrchestrator with wired dependencies
    # ========================================================================
    
    logger.debug("Step 3: Creating POSIX QueryOrchestrator...")
    
    try:
        # Import the POSIX version from uploaded file
        from query_orchestrator_posix import QueryOrchestrator as POSIXQueryOrchestrator
        
        orchestrator = POSIXQueryOrchestrator(
            triage_agent=triage_agent,
            engines=engines,
            mamba_service=mamba_service,
            resonance=resonance_service,
            heartbeat=heartbeat,
            metabolism_scheduler=None,  # Phase 3B: no background scheduler yet
            shannon=grain_orchestrator if enable_grain_system else None,
            diagnostic_feed=None,  # Phase 3B: no Redis yet
            redis_client=None,  # Phase 3B: no Redis yet
        )
        
        logger.info(f"  ✓ QueryOrchestrator initialized with {len(engines)} engines")
        
    except ImportError:
        logger.warning("POSIX QueryOrchestrator not available, using Phase 3E version")
        # Fallback: return Phase 3E orchestrator (different interface)
        # This won't have full POSIX compliance but will work
        raise RuntimeError(
            "POSIX QueryOrchestrator (query_orchestrator_posix.py) not found. "
            "Please provide the POSIX version to enable interface-based architecture."
        )
    except Exception as e:
        logger.error(f"  ✗ Failed to create QueryOrchestrator: {e}")
        raise
    
    logger.info("✓ QueryOrchestrator fully initialized with POSIX interfaces")
    
    return orchestrator


def create_minimal_orchestrator(
    cartridges_dir: str = "./cartridges",
) -> Any:
    """
    Create a minimal QueryOrchestrator with just essentials.
    
    Useful for testing or MVP deployments.
    
    Args:
        cartridges_dir: Path to cartridges
        
    Returns:
        QueryOrchestrator instance
    """
    return create_query_orchestrator(
        cartridges_dir=cartridges_dir,
        device="cpu",
        enable_grain_system=False,
        enable_bitnet=False,
        verbose=False
    )


if __name__ == "__main__":
    """Test the factory."""
    import sys
    
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s: %(message)s"
    )
    
    try:
        print("Creating QueryOrchestrator via factory...\n")
        orch = create_query_orchestrator(verbose=True)
        print("\n✓ Factory test successful!")
        print(f"Orchestrator: {orch}")
        print(f"Engines: {list(orch.engines.keys())}")
    except Exception as e:
        print(f"\n✗ Factory test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
