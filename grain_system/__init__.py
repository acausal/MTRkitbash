"""
Grain System Package - Modularized Crystallization Pipeline

A local-first, deterministic grain crystallization system for continuous learning
without catastrophic forgetting.

Core pipeline:
1. PhantomTracker - Track persistent query patterns
2. GrainRegistry - Store and index crystallized grains
3. TernaryCrush - Compress patterns to ternary representation
4. GrainCrystallizer - Persist grains to storage
5. ShannonGrainOrchestrator - End-to-end coordination

Public API:
- Use ShannonGrainOrchestrator for end-to-end operations
- Data structures: EpistemicLevel, GrainState, PhantomCandidate, GrainMetadata, TernaryDelta
- Sub-components available for direct use if needed (PhantomTracker, GrainRegistry, etc.)

Author: Kitbash Team
Date: February 2026
Phase: 2C Consolidation → Modularized (May 2026)
"""

# Data structures (leaf node, no dependencies)
from .data_structures import (
    EpistemicLevel,
    GrainState,
    PhantomCandidate,
    GrainMetadata,
    TernaryDelta,
)

# Core modules
from .phantom_tracker import PhantomTracker
from .grain_registry import GrainRegistry
from .ternary_crush import TernaryCrush
from .grain_crystallizer import GrainCrystallizer, GrainCrystallizationReport
from .orchestrator import ShannonGrainOrchestrator

__all__ = [
    # Enums
    "EpistemicLevel",
    "GrainState",
    # Dataclasses
    "PhantomCandidate",
    "GrainMetadata",
    "TernaryDelta",
    # Core components
    "PhantomTracker",
    "GrainRegistry",
    "TernaryCrush",
    "GrainCrystallizer",
    "GrainCrystallizationReport",
    # Orchestrator (main entry point)
    "ShannonGrainOrchestrator",
]

__version__ = "2.0.0-modularized"
__author__ = "Kitbash Team"
