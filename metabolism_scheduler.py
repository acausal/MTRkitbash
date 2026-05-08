"""
MetabolismScheduler: Coordinates timing for all metabolism cycles.

Manages three types of background work:
  - Background: Continuous maintenance (decay, etc). Runs every N turns.
  - Daydreaming: Testing/validation. Triggered on idle (Phase 4).
  - Sleep: Consolidation. Triggered on schedule (Phase 4).

MVP implements background cycle (via HeartbeatService). Other cycles are stubs.
"""

from typing import Dict, Any, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CycleType(Enum):
    """Types of metabolism cycles."""
    BACKGROUND = "background"  # Continuous maintenance
    DAYDREAMING = "daydreaming"  # Testing/validation (Phase 4)
    SLEEP = "sleep"  # Consolidation (Phase 4)


class MetabolismScheduler:
    """
    Coordinates timing and execution of metabolism cycles.

    Responsible for:
      - Triggering background work every N turns
      - Offering daydream opportunities on idle
      - Scheduling sleep consolidation
      - Tracking cycle statistics
    """

    def __init__(
        self,
        background_cycle,  # BackgroundMetabolismCycle
        heartbeat_service,  # HeartbeatService
        background_interval: int = 100,  # Run background every N turns
    ):
        """
        Initialize scheduler.

        Args:
            background_cycle: BackgroundMetabolismCycle instance
            heartbeat_service: HeartbeatService instance
            background_interval: Turns between background work (default 100)
        """
        self.background_cycle = background_cycle
        self.heartbeat_service = heartbeat_service
        self.background_interval = background_interval

        # Track last execution turns
        self.last_background_turn = -background_interval  # Force first run
        self.last_daydream_turn = -1
        self.last_sleep_turn = -1

        # Statistics
        self.background_runs = 0
        self.daydream_runs = 0
        self.sleep_runs = 0

    def step(self, system_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute one scheduler step.

        Called periodically (typically once per query cycle). Checks if any
        cycles are due and executes them if appropriate.

        Args:
            system_state: Optional system state dict (for Phase 4 scheduling)

        Returns:
            Information about what was executed
        """
        current_turn = self.heartbeat_service.current_turn
        results = {"turn": current_turn, "cycles_executed": []}

        # Check if background is due
        if current_turn - self.last_background_turn >= self.background_interval:
            result = self._execute_background(current_turn)
            results["cycles_executed"].append(result)

        # Phase 4: Daydream and sleep are stubs
        # (No condition checking for MVP)

        return results

    def trigger_daydream(self, system_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Manually trigger daydream cycle.

        Phase 4: Will test consistency and update learned patterns.
        MVP: Stub that logs and returns.

        Args:
            system_state: Optional system state for Phase 4 logic

        Returns:
            Execution result
        """
        self.daydream_runs += 1

        logger.info(
            f"MetabolismScheduler.trigger_daydream() #{self.daydream_runs}: "
            f"(Phase 4 implementation, skipping for MVP)"
        )

        return {
            "cycle_type": CycleType.DAYDREAMING.value,
            "status": "stubbed_for_phase_4",
            "run_number": self.daydream_runs,
        }

    def trigger_sleep(self, system_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Manually trigger sleep cycle.

        Phase 4: Will consolidate patterns, clear cruft, update ShannonGrainOrchestrator.
        MVP: Stub that logs and returns.

        Args:
            system_state: Optional system state for Phase 4 logic

        Returns:
            Execution result
        """
        self.sleep_runs += 1

        logger.info(
            f"MetabolismScheduler.trigger_sleep() #{self.sleep_runs}: "
            f"(Phase 4 implementation, skipping for MVP)"
        )

        return {
            "cycle_type": CycleType.SLEEP.value,
            "status": "stubbed_for_phase_4",
            "run_number": self.sleep_runs,
        }

    def _execute_background(self, current_turn: int) -> Dict[str, Any]:
        """
        Execute background cycle if due.

        Internal method called by step(). Checks interval, executes cycle,
        updates tracking.

        Args:
            current_turn: Current turn number from heartbeat

        Returns:
            Execution result
        """
        self.background_runs += 1
        self.last_background_turn = current_turn

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"MetabolismScheduler._execute_background() #{self.background_runs}: "
                f"turn {current_turn}"
            )

        # Execute via heartbeat (respects pause/resume)
        try:
            result = self.heartbeat_service.step(self.background_cycle)
            result["cycle_type"] = CycleType.BACKGROUND.value
            result["run_number"] = self.background_runs
            return result
        except Exception as e:
            logger.error(
                f"MetabolismScheduler._execute_background(): heartbeat.step() failed: {e}"
            )
            return {
                "cycle_type": CycleType.BACKGROUND.value,
                "run_number": self.background_runs,
                "success": False,
                "error": str(e),
            }

    def get_status(self) -> Dict[str, Any]:
        """
        Get scheduler status (for REPL/diagnostics).

        Returns:
            Dictionary with cycle counts and scheduling info
        """
        current_turn = self.heartbeat_service.current_turn
        turns_since_background = current_turn - self.last_background_turn

        return {
            "current_turn": current_turn,
            "background_interval": self.background_interval,
            "background_runs": self.background_runs,
            "background_due_in": max(0, self.background_interval - turns_since_background),
            "daydream_runs": self.daydream_runs,
            "sleep_runs": self.sleep_runs,
            "heartbeat_status": self.heartbeat_service.get_status(),
        }

    def reset(self) -> None:
        """
        Reset all counters.

        Used for testing. Clears run counts and scheduling state.
        """
        self.last_background_turn = -self.background_interval
        self.last_daydream_turn = -1
        self.last_sleep_turn = -1
        self.background_runs = 0
        self.daydream_runs = 0
        self.sleep_runs = 0

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("MetabolismScheduler.reset(): counters cleared")
