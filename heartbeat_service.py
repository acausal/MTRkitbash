"""
HeartbeatService: Manages pause/resume for background metabolism cycles.

Provides OpenClaw-style heartbeat mechanism for coordinating background work
with incoming queries. When a query arrives, heartbeat pauses background work
and saves a checkpoint. When the query completes, background resumes.

Checkpoint contains:
  - turn_number: which turn we're on (for resuming decay)
  - interrupted_priority: what background work was paused (for diagnostics)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    """Minimal checkpoint state for background cycles."""
    turn_number: int
    interrupted_priority: Optional[str] = None
    timestamp: Optional[float] = None


class HeartbeatService:
    """
    Manages pause/resume lifecycle for background work.

    Maintains single checkpoint state. Can be paused/resumed multiple times
    during a single query (though MVP only pauses once per query).

    Thread-unsafe (designed for synchronous MVP, not async).
    """

    def __init__(self, initial_turn: int = 0):
        """
        Initialize heartbeat service.

        Args:
            initial_turn: Starting turn number (typically 0)
        """
        self.current_turn = initial_turn
        self.is_running = True  # Background cycle is running
        self.checkpoint: Optional[Checkpoint] = None

    def pause(self) -> Dict[str, Any]:
        """
        Save state and pause background work.

        Called when a query arrives. Saves current turn and the last priority
        that was being processed.

        Returns:
            Checkpoint data (for logging/diagnostics)
        """
        if not self.is_running:
            # Already paused, don't overwrite checkpoint
            if self.checkpoint:
                return {
                    "turn_number": self.checkpoint.turn_number,
                    "interrupted_priority": self.checkpoint.interrupted_priority,
                    "was_already_paused": True,
                }
            return {"error": "Already paused, no checkpoint"}

        # Save checkpoint
        self.checkpoint = Checkpoint(
            turn_number=self.current_turn,
            interrupted_priority=None,  # Will be set by background cycle
        )
        self.is_running = False

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"HeartbeatService.pause(): saved checkpoint at turn {self.current_turn}"
            )

        return {
            "turn_number": self.current_turn,
            "interrupted_priority": None,
            "was_already_paused": False,
        }

    def resume(self) -> Dict[str, Any]:
        """
        Restore state and resume background work.

        Called when a query completes (typically in finally block).
        Restores from checkpoint and allows background cycle to continue.

        Returns:
            Information about resumed state
        """
        if self.is_running:
            # Already running, nothing to do
            return {"was_already_running": True, "turn_number": self.current_turn}

        # Clear checkpoint and resume
        checkpoint_data = {
            "turn_number": self.checkpoint.turn_number if self.checkpoint else None,
            "interrupted_priority": (
                self.checkpoint.interrupted_priority if self.checkpoint else None
            ),
        }

        self.checkpoint = None
        self.is_running = True

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"HeartbeatService.resume(): background work resumed at turn {self.current_turn}"
            )

        return {
            "was_already_running": False,
            "checkpoint_data": checkpoint_data,
        }

    def step(self, background_cycle: Optional[Any] = None) -> Dict[str, Any]:
        """
        Execute one step of background work (if running and not paused).

        Called by MetabolismScheduler. If paused, this is a no-op.
        If running, calls background_cycle.run() to do the work.

        Args:
            background_cycle: BackgroundMetabolismCycle instance to call

        Returns:
            Information about what was executed
        """
        if not self.is_running:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("HeartbeatService.step(): paused, skipping background work")
            return {"executed": False, "reason": "paused"}

        if background_cycle is None:
            return {"executed": False, "reason": "no background_cycle provided"}

        try:
            # Run the background cycle
            checkpoint = background_cycle.run()

            # Record what priority was just executed (for checkpoint)
            if checkpoint and "priority" in checkpoint:
                if self.checkpoint:
                    self.checkpoint.interrupted_priority = checkpoint["priority"]

            return {
                "executed": True,
                "turn": self.current_turn,
                "cycle_checkpoint": checkpoint,
            }
        except Exception as e:
            logger.error(f"HeartbeatService.step(): background cycle failed: {e}")
            return {"executed": False, "error": str(e)}

    def advance_turn(self) -> int:
        """
        Increment turn counter.

        Called by background cycle when work is complete. This updates
        the reference point for resonance decay calculations.

        Returns:
            New turn number
        """
        self.current_turn += 1
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"HeartbeatService.advance_turn(): now at turn {self.current_turn}")
        return self.current_turn

    def get_status(self) -> Dict[str, Any]:
        """
        Get current heartbeat status (for REPL/diagnostics).

        Returns:
            Dictionary with running state, turn number, checkpoint info
        """
        return {
            "is_running": self.is_running,
            "current_turn": self.current_turn,
            "has_checkpoint": self.checkpoint is not None,
            "checkpoint": {
                "turn_number": self.checkpoint.turn_number,
                "interrupted_priority": self.checkpoint.interrupted_priority,
            }
            if self.checkpoint
            else None,
        }
