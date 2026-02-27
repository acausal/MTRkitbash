"""
MockMambaService - MVP stub for MambaContextService.

Returns empty MambaContext for all requests. All four time windows are
populated with empty dicts so downstream code can safely read any window
without KeyError. MVP queries primarily use 1hour and 1day.

Tracks call count for test assertions. Replace with AccessLogMambaService
(Phase 3C) or real Mamba integration (Phase 4) without changing callers.

Location: src/context/mock_mamba_service.py
"""

from typing import List
from interfaces.mamba_context_service import (
    MambaContextService,
    MambaContextRequest,
    MambaContext,
)


class MockMambaService(MambaContextService):
    """
    MVP no-op implementation of MambaContextService.

    Always returns an empty MambaContext with all four time windows
    populated. Active topics and topic shifts are empty - triage will
    see no active context and route accordingly.

    The call_count attribute lets test harnesses assert that the service
    was actually called without needing to inspect internals.

    Expansion path:
        Phase 3C → AccessLogMambaService (uses cartridge access logs)
        Phase 4  → Real Mamba with stateful hidden_state swapping
    """

    def __init__(self) -> None:
        self.call_count: int = 0
        self.last_request: MambaContextRequest | None = None

    def get_context(self, request: MambaContextRequest) -> MambaContext:
        """
        Return empty context for all requests.

        Args:
            request: MambaContextRequest (windows field ignored for MVP)

        Returns:
            MambaContext with all windows empty and no active topics.
            Never returns None - contract matches MambaContextService ABC.
        """
        self.call_count += 1
        self.last_request = request

        return MambaContext(
            context_1hour={},
            context_1day={},
            context_72hours={},
            context_1week={},
            active_topics=[],
            topic_shifts=[],
            hidden_state=None,
        )

    def reset(self) -> None:
        """Reset call counter. Useful between test cases."""
        self.call_count = 0
        self.last_request = None
