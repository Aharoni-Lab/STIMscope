"""Shared fixtures + MockI2CBackend for L3-projector test modules.

The HAL Protocol pattern from `tests/L3_hardware/fakes_ids_peak.py`
applied to the I²C bus seam in `dlpc_i2c.py` (and its sibling files
that share the same `execute_i2c_transfer` import).

Stage-2 chars for dlpc_i2c.py and the related ZMQ_sender_mask
Python modules patch `dlpc_i2c.execute_i2c_transfer` to point at a
`MockI2CBackend` instance, allowing tests to:
- Record every (bus, addr, cmd, data, read_len) call made
- Return canned read responses (configurable per opcode)
- Assert byte-exact payload structure against TI datasheet
- Verify call ordering (e.g. fast_phase_switch order = 0x96 → 0x54 → 0x05)

No real I²C bus access. No hardware required. Tests run on any host.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ZMQ_PATH = REPO_ROOT / "STIMscope" / "ZMQ_sender_mask"
if str(ZMQ_PATH) not in sys.path:
    sys.path.insert(0, str(ZMQ_PATH))


@dataclass
class I2CCall:
    """One captured call to execute_i2c_transfer."""
    bus: int
    addr: int
    opcode: int
    data: List[int] = field(default_factory=list)
    read_len: int = 0


class MockI2CBackend:
    """HAL Protocol-shaped fake for ``i2c_send_custom_cmd.execute_i2c_transfer``.

    Records every call + serves canned read responses.

    Usage:
        from tests.L3_projector.conftest import MockI2CBackend
        mock = MockI2CBackend()
        mock.set_read_response(opcode=0xD0, response=[0x01])  # init_complete
        mock.set_read_response(opcode=0xD4, response=[0x00, 0x0C])  # DLPC3479 id

        with patch.object(dlpc_i2c, 'execute_i2c_transfer', mock):
            dlpc_i2c.wait_init_done(bus=1)

        assert mock.calls[0].opcode == 0xD0
        assert mock.write_calls[0].opcode == ...  # filter helper
    """

    def __init__(self) -> None:
        self.calls: List[I2CCall] = []
        self._read_responses: Dict[int, List[int]] = {}
        # Per-call dynamic response (overrides static map)
        self._dynamic_response: Optional[Callable[[I2CCall], List[int]]] = None
        # Errors to raise on next-N calls (one-shot list, popped)
        self._error_queue: List[Exception] = []

    # ─── Configuration ─────────────────────────────────────────────────────

    def set_read_response(self, opcode: int, response: Sequence[int]) -> None:
        """Set static canned response for a given read opcode."""
        self._read_responses[opcode] = list(response)

    def set_dynamic_response(self, fn: Callable[[I2CCall], List[int]]) -> None:
        """Set a callable that produces response per-call (overrides static map)."""
        self._dynamic_response = fn

    def raise_on_next_call(self, exc: Exception) -> None:
        """Queue an exception to raise on the next execute_i2c_transfer call."""
        self._error_queue.append(exc)

    # ─── Filters / introspection ──────────────────────────────────────────

    @property
    def write_calls(self) -> List[I2CCall]:
        """Calls where read_len == 0 (pure writes)."""
        return [c for c in self.calls if c.read_len == 0]

    @property
    def read_calls(self) -> List[I2CCall]:
        return [c for c in self.calls if c.read_len > 0]

    def calls_for_opcode(self, opcode: int) -> List[I2CCall]:
        return [c for c in self.calls if c.opcode == opcode]

    def opcode_sequence(self) -> List[int]:
        """Ordered list of opcodes called."""
        return [c.opcode for c in self.calls]

    def reset(self) -> None:
        self.calls.clear()
        self._read_responses.clear()
        self._dynamic_response = None
        self._error_queue.clear()

    # ─── The mock callable ─────────────────────────────────────────────────

    def __call__(
        self,
        bus_num: int,
        addr: int,
        cmd: int,
        data: Optional[Sequence[int]] = None,
        read_len: int = 0,
    ) -> List[int]:
        """Mimics execute_i2c_transfer signature."""
        if self._error_queue:
            raise self._error_queue.pop(0)
        call = I2CCall(
            bus=bus_num,
            addr=addr,
            opcode=cmd,
            data=list(data or []),
            read_len=read_len,
        )
        self.calls.append(call)

        if read_len == 0:
            return []

        # Read call — serve canned response
        if self._dynamic_response is not None:
            return self._dynamic_response(call)
        if cmd in self._read_responses:
            return list(self._read_responses[cmd])
        # Default: return zero bytes (caller-side decoders will see init_complete=False, etc.)
        return [0] * read_len


@pytest.fixture
def mock_i2c():
    """Per-test MockI2CBackend instance."""
    return MockI2CBackend()
