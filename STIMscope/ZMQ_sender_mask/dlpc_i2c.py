"""DLPC3479 I²C helper — datasheet-grounded primitives.

Source: DLPU081A Rev. A (June 2019). Every opcode and parameter layout
here is cross-referenced in `docs/hardware/I2C_COMMAND_REFERENCE.md`.

This module is the single source of truth for the DMD's I²C interface.
All other code that writes to the DLPC should go through `write_with_check`,
which reads back `0xD3` Communication Status after every write and raises
`DLPCRejected` if the controller rejected the opcode or a parameter.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from i2c_send_custom_cmd import execute_i2c_transfer

# ---------- Transport defaults ----------

ADDR_DEFAULT = 0x1B          # 7-bit; write=0x36, read=0x37
ADDR_ALT = 0x1D
BUS_DEFAULT = 1

# ---------- Opcodes (from I2C_COMMAND_REFERENCE.md) ----------

OP_OP_MODE_W = 0x05          # Operating Mode Select (write) — p. 8
OP_OP_MODE_R = 0x06
OP_EXT_VIDEO_FMT_W = 0x07    # External Video Source Format (write) — p. 11
OP_DISPLAY_SIZE_W = 0x12     # Display Size (write) — p. 23
OP_CURTAIN_W = 0x16          # Display Image Curtain (write) — p. 27
OP_FREEZE_W = 0x1A           # Image Freeze (write) — p. 29
OP_INPUT_SIZE_W = 0x2E       # Input Image Size (write) — p. 37
OP_LED_CTRL_METHOD_W = 0x50  # LED Output Control Method (write) — p. 40
OP_LED_ENABLE_W = 0x52       # RGB LED Enable (Display modes only) — p. 42
OP_LED_CURRENT_PWM_W = 0x54  # RGB LED Current PWM (write) — p. 44
OP_LED_MAX_PWM_W = 0x5C      # RGB LED Max Current PWM (write) — p. 47
OP_TRIG_IN_CFG_W = 0x90      # Trigger In Config (Internal only) — p. 55
OP_TRIG_OUT_CFG_W = 0x92     # Trigger Out Config — p. 57
OP_PATTERN_READY_W = 0x94    # Pattern Ready Config (Internal only) — p. 59
OP_PATTERN_CONFIG_W = 0x96   # Pattern Configuration — p. 61
OP_VALIDATE_EXPOSURE_R = 0x9D  # Validate Exposure Time — p. 67
OP_PATTERN_ORDER_TABLE_W = 0x98  # Pattern Order Table Entry — p. 63
OP_INT_PATTERN_CTRL_W = 0x9E   # Internal Pattern Control — p. 68
OP_SHORT_STATUS_R = 0xD0     # Short Status — p. 72
OP_SYSTEM_STATUS_R = 0xD1    # System Status — p. 73
OP_COMM_STATUS_R = 0xD3      # Communication Status — p. 76
OP_CONTROLLER_ID_R = 0xD4    # Controller Device ID — p. 77
OP_DMD_ID_R = 0xD5           # DMD Device ID — p. 78
OP_TEMPERATURE_R = 0xD6      # System Temperature — p. 79

# ---------- Enums ----------

# Operating modes (05h payload)
MODE_DISPLAY_EXT_VIDEO = 0x00
MODE_DISPLAY_TPG = 0x01
MODE_DISPLAY_SPLASH = 0x02
MODE_LIGHT_EXT_STREAM = 0x03
MODE_LIGHT_INT_STREAM = 0x04
MODE_LIGHT_SPLASH = 0x05
MODE_STANDBY = 0xFF

# Sequence type (96h byte 1)
SEQ_TYPE_1BIT_MONO = 0x00
SEQ_TYPE_1BIT_RGB = 0x01
SEQ_TYPE_8BIT_MONO = 0x02
SEQ_TYPE_8BIT_RGB = 0x03

# Illumination select (96h byte 3) — bits
ILLUM_RED = 0x01
ILLUM_GREEN = 0x02
ILLUM_BLUE = 0x04

# Trigger out config (92h byte 1)
TRIG_OUT_1 = 0
TRIG_OUT_2 = 1

# External video format (07h) — p. 11 Table 2
EXT_FMT_RGB888_24B_1CLK = 0x43  # default


# ---------- Exceptions ----------


class DLPCError(Exception):
    """Base for DLPC I²C failures."""


class DLPCTimeout(DLPCError):
    """Init-done or status poll exceeded the timeout."""


class DLPCRejected(DLPCError):
    """The DLPC rejected the last write (per 0xD3 bits). Check.status_byte."""

    def __init__(self, message: str, status_byte: int, rejected_opcode: int) -> None:
        super().__init__(message)
        self.status_byte = status_byte
        self.rejected_opcode = rejected_opcode


# ---------- Primitive I/O (uses i2c_send_custom_cmd.execute_i2c_transfer) ----------


def raw_write(bus: int, addr: int, opcode: int, data: Sequence[int] = ()) -> None:
    """Write opcode + data, no status check. Use `write_with_check` instead."""
    execute_i2c_transfer(bus, addr, opcode, list(data), 0)


def raw_read(bus: int, addr: int, opcode: int, data: Sequence[int], read_len: int) -> List[int]:
    """Write opcode + in-data, then read `read_len` bytes back."""
    return execute_i2c_transfer(bus, addr, opcode, list(data), read_len)


# ---------- Status readers ----------


@dataclass
class ShortStatus:
    raw: int
    init_complete: bool  # b(0)
    comm_error: bool     # b(1)
    system_error: bool   # b(3)
    flash_erase_complete: bool  # b(4)
    flash_error: bool    # b(5)
    light_control_seq_error: bool  # b(6)
    main_or_boot: bool   # b(7)  0=Main, 1=Boot

    @classmethod
    def decode(cls, byte: int) -> "ShortStatus":
        return cls(
            raw=byte,
            init_complete=bool(byte & 0x01),
            comm_error=bool(byte & 0x02),
            system_error=bool(byte & 0x08),
            flash_erase_complete=bool(byte & 0x10),
            flash_error=bool(byte & 0x20),
            light_control_seq_error=bool(byte & 0x40),
            main_or_boot=bool(byte & 0x80),
        )


@dataclass
class CommStatus:
    """Decoded D3h byte 5 flags. See p. 76."""
    raw_status: int
    rejected_opcode: int
    invalid_command: bool
    invalid_param_value: bool
    invalid_param_count: bool
    read_command_error: bool
    command_processing_error: bool
    flash_batch_error: bool
    bus_timeout: bool

    @property
    def ok(self) -> bool:
        # Bit 7 is reserved per the datasheet; only the seven defined error
        # bits (b0..b6) count as a real failure.
        return (self.raw_status & 0x7F) == 0

    @classmethod
    def decode(cls, resp: Sequence[int]) -> "CommStatus":
        # Response is 6 bytes. Byte 5 = error flags, byte 6 = last rejected opcode.
        if len(resp) < 6:
            raise DLPCError(f"0xD3 response too short: {len(resp)} bytes")
        status = resp[4]
        opcode = resp[5]
        return cls(
            raw_status=status,
            rejected_opcode=opcode,
            invalid_command=bool(status & 0x01),
            invalid_param_value=bool(status & 0x02),
            invalid_param_count=bool(status & 0x04),
            read_command_error=bool(status & 0x08),
            command_processing_error=bool(status & 0x10),
            flash_batch_error=bool(status & 0x20),
            bus_timeout=bool(status & 0x40),
        )

    def describe(self) -> str:
        if self.ok:
            return "OK"
        flags = []
        if self.invalid_command: flags.append("invalid_command")
        if self.invalid_param_value: flags.append("invalid_param_value")
        if self.invalid_param_count: flags.append("invalid_param_count")
        if self.read_command_error: flags.append("read_command_error")
        if self.command_processing_error: flags.append("command_processing_error")
        if self.flash_batch_error: flags.append("flash_batch_error")
        if self.bus_timeout: flags.append("bus_timeout")
        return f"rejected op=0x{self.rejected_opcode:02X} flags=[{','.join(flags)}]"


@dataclass
class SystemStatus:
    """Decoded D1h 4 bytes. See pp. 73–74."""
    raw: Tuple[int, int, int, int]
    light_control_error_code: int  # byte 1 b(7:3): 0=OK, 1=illum_time, 2=pre_dark, 3=post_dark, 4=trig_out_1_delay, 5=trig_out_2_delay
    dmd_device_error: bool         # byte 1 b(2)
    dmd_interface_error: bool      # byte 1 b(1)
    sequence_abort: bool           # byte 1 b(0)
    red_led_enabled: bool          # byte 2 b(4)
    green_led_enabled: bool        # byte 2 b(5)
    blue_led_enabled: bool         # byte 2 b(6)
    watchdog_timeout: bool         # byte 3 b(5)
    product_config_error: bool     # byte 3 b(3)

    LIGHT_CTRL_ERR_NAMES = {
        0: "OK",
        1: "illumination_time_not_supported",
        2: "pre_illumination_dark_time_not_supported",
        3: "post_illumination_dark_time_not_supported",
        4: "trig_out_1_delay_not_supported",
        5: "trig_out_2_delay_not_supported",
    }

    @classmethod
    def decode(cls, resp: Sequence[int]) -> "SystemStatus":
        if len(resp) < 4:
            raise DLPCError(f"0xD1 response too short: {len(resp)} bytes")
        b0, b1, b2, b3 = resp[0], resp[1], resp[2], resp[3]
        return cls(
            raw=(b0, b1, b2, b3),
            light_control_error_code=(b1 >> 3) & 0x1F,
            dmd_device_error=bool(b1 & 0x04),
            dmd_interface_error=bool(b1 & 0x02),
            sequence_abort=bool(b1 & 0x01),
            red_led_enabled=bool(b2 & 0x10),
            green_led_enabled=bool(b2 & 0x20),
            blue_led_enabled=bool(b2 & 0x40),
            watchdog_timeout=bool(b3 & 0x20),
            product_config_error=bool(b3 & 0x08),
        )

    def describe(self) -> str:
        lc = self.LIGHT_CTRL_ERR_NAMES.get(self.light_control_error_code,
                                           f"unknown({self.light_control_error_code})")
        leds = [name for name, on in
                (("R", self.red_led_enabled), ("G", self.green_led_enabled), ("B", self.blue_led_enabled)) if on]
        problems = []
        if self.dmd_device_error: problems.append("dmd_device_error")
        if self.dmd_interface_error: problems.append("dmd_interface_error")
        if self.sequence_abort: problems.append("sequence_abort")
        if self.watchdog_timeout: problems.append("watchdog_timeout")
        if self.product_config_error: problems.append("product_config_error")
        parts = [f"lc={lc}", f"leds={''.join(leds) or '-'}"]
        if problems:
            parts.append(f"problems=[{','.join(problems)}]")
        return " ".join(parts)


def read_short_status(bus: int, addr: int = ADDR_DEFAULT) -> ShortStatus:
    resp = raw_read(bus, addr, OP_SHORT_STATUS_R, (), 1)
    if not resp:
        raise DLPCError("0xD0 returned no data")
    return ShortStatus.decode(resp[0])


def read_system_status(bus: int, addr: int = ADDR_DEFAULT) -> SystemStatus:
    resp = raw_read(bus, addr, OP_SYSTEM_STATUS_R, (), 4)
    return SystemStatus.decode(resp)


def read_comm_status(bus: int, addr: int = ADDR_DEFAULT, bus_selector: int = 0x02) -> CommStatus:
    """bus_selector: 0x01=USB/DebugPort, 0x02=I²C. Default I²C per p. 76."""
    resp = raw_read(bus, addr, OP_COMM_STATUS_R, (bus_selector,), 6)
    return CommStatus.decode(resp)


def read_controller_id(bus: int, addr: int = ADDR_DEFAULT) -> int:
    resp = raw_read(bus, addr, OP_CONTROLLER_ID_R, (), 1)
    return resp[0] if resp else 0


def read_dmd_id(bus: int, addr: int = ADDR_DEFAULT, sub: int = 0x00) -> List[int]:
    return raw_read(bus, addr, OP_DMD_ID_R, (sub,), 4)


# ---------- Init wait ----------


def wait_init_done(
    bus: int,
    addr: int = ADDR_DEFAULT,
    timeout_s: float = 3.0,
    poll_interval_s: float = 0.05,
) -> ShortStatus:
    """Poll 0xD0 until b(0) System Initialization Complete = 1.

    Datasheet p. 5: do not issue I²C before HOST_IRQ goes low; doing so
    can prevent the system from booting. Since we don't have a HOST_IRQ
    GPIO exposed, we poll 0xD0 and trust that the first successful read
    implies HOST_IRQ has dropped. See p. 72 note 7: do not poll
    continuously — this function sleeps between polls.
    """
    deadline = time.monotonic() + timeout_s
    last_exc: Optional[Exception] = None
    while time.monotonic() < deadline:
        try:
            ss = read_short_status(bus, addr)
            if ss.init_complete:
                return ss
        except Exception as exc:  # no ack yet → bus NACK → still booting
            last_exc = exc
        time.sleep(poll_interval_s)
    detail = f" (last error: {last_exc})" if last_exc else ""
    raise DLPCTimeout(f"DLPC init did not complete within {timeout_s}s{detail}")


# ---------- Checked write ----------


def write_with_check(
    bus: int,
    addr: int,
    opcode: int,
    data: Sequence[int] = (),
    *,
    raise_on_error: bool = True,
) -> CommStatus:
    """Write opcode + data, then read 0xD3 and raise if any error bit is set.

    Returns the decoded CommStatus either way; callers who want to
    tolerate failures can pass raise_on_error=False and inspect.ok.
    """
    raw_write(bus, addr, opcode, data)
    status = read_comm_status(bus, addr)
    if not status.ok and raise_on_error:
        raise DLPCRejected(
            f"DLPC rejected 0x{opcode:02X}: {status.describe()}",
            status_byte=status.raw_status,
            rejected_opcode=status.rejected_opcode,
        )
    return status


# ---------- Payload builders ----------


def _u32_le(value: int) -> List[int]:
    if value < 0 or value > 0xFFFFFFFF:
        raise ValueError(f"u32 out of range: {value}")
    return [value & 0xFF, (value >> 8) & 0xFF, (value >> 16) & 0xFF, (value >> 24) & 0xFF]


def _s32_le(value: int) -> List[int]:
    """Little-endian 32-bit signed (for trig out 2 delay)."""
    if value < -0x80000000 or value > 0x7FFFFFFF:
        raise ValueError(f"s32 out of range: {value}")
    if value < 0:
        value = (1 << 32) + value
    return _u32_le(value)


def _u16_pair(value: int) -> List[int]:
    """LSB, MSB."""
    if value < 0 or value > 0xFFFF:
        raise ValueError(f"u16 out of range: {value}")
    return [value & 0xFF, (value >> 8) & 0xFF]


def pattern_config_payload(
    *,
    seq_type: int = SEQ_TYPE_1BIT_MONO,
    num_patterns: int = 1,
    illum_select: int = ILLUM_RED,
    illum_us: int = 16000,
    pre_dark_us: int = 0,
    post_dark_us: int = 0,
) -> List[int]:
    """Build the 15-byte 0x96 Pattern Configuration payload (p. 61)."""
    if not (0 <= seq_type <= 3):
        raise ValueError(f"seq_type out of range 0-3: {seq_type}")
    if not (1 <= num_patterns <= 128):
        raise ValueError(f"num_patterns out of range 1-128: {num_patterns}")
    if illum_select & ~0x07:
        raise ValueError(f"illum_select must be bitmask of RGB bits: 0x{illum_select:02X}")
    return [seq_type, num_patterns, illum_select] + _u32_le(illum_us) + _u32_le(pre_dark_us) + _u32_le(post_dark_us)


def trigger_out_payload(
    *,
    select: int = TRIG_OUT_2,
    enable: bool = True,
    inversion: bool = False,
    delay_us: int = 0,
) -> List[int]:
    """Build the 5-byte 0x92 Trigger Out Configuration payload (p. 57).

    For TRIG_OUT_2, delay may be negative (signed pre-trigger).
    """
    if select not in (TRIG_OUT_1, TRIG_OUT_2):
        raise ValueError(f"select must be 0 (OUT1) or 1 (OUT2), got {select}")
    cfg = (select & 0x01) | ((1 if enable else 0) << 1) | ((1 if inversion else 0) << 2)
    return [cfg] + _s32_le(delay_us)


def led_pwm_payload(r: int, g: int, b: int) -> List[int]:
    """Build the 6-byte 0x54 RGB LED Current PWM payload (p. 44).

    Each value is 10-bit (0–1023) PWM. MSB, LSB order per datasheet is
    little-endian 16-bit per color: R_LSB R_MSB G_LSB G_MSB B_LSB B_MSB.
    """
    for name, v in (("r", r), ("g", g), ("b", b)):
        if not (0 <= v <= 0x03FF):
            raise ValueError(f"{name}_pwm out of 10-bit range (0-1023): {v}")
    return _u16_pair(r) + _u16_pair(g) + _u16_pair(b)


def display_size_payload(width: int, height: int) -> List[int]:
    """Build the 4-byte 0x12 Display Size payload (p. 23)."""
    return _u16_pair(width) + _u16_pair(height)


def input_size_payload(width: int, height: int) -> List[int]:
    """Build the 4-byte 0x2E Input Image Size payload (p. 37)."""
    return _u16_pair(width) + _u16_pair(height)


def pattern_order_table_entry_payload(
    *,
    index: int,
    illum_select: int,
    illum_us: int = 16000,
) -> List[int]:
    """Build one 0x98 Pattern Order Table Entry (p. 63).

    Dark times use the values from 0x96 (flags = 0x00).
    """
    if not (0 <= index <= 127):
        raise ValueError(f"pattern index out of range 0-127: {index}")
    if illum_select & ~0x07:
        raise ValueError(f"illum_select must be bitmask of RGB bits: 0x{illum_select:02X}")
    return [index, illum_select] + _u32_le(illum_us) + [0x00, 0x00]


# ---------- Validate exposure (0x9D) ----------


@dataclass
class ExposureValidation:
    supported: bool
    min_pre_dark_us: int
    min_post_dark_us: int
    max_pre_dark_us: int
    max_post_dark_us: int

    @classmethod
    def decode(cls, resp: Sequence[int]) -> "ExposureValidation":
        # 13 bytes out. Byte 1 b(0) = supported. Bytes 2–5 min pre, 6–9 min post,
        # 10–13 max pre — per p. 67; if b(0)=0 the other bytes are junk.
        if len(resp) < 13:
            raise DLPCError(f"0x9D response too short: {len(resp)} bytes")
        supported = bool(resp[0] & 0x01)
        if not supported:
            return cls(False, 0, 0, 0, 0)
        min_pre = resp[1] | (resp[2] << 8) | (resp[3] << 16) | (resp[4] << 24)
        min_post = resp[5] | (resp[6] << 8) | (resp[7] << 16) | (resp[8] << 24)
        max_pre = resp[9] | (resp[10] << 8) | (resp[11] << 16) | (resp[12] << 24)
        return cls(True, min_pre, min_post, max_pre, 0)


def validate_exposure(
    bus: int,
    addr: int,
    *,
    pattern_mode: int = MODE_LIGHT_EXT_STREAM,
    bit_depth: int = 1,
    illum_us: int = 16000,
) -> ExposureValidation:
    """Call 0x9D Validate Exposure Time. Returns whether the combo is supported."""
    # Input: 6 bytes — pattern mode, bit depth, illum_us (4 bytes LE). Output: 13 bytes.
    data = [pattern_mode, bit_depth] + _u32_le(illum_us)
    resp = raw_read(bus, addr, OP_VALIDATE_EXPOSURE_R, data, 13)
    return ExposureValidation.decode(resp)


# ---------- Boot transcript (matches DMD_RED_BLUE_WORKFLOW.md §6) ----------


def boot_external_pattern_streaming(
    bus: int,
    addr: int = ADDR_DEFAULT,
    *,
    width: int = 1920,
    height: int = 1080,
    r_pwm: int | None = None,
    g_pwm: int = 0x0000,
    b_pwm: int | None = None,
    max_pwm: int = 0x03FF,
    initial_illum: int = ILLUM_RED,
    illum_us: int = 11000,
    pre_dark_us: int = 2200,
    post_dark_us: int = 5000,
    seq_type: int = SEQ_TYPE_8BIT_RGB,
    trig_out_select: int = TRIG_OUT_2,
    trig_out_delay_us: int = 0,
    trig_out_enable: bool = True,
    validate: bool = True,
    verbose: bool = True,
    rgb_cycle_mode: bool = False,  # R-B3: Mode B preset (Simultaneous RGB)
) -> None:
    """Bring the DLPC into Mode 03h External Pattern Streaming.

    Mirrors the proven 4-command sequence from the original
    i2c_test_send_commands.py (which the lab confirmed worked for months
    of stim experiments) — 0x92 → 0x96 → 0x54 → 0x05. We add 0xD3
    read-back via write_with_check on each one so silent failures get
    surfaced as DLPCRejected exceptions.

    Defaults match the proven values:
      - 0x96 timing: 11 ms illum / 2.2 ms pre-dark / 5 ms post-dark
        (the DLPC needs non-zero dark times — 0/0 may be rejected)
      - 0x96 sequence type: 8-bit RGB (matches the working byte 1 = 0x03)
      - 0x92: Trigger Out 2 enabled, delay = 0
      - 0x54: LED PWM auto-derived from `initial_illum` (full PWM on the
        chosen color, 0 on others) unless r_pwm / b_pwm are passed explicitly

    The "extra" datasheet-recommended commands (curtain, freeze, video
    format, display size, input size, max PWM ceiling, LED ctrl method)
    are deliberately omitted — they were never needed by the working
    sequence and any one of them being rejected would abort the boot.
    """
    def say(msg: str) -> None:
        if verbose:
            print(f"[DLPC] {msg}")

    # R-B3: Mode B — Simultaneous RGB sub-frame mode
    # When rgb_cycle_mode=True, configure for the DMD's 8-bit RGB sub-frame
    # engine: stim mask in R channel + observe mask in B channel of ONE HDMI
    # frame. DMD decomposes into sub-frames automatically at 1440 Hz bit-plane
    # rate. See memory/project_stim_observe_three_modes_20260420.md.
    # Forces illum=0x05 (R+B gated, G off), seq_type=0x03 (8-bit RGB),
    # full PWM on R and B.
    if rgb_cycle_mode:
        initial_illum = ILLUM_RED | ILLUM_BLUE  # 0x05
        seq_type = SEQ_TYPE_8BIT_RGB             # 0x03
        if r_pwm is None:
            r_pwm = max_pwm
        if b_pwm is None:
            b_pwm = max_pwm

    # LED PWM defaults reflect the initial_illum bitmask — only the chosen
    # color is driven initially. Live switching is handled by rewriting 0x54
    # (switch_led_color), not by rewriting 0x96. The 0x96 Pattern Config we
    # write in step [2/4] below gates ALL three LEDs on (illum_select = 0x07)
    # so subsequent 0x54 writes can light any color without a mode cycle.
    if r_pwm is None:
        r_pwm = 0x03FF if (initial_illum & ILLUM_RED) else 0x0000
    if b_pwm is None:
        b_pwm = 0x03FF if (initial_illum & ILLUM_BLUE) else 0x0000

    say(f"Waiting for init-done on bus={bus} addr=0x{addr:02X}...")
    ss = wait_init_done(bus, addr)
    say(f"init done, short_status=0x{ss.raw:02X}")

    ctrl_id = read_controller_id(bus, addr)
    if ctrl_id != 0x0C:
        say(f"WARNING: controller ID 0x{ctrl_id:02X} is not 0x0C (DLPC3479)")
    else:
        say("controller ID = 0x0C (DLPC3479) — OK")

    if validate:
        validate_bit_depth = 1 if seq_type in (0, 1) else 8
        ev = validate_exposure(bus, addr, bit_depth=validate_bit_depth, illum_us=illum_us)
        if not ev.supported:
            say(f"WARNING: 0x9D says illum_us={illum_us} not officially supported in "
                f"{validate_bit_depth}-bit mode. Proceeding — 0x96 write will be checked via 0xD3.")
        else:
            say(f"exposure {illum_us} µs validated; min_pre_dark={ev.min_pre_dark_us} µs "
                f"min_post_dark={ev.min_post_dark_us} µs")

    # ----- The proven 4-command boot sequence -----
    # Use raw_write (no per-command D3h check) to mirror the original working
    # code. Per-write D3h reads were producing false positives because D3h's
    # "last rejected opcode" register holds STALE values from prior sessions —
    # raising on those aborted boots that were actually succeeding. We do
    # ONE D3h read at the end as info-only.
    say(f"[1/4] 0x92 Trigger Out {trig_out_select+1} "
        f"enable={trig_out_enable} delay={trig_out_delay_us} µs")
    raw_write(
        bus, addr, OP_TRIG_OUT_CFG_W,
        trigger_out_payload(select=trig_out_select, enable=trig_out_enable,
                            delay_us=trig_out_delay_us),
    )

    # 0x96 byte 3 (Illumination Select) = caller-supplied initial_illum.
    # Earlier attempt at 0x07 (all LEDs gated so 0x54 could live-switch)
    # silently broke physical projection on the DLPC3479 — the DMD stayed
    # dark. Reverted to the proven single-color gating pattern. True live
    # color switching requires a Stop→Start mode cycle (handled at the
    # caller level — qt_interface._on_led_color_changed_live).
    illum_name = {ILLUM_RED: "RED", ILLUM_GREEN: "GREEN", ILLUM_BLUE: "BLUE"}.get(
        initial_illum, f"bitmask=0x{initial_illum:02X}")
    seq_name = {0: "1-bit mono", 1: "1-bit RGB", 2: "8-bit mono", 3: "8-bit RGB"}.get(
        seq_type, f"type=0x{seq_type:02X}")
    say(f"[2/4] 0x96 Pattern Config: {seq_name}, 1 pattern, {illum_name}, "
        f"illum={illum_us} µs pre={pre_dark_us} µs post={post_dark_us} µs")
    raw_write(
        bus, addr, OP_PATTERN_CONFIG_W,
        pattern_config_payload(
            seq_type=seq_type,
            num_patterns=1,
            illum_select=initial_illum,
            illum_us=illum_us,
            pre_dark_us=pre_dark_us,
            post_dark_us=post_dark_us,
        ),
    )

    say(f"[3/4] 0x54 LED Current PWM: R=0x{r_pwm:03X} G=0x{g_pwm:03X} B=0x{b_pwm:03X}")
    raw_write(bus, addr, OP_LED_CURRENT_PWM_W, led_pwm_payload(r_pwm, g_pwm, b_pwm))

    say("[4/4] 0x05 Operating Mode = 0x03 (Light Control – External Pattern Streaming)")
    raw_write(bus, addr, OP_OP_MODE_W, [MODE_LIGHT_EXT_STREAM])

    # Single post-boot diagnostic — log only, never aborts.
    try:
        comm = read_comm_status(bus, addr)
        if comm.ok:
            say("0xD3 post-boot status: OK (no error flags set)")
        else:
            say(f"0xD3 post-boot status: {comm.describe()} "
                f"(may be stale from prior session — physical DMD state is the truth)")
        sys_status = read_system_status(bus, addr)
        say(f"0xD1 post-boot system_status: {sys_status.describe()}")
    except Exception as exc:
        say(f"(post-boot diagnostic read failed — non-fatal: {exc})")

    say("boot sequence complete — DMD streaming from HDMI in mode 03h")


def boot_internal_pattern_streaming(
    bus: int,
    addr: int = ADDR_DEFAULT,
    *,
    width: int = 1920,
    height: int = 1080,
    patterns: Optional[List[dict]] = None,
    pre_dark_us: int = 2200,
    post_dark_us: int = 5000,
    r_pwm: int = 0x03FF,
    g_pwm: int = 0x0000,
    b_pwm: int = 0x03FF,
    max_pwm: int = 0x03FF,
    trig_out_select: int = TRIG_OUT_2,
    trig_out_delay_us: int = 0,
    trig_out_enable: bool = True,
    trig_out_per_pattern: bool = True,
    validate: bool = True,
    verbose: bool = True,
) -> None:
    """Bring the DLPC into Mode 04h Internal Pattern Streaming.

    Implements Mode A — temporal alternation with a multi-pattern sequence
    (default: RED stim then BLUE observe). The DMD cycles through the
    Pattern Order Table entries autonomously; no HDMI frames are needed.

    Boot sequence (DLPU081A programmer's guide):
      1. Wait for init done
      2. Read controller ID
      3. Optionally validate exposure (0x9D)
      4. Write 0x92 Trigger Out config
      5. Write 0x96 Pattern Config (num_patterns, seq_type, dark times)
      6. Write 0x98 Pattern Order Table — one entry per pattern
      7. Write 0x54 LED PWM — enable ALL colors that appear in any pattern
      8. Write 0x05 with MODE_LIGHT_INT_STREAM (0x04)
      9. Write 0x9E with [0x00, 0xFF] to start (infinite repeat)
     10. Post-boot diagnostic (info-only)
    """
    if patterns is None:
        patterns = [
            {"illum_select": ILLUM_RED, "illum_us": 16000},
            {"illum_select": ILLUM_BLUE, "illum_us": 16000},
        ]

    def say(msg: str) -> None:
        if verbose:
            print(f"[DLPC] {msg}")

    num_pat = len(patterns)
    if not (1 <= num_pat <= 128):
        raise ValueError(f"patterns list must have 1-128 entries, got {num_pat}")

    # Derive the combined illumination bitmask across all patterns
    combined_illum = 0
    for pat in patterns:
        combined_illum |= pat["illum_select"]

    # Use first pattern's illum_select for the 0x96 command (required field)
    first_illum = patterns[0]["illum_select"]

    say(f"Waiting for init-done on bus={bus} addr=0x{addr:02X}...")
    ss = wait_init_done(bus, addr)
    say(f"init done, short_status=0x{ss.raw:02X}")

    ctrl_id = read_controller_id(bus, addr)
    if ctrl_id != 0x0C:
        say(f"WARNING: controller ID 0x{ctrl_id:02X} is not 0x0C (DLPC3479)")
    else:
        say("controller ID = 0x0C (DLPC3479) — OK")

    if validate:
        # Validate against the first pattern's illumination time
        ev = validate_exposure(
            bus, addr,
            pattern_mode=MODE_LIGHT_INT_STREAM,
            bit_depth=1,
            illum_us=patterns[0]["illum_us"],
        )
        if not ev.supported:
            say(f"WARNING: 0x9D says illum_us={patterns[0]['illum_us']} not officially "
                f"supported in 1-bit mode for internal streaming. Proceeding anyway.")
        else:
            say(f"exposure {patterns[0]['illum_us']} µs validated; "
                f"min_pre_dark={ev.min_pre_dark_us} µs "
                f"min_post_dark={ev.min_post_dark_us} µs")

    # ----- Boot sequence — raw_write (no per-command D3h check) -----
    # Same rationale as boot_external_pattern_streaming: D3h stale values
    # from prior sessions cause false-positive aborts.

    say(f"[1/6] 0x92 Trigger Out {trig_out_select+1} "
        f"enable={trig_out_enable} delay={trig_out_delay_us} µs")
    raw_write(
        bus, addr, OP_TRIG_OUT_CFG_W,
        trigger_out_payload(
            select=trig_out_select, enable=trig_out_enable,
            delay_us=trig_out_delay_us,
        ),
    )

    illum_name = {ILLUM_RED: "RED", ILLUM_GREEN: "GREEN", ILLUM_BLUE: "BLUE"}.get(
        first_illum, f"bitmask=0x{first_illum:02X}")
    say(f"[2/6] 0x96 Pattern Config: 1-bit mono, {num_pat} pattern(s), "
        f"illum_select={illum_name}, pre={pre_dark_us} µs post={post_dark_us} µs")
    raw_write(
        bus, addr, OP_PATTERN_CONFIG_W,
        pattern_config_payload(
            seq_type=SEQ_TYPE_1BIT_MONO,
            num_patterns=num_pat,
            illum_select=first_illum,
            illum_us=patterns[0]["illum_us"],
            pre_dark_us=pre_dark_us,
            post_dark_us=post_dark_us,
        ),
    )

    say(f"[3/6] 0x98 Pattern Order Table — {num_pat} entries:")
    for i, pat in enumerate(patterns):
        illum_s = pat["illum_select"]
        illum_t = pat["illum_us"]
        color_str = {ILLUM_RED: "RED", ILLUM_GREEN: "GREEN", ILLUM_BLUE: "BLUE"}.get(
            illum_s, f"0x{illum_s:02X}")
        say(f"       [{i}] {color_str} illum={illum_t} µs")
        raw_write(
            bus, addr, OP_PATTERN_ORDER_TABLE_W,
            pattern_order_table_entry_payload(
                index=i, illum_select=illum_s, illum_us=illum_t,
            ),
        )

    # Enable LEDs for all colors that appear in any pattern entry.
    # Override caller PWM values: any color present in the table gets its
    # PWM value; colors not in the table get 0.
    eff_r = r_pwm if (combined_illum & ILLUM_RED) else 0
    eff_g = g_pwm if (combined_illum & ILLUM_GREEN) else 0
    eff_b = b_pwm if (combined_illum & ILLUM_BLUE) else 0
    say(f"[4/6] 0x54 LED Current PWM: R=0x{eff_r:03X} G=0x{eff_g:03X} B=0x{eff_b:03X}")
    raw_write(bus, addr, OP_LED_CURRENT_PWM_W, led_pwm_payload(eff_r, eff_g, eff_b))

    say("[5/6] 0x05 Operating Mode = 0x04 (Light Control – Internal Pattern Streaming)")
    raw_write(bus, addr, OP_OP_MODE_W, [MODE_LIGHT_INT_STREAM])

    say("[6/6] 0x9E Internal Pattern Control: start, infinite repeat")
    raw_write(bus, addr, OP_INT_PATTERN_CTRL_W, [0x00, 0xFF])

    # Single post-boot diagnostic — log only, never aborts.
    try:
        comm = read_comm_status(bus, addr)
        if comm.ok:
            say("0xD3 post-boot status: OK (no error flags set)")
        else:
            say(f"0xD3 post-boot status: {comm.describe()} "
                f"(may be stale from prior session — physical DMD state is the truth)")
        sys_status = read_system_status(bus, addr)
        say(f"0xD1 post-boot system_status: {sys_status.describe()}")
    except Exception as exc:
        say(f"(post-boot diagnostic read failed — non-fatal: {exc})")

    say("boot sequence complete — DMD internal pattern streaming in mode 04h")


def set_illumination_for_next_frame(
    bus: int,
    addr: int,
    illum_select: int,
    illum_us: int = 16000,
) -> None:
    """Re-issue 0x96 with a new illumination select. Call ~200 µs after vsync.

    NOTE: 0x96 is a *source-associated* command (datasheet p. 9) — it only
    applies when the External Video source is (re)selected via 0x05.
    Writing while already in mode 03h just stores the value; it does NOT
    re-latch on the next HDMI frame as the subagent's workflow doc implied.
    Use `switch_led_color()` (which writes 0x54 PWM) for live color
    switching. This helper is kept for completeness and offline
    reconfiguration flows.
    """
    raw_write(
        bus, addr, OP_PATTERN_CONFIG_W,
        pattern_config_payload(
            seq_type=SEQ_TYPE_1BIT_MONO,
            num_patterns=1,
            illum_select=illum_select,
            illum_us=illum_us,
        ),
    )


def switch_led_color(
    bus: int,
    addr: int,
    illum_select: int,
    *,
    pwm: int = 0x03FF,
) -> None:
    """Switch which LED is physically lit by rewriting 0x54 LED Current PWM.

    0x54 is NOT source-associated — it applies immediately. Combined with
    a boot-time 0x96 byte 3 = 0x07 (all three LEDs gated on), this lets
    us switch color live without cycling the operating mode.

    The LED that should light up gets `pwm` drive current; the others get 0.
    For combos (e.g. R+B), bits set in `illum_select` all get `pwm`.
    """
    r_pwm = pwm if (illum_select & ILLUM_RED) else 0
    g_pwm = pwm if (illum_select & ILLUM_GREEN) else 0
    b_pwm = pwm if (illum_select & ILLUM_BLUE) else 0
    raw_write(
        bus, addr, OP_LED_CURRENT_PWM_W,
        led_pwm_payload(r_pwm, g_pwm, b_pwm),
    )


def fast_phase_switch(
    bus: int,
    addr: int = ADDR_DEFAULT,
    color: str = 'red',
    *,
    illum_us: int = 11000,
    pre_dark_us: int = 2200,
    post_dark_us: int = 5000,
    pwm: int = 0x03FF,
) -> None:
    """Mode-A per-phase LED switch — minimal I²C overhead version of boot.

    Skips the boot script's init wait, controller-ID read, exposure validation,
    and post-write status read-backs. Just does the 4 essential writes:
      1. 0x05 0xFF  → Standby (kills LEDs, true-off; required because 0x96
                      changes only apply on next mode-select transition)
      2. 0x96...   → Pattern Config with new illum_select for this phase
      3. 0x54...   → LED PWM (only the chosen color non-zero)
      4. 0x05 0x03  → External Pattern Streaming (applies the new 0x96)

    Designed for the stim trial loop in MONO mode — caller invokes once
    per phase transition. Measured latency on Jetson Orin: ~20-40 ms per call
    (vs ~244 ms for the full boot script).

    Parameters
    ----------
    color : 'red' | 'blue' | 'standby' | 'green' | 'rb'
        'standby' just enters Mode 0xFF and returns (true LED-off).
        Others reconfigure 0x96 + 0x54 and re-enter Mode 0x03.
    illum_us : int
        Pattern illumination time per frame (default 11000 = 11 ms).
        Set to 16000 to give each frame the full 60-Hz HDMI period.
    pwm : int
        PWM for the chosen LED(s) when active. 0x03FF = full brightness.
    """
    if color == 'standby':
        # Enter Mode 0xFF (Standby) — true LED off, but TRIG_OUT also stops.
        # Use this only between trials, NOT for live phase switching, because
        # the camera HW trigger needs continuous TRIG_OUT pulses.
        raw_write(bus, addr, OP_OP_MODE_W, [MODE_STANDBY])
        return

    # Map color → bitmask + per-LED PWMs
    color_map = {
        'red':   (ILLUM_RED,                 pwm,  0,    0),
        'blue':  (ILLUM_BLUE,                0,    0,    pwm),
        'green': (ILLUM_GREEN,               0,    pwm,  0),
        'rb':    (ILLUM_RED | ILLUM_BLUE,    pwm,  0,    pwm),
    }
    if color not in color_map:
        raise ValueError(f"color must be one of {list(color_map.keys())}, got {color!r}")
    illum_select, r_pwm, g_pwm, b_pwm = color_map[color]

    # Bench-tested : skipping Standby keeps TRIG_OUT firing
    # continuously, which is critical for the camera HW-trigger ordering.
    # The 0x96 byte 3 illum_select change applies on the next 0x05 mode-select
    # transition — so we just rewrite 0x05 mode 0x03 again (a no-op transition
    # from the firmware's perspective if already in mode 0x03, but it does
    # apply the new pattern config). 4.7-5.1 ms measured per call.
    #
    # Sequence:
    #   1. 0x96  → new MONO pattern config with new illum_select
    #   2. 0x54  → new LED PWM (other colors zeroed)
    #   3. 0x05  → re-apply Mode 03h (External Pattern Streaming)
    raw_write(bus, addr, OP_PATTERN_CONFIG_W, pattern_config_payload(
        seq_type=SEQ_TYPE_8BIT_MONO,
        num_patterns=1,
        illum_select=illum_select,
        illum_us=illum_us,
        pre_dark_us=pre_dark_us,
        post_dark_us=post_dark_us,
    ))
    raw_write(bus, addr, OP_LED_CURRENT_PWM_W, led_pwm_payload(r_pwm, g_pwm, b_pwm))
    raw_write(bus, addr, OP_OP_MODE_W, [MODE_LIGHT_EXT_STREAM])


def shutdown_to_standby(bus: int, addr: int = ADDR_DEFAULT, verbose: bool = True) -> None:
    """Issue 0x05 0xFF to move the DLPC to Standby (safe shutter state)."""
    if verbose:
        print("[DLPC] entering Standby (mode 0xFF) — LEDs off, DMD life-preserve")
    write_with_check(bus, addr, OP_OP_MODE_W, [MODE_STANDBY])
