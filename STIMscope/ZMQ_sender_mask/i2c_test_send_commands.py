#!/usr/bin/env python3
"""DLPC3479 DMD bring-up / teardown CLI — datasheet-correct.

Subcommands:
  boot           Issue the full §6 boot transcript (init → mode 03h ext streaming).
  boot-internal  Boot into mode 04h Internal Pattern Streaming (Mode A).
  stop           Drive the controller to Standby (0x05 0xFF).
  status    Read D0/D1/D3/D4 and pretty-print.
  led-pwm   Write 0x54 with R/G/B PWM values (10-bit each).
  trig-out  Write 0x92 Trigger Out Configuration.
  pattern   Write 0x96 Pattern Configuration (red-only or blue-only per flag).
  validate  Read 0x9D Validate Exposure Time for a proposed timing.

Every write is followed by a 0x D3 Communication Status read; any
rejected opcode raises and prints the status byte.

See docs/hardware/I2C_COMMAND_REFERENCE.md and
docs/hardware/DMD_RED_BLUE_WORKFLOW.md for the paper trail.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dlpc_i2c  # noqa: E402
from i2c_send_custom_cmd import parse_int_token  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="DLPC3479 bring-up CLI. Every write is verified with 0xD3.",
    )
    p.add_argument("--bus", default="1", help="I²C bus number (default: 1)")
    p.add_argument("--addr", default="0x1B", help="7-bit I²C address (default: 0x1B)")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("boot", help="Issue the proven 4-command boot sequence "
                                    "(0x92 → 0x96 → 0x54 → 0x05)")
    b.add_argument("--width", type=int, default=1920)
    b.add_argument("--height", type=int, default=1080)
    b.add_argument("--r-pwm", default=None,
                   help="Red LED PWM 0-1023. Default: derived from --illum (0x03FF if RED bit set, 0 otherwise)")
    b.add_argument("--g-pwm", default="0x0000",
                   help="Green LED PWM (unused in our workflow; default 0)")
    b.add_argument("--b-pwm", default=None,
                   help="Blue LED PWM 0-1023. Default: derived from --illum")
    b.add_argument("--max-pwm", default="0x03FF", help="Max PWM ceiling")
    b.add_argument("--illum", default="red",
                   help="Initial illumination for first pattern. "
                        "Accepts color name (red|green|blue) or hex bitmask "
                        "where bit0=R bit1=G bit2=B (e.g. 0x01 0x04 0x05).")
    b.add_argument("--illum-us", type=int, default=11000,
                   help="Illumination time µs (default: 11000 — proven working value)")
    b.add_argument("--pre-dark-us", type=int, default=2200,
                   help="Pre-illumination dark time µs (default: 2200 — proven working value; "
                        "the DLPC may reject 0/0 dark times)")
    b.add_argument("--post-dark-us", type=int, default=5000,
                   help="Post-illumination dark time µs (default: 5000 — proven working value)")
    b.add_argument("--seq-type", type=int, default=3, choices=[0, 1, 2, 3],
                   help="Sequence type: 0=1-bit mono, 1=1-bit RGB, 2=8-bit mono, "
                        "3=8-bit RGB (default — matches the proven sequence byte 1 = 0x03)")
    b.add_argument("--trig-out", type=int, default=2, choices=[1, 2],
                   help="Trigger Out number (default: 2, supports signed pre-trigger)")
    b.add_argument("--trig-delay-us", type=int, default=0,
                   help="Trigger Out delay in µs (signed for OUT2)")
    b.add_argument("--rgb-cycle", action="store_true",
                   help="Mode B: simultaneous R+B sub-frame mode (sets illum=R+B, seq=8-bit RGB, full PWM)")
    b.add_argument("--no-validate", action="store_true",
                   help="Skip the 0x9D exposure validation pre-check")

    bi = sub.add_parser("boot-internal",
                        help="Boot into mode 04h Internal Pattern Streaming "
                             "(Mode A: temporal RED/BLUE alternation)")
    bi.add_argument("--width", type=int, default=1920)
    bi.add_argument("--height", type=int, default=1080)
    bi.add_argument("--stim-illum-us", type=int, default=16000,
                    help="Stim pattern illumination time µs (default: 16000)")
    bi.add_argument("--obs-illum-us", type=int, default=16000,
                    help="Observe pattern illumination time µs (default: 16000)")
    bi.add_argument("--stim-color", default="red",
                    help="Stim LED color (default: red). Accepts name or hex bitmask.")
    bi.add_argument("--obs-color", default="blue",
                    help="Observe LED color (default: blue). Accepts name or hex bitmask.")
    bi.add_argument("--pre-dark-us", type=int, default=2200,
                    help="Pre-illumination dark time µs (default: 2200)")
    bi.add_argument("--post-dark-us", type=int, default=5000,
                    help="Post-illumination dark time µs (default: 5000)")
    bi.add_argument("--trig-out", type=int, default=2, choices=[1, 2],
                    help="Trigger Out number (default: 2)")
    bi.add_argument("--trig-delay-us", type=int, default=0,
                    help="Trigger Out delay in µs (signed for OUT2)")
    bi.add_argument("--no-validate", action="store_true",
                    help="Skip the 0x9D exposure validation pre-check")

    sub.add_parser("stop", help="Issue 0x05 0xFF (Standby)")

    s = sub.add_parser("status", help="Read D0/D1/D3/D4 diagnostic status")
    s.add_argument("--full", action="store_true", help="Include raw register dumps")

    lp = sub.add_parser("led-pwm", help="Write 0x54 RGB LED Current PWM")
    lp.add_argument("--r", default="0x03FF")
    lp.add_argument("--g", default="0x0000")
    lp.add_argument("--b", default="0x03FF")

    to = sub.add_parser("trig-out", help="Write 0x92 Trigger Out Configuration")
    to.add_argument("--select", type=int, default=2, choices=[1, 2])
    to.add_argument("--disable", action="store_true")
    to.add_argument("--invert", action="store_true")
    to.add_argument("--delay-us", type=int, default=0)

    pt = sub.add_parser("pattern",
                        help="Write 0x96 Pattern Configuration (source-associated — "
                             "applies on next 0x05 mode transition, not live)")
    pt.add_argument("--illum", default="red",
                    help="Accepts color name (red|green|blue) or hex bitmask (e.g. 0x05)")
    pt.add_argument("--illum-us", type=int, default=16000)
    pt.add_argument("--pre-dark-us", type=int, default=0)
    pt.add_argument("--post-dark-us", type=int, default=0)

    sc = sub.add_parser("switch-color",
                        help="Live color switch via 0x54 LED PWM (applies immediately; "
                             "requires boot to have set 0x96 illum_select = 0x07)")
    sc.add_argument("--illum", default="red",
                    help="Which LED(s) to drive. Accepts color name or hex bitmask.")
    sc.add_argument("--pwm", default="0x03FF", help="PWM for each enabled color (0-1023).")

    v = sub.add_parser("validate", help="Read 0x9D Validate Exposure Time")
    v.add_argument("--illum-us", type=int, default=16000)
    v.add_argument("--bit-depth", type=int, default=1,
                   help="1 for 1-bit mono (binary masks), 8 for 8-bit")
    return p


def _illum_bits(value: str) -> int:
    """Accept 'red'|'green'|'blue' or a hex bitmask like '0x05'."""
    named = {"red": dlpc_i2c.ILLUM_RED,
             "green": dlpc_i2c.ILLUM_GREEN,
             "blue": dlpc_i2c.ILLUM_BLUE}
    lower = value.strip().lower()
    if lower in named:
        return named[lower]
    bits = parse_int_token(value, bits=8)
    if bits & ~0x07:
        raise ValueError(f"illum bitmask must use only bits 0-2 (R/G/B), got 0x{bits:02X}")
    if bits == 0:
        raise ValueError("illum bitmask must enable at least one color")
    return bits


def _hex(tok: str, bits: int = 16) -> int:
    return parse_int_token(tok, bits=bits)


def _cmd_boot(args, bus: int, addr: int) -> int:
    # r/b PWM: None means "derive from --illum" inside the helper
    r_pwm = _hex(args.r_pwm, 16) if args.r_pwm is not None else None
    b_pwm = _hex(args.b_pwm, 16) if args.b_pwm is not None else None
    dlpc_i2c.boot_external_pattern_streaming(
        bus, addr,
        width=args.width, height=args.height,
        r_pwm=r_pwm,
        g_pwm=_hex(args.g_pwm, 16),
        b_pwm=b_pwm,
        max_pwm=_hex(args.max_pwm, 16),
        initial_illum=_illum_bits(args.illum),
        illum_us=args.illum_us,
        pre_dark_us=args.pre_dark_us,
        post_dark_us=args.post_dark_us,
        seq_type=args.seq_type,
        trig_out_select=args.trig_out - 1,
        trig_out_delay_us=args.trig_delay_us,
        validate=not args.no_validate,
        rgb_cycle_mode=getattr(args, 'rgb_cycle', False),
    )
    return 0


def _cmd_boot_internal(args, bus: int, addr: int) -> int:
    patterns = [
        {"illum_select": _illum_bits(args.stim_color), "illum_us": args.stim_illum_us},
        {"illum_select": _illum_bits(args.obs_color), "illum_us": args.obs_illum_us},
    ]
    dlpc_i2c.boot_internal_pattern_streaming(
        bus, addr,
        width=args.width, height=args.height,
        patterns=patterns,
        pre_dark_us=args.pre_dark_us,
        post_dark_us=args.post_dark_us,
        trig_out_select=args.trig_out - 1,
        trig_out_delay_us=args.trig_delay_us,
        validate=not args.no_validate,
    )
    return 0


def _cmd_stop(args, bus: int, addr: int) -> int:
    dlpc_i2c.shutdown_to_standby(bus, addr)
    return 0


def _cmd_status(args, bus: int, addr: int) -> int:
    ss = dlpc_i2c.read_short_status(bus, addr)
    ctrl = dlpc_i2c.read_controller_id(bus, addr)
    sys_s = dlpc_i2c.read_system_status(bus, addr)
    comm = dlpc_i2c.read_comm_status(bus, addr)
    print(f"controller_id  = 0x{ctrl:02X} "
          f"({'DLPC3479' if ctrl == 0x0C else 'UNKNOWN'})")
    print(f"short_status   = 0x{ss.raw:02X}  "
          f"init_complete={ss.init_complete} comm_err={ss.comm_error} "
          f"sys_err={ss.system_error} lc_seq_err={ss.light_control_seq_error}")
    print(f"system_status  = {sys_s.describe()}")
    print(f"comm_status    = {comm.describe()}")
    if args.full:
        dmd_id = dlpc_i2c.read_dmd_id(bus, addr)
        print(f"dmd_id_bytes   = {' '.join(f'0x{b:02X}' for b in dmd_id)}")
    return 0


def _cmd_led_pwm(args, bus: int, addr: int) -> int:
    r, g, b = _hex(args.r, 16), _hex(args.g, 16), _hex(args.b, 16)
    print(f"[0x54] writing R=0x{r:03X} G=0x{g:03X} B=0x{b:03X}")
    dlpc_i2c.write_with_check(
        bus, addr, dlpc_i2c.OP_LED_CURRENT_PWM_W, dlpc_i2c.led_pwm_payload(r, g, b)
    )
    return 0


def _cmd_trig_out(args, bus: int, addr: int) -> int:
    payload = dlpc_i2c.trigger_out_payload(
        select=args.select - 1,
        enable=not args.disable,
        inversion=args.invert,
        delay_us=args.delay_us,
    )
    print(f"[0x92] writing OUT{args.select} "
          f"enable={not args.disable} invert={args.invert} "
          f"delay={args.delay_us} µs")
    dlpc_i2c.write_with_check(bus, addr, dlpc_i2c.OP_TRIG_OUT_CFG_W, payload)
    return 0


def _cmd_pattern(args, bus: int, addr: int) -> int:
    illum = _illum_bits(args.illum)
    payload = dlpc_i2c.pattern_config_payload(
        seq_type=dlpc_i2c.SEQ_TYPE_1BIT_MONO,
        num_patterns=1,
        illum_select=illum,
        illum_us=args.illum_us,
        pre_dark_us=args.pre_dark_us,
        post_dark_us=args.post_dark_us,
    )
    print(f"[0x96] pattern: illum={args.illum} illum_us={args.illum_us} "
          f"pre_dark={args.pre_dark_us} post_dark={args.post_dark_us}")
    dlpc_i2c.write_with_check(bus, addr, dlpc_i2c.OP_PATTERN_CONFIG_W, payload)
    return 0


def _cmd_validate(args, bus: int, addr: int) -> int:
    ev = dlpc_i2c.validate_exposure(
        bus, addr, bit_depth=args.bit_depth, illum_us=args.illum_us,
    )
    if not ev.supported:
        print(f"NOT SUPPORTED: illum_us={args.illum_us} bit_depth={args.bit_depth}")
        return 2
    print(f"supported: illum_us={args.illum_us} "
          f"min_pre_dark={ev.min_pre_dark_us} µs min_post_dark={ev.min_post_dark_us} µs")
    return 0


def _cmd_switch_color(args, bus: int, addr: int) -> int:
    illum = _illum_bits(args.illum)
    pwm = _hex(args.pwm, 16)
    print(f"[0x54 live] switch to illum=0x{illum:02X} pwm=0x{pwm:03X}")
    dlpc_i2c.switch_led_color(bus, addr, illum, pwm=pwm)
    return 0


_DISPATCH = {
    "boot": _cmd_boot,
    "boot-internal": _cmd_boot_internal,
    "stop": _cmd_stop,
    "status": _cmd_status,
    "led-pwm": _cmd_led_pwm,
    "trig-out": _cmd_trig_out,
    "pattern": _cmd_pattern,
    "switch-color": _cmd_switch_color,
    "validate": _cmd_validate,
}


def main(argv: List[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        bus = _hex(args.bus, 16)
        addr = _hex(args.addr, 8)
    except ValueError as exc:
        print(f"argument error: {exc}", file=sys.stderr)
        return 2

    try:
        return _DISPATCH[args.cmd](args, bus, addr)
    except dlpc_i2c.DLPCRejected as exc:
        print(f"REJECTED: {exc}", file=sys.stderr)
        return 1
    except dlpc_i2c.DLPCError as exc:
        print(f"DLPC error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
