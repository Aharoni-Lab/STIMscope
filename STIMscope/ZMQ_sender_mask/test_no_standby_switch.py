#!/usr/bin/env python3
"""DMD R/B per-phase color switch latency test — no-Standby variant.

Bench-tested  on Jetson Orin AGX → DLPC3479 + DLP4710.
Confirmed sub-5ms switch latency with NO Standby intermission, which
keeps TRIG_OUT firing continuously and avoids disrupting the IDS Peak
camera's HW-trigger ordering.

Use when:
- Sanity-checking the I²C path after hardware changes (cabling, power-cycle)
- Validating that the DLPC accepts MONO+R / MONO+B reconfig in Mode 0x03
- Measuring per-switch latency on a different host/clock

Pre-requisites:
- DLPC must be already booted into 8-bit MONO + RED:
    python3 i2c_test_send_commands.py boot --illum 0x01 --seq-type 2
- Click Project ON in the GUI (white HDMI content) so colors are visible
- Optical bench in line of sight to the DMD projection

Run:
    docker exec crispi-gui python3 /app/ZMQ_sender_mask/test_no_standby_switch.py

Expected output (~30 sec total):
    test no-standby switch
      switch 1 → red    4.8 ms
      switch 2 → blue   5.0 ms... (alternating, 6 switches @ 0.5s pause each)
    done

If colors don't visibly alternate, OR latencies > 20ms, OR errors fire,
the I²C path is broken and the per-phase production helper
`dlpc_i2c.fast_phase_switch` will fail in the same way.
"""
import sys
import time

sys.path.insert(0, "/app/ZMQ_sender_mask")
from dlpc_i2c import (
    raw_write, OP_OP_MODE_W, OP_PATTERN_CONFIG_W, OP_LED_CURRENT_PWM_W,
    pattern_config_payload, led_pwm_payload,
    SEQ_TYPE_8BIT_MONO, ILLUM_RED, ILLUM_BLUE, MODE_LIGHT_EXT_STREAM,
)

BUS, ADDR = 1, 0x1B


def switch_no_standby(color: str) -> None:
    """Reconfigure DMD to MONO+(red|blue) without going through Standby.

    The 0x96 byte 3 illum_select change applies on the next 0x05 mode select
    transition. Re-asserting Mode 0x03 (External Pattern Streaming) while
    already in Mode 0x03 applies the queued 0x96 + 0x54 changes without
    interrupting TRIG_OUT — critical for the camera HW trigger.
    """
    illum = ILLUM_RED if color == "red" else ILLUM_BLUE
    rp, bp = (0x3FF, 0) if color == "red" else (0, 0x3FF)
    raw_write(BUS, ADDR, OP_PATTERN_CONFIG_W, pattern_config_payload(
        seq_type=SEQ_TYPE_8BIT_MONO, num_patterns=1, illum_select=illum,
        illum_us=11000, pre_dark_us=2200, post_dark_us=5000,
    ))
    raw_write(BUS, ADDR, OP_LED_CURRENT_PWM_W, led_pwm_payload(rp, 0, bp))
    raw_write(BUS, ADDR, OP_OP_MODE_W, [MODE_LIGHT_EXT_STREAM])


def main() -> None:
    print("test no-standby switch")
    sequence = ["red", "blue", "red", "blue", "red", "blue"]
    for i, color in enumerate(sequence):
        t0 = time.monotonic()
        switch_no_standby(color)
        dt_ms = (time.monotonic() - t0) * 1000
        print(f"  switch {i+1} → {color:5s}  {dt_ms:.1f} ms")
        time.sleep(0.5)
    print("done")


if __name__ == "__main__":
    main()
