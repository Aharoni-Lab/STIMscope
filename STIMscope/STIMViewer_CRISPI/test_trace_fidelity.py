#!/usr/bin/env python3
"""TF3: Synthetic known-truth trace validation harness.

Creates synthetic frames with known per-ROI intensities, runs them
through TraceExtractor (with and without neuropil subtraction and
dF/F₀), and validates the output against ground truth.

Run:
    python3 test_trace_fidelity.py          # all tests
    python3 test_trace_fidelity.py -v       # verbose
    python3 test_trace_fidelity.py -k dff   # run only dff tests
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from trace_extractor import TraceExtractor, build_neuropil_labels


def _make_labels(h: int = 64, w: int = 64, n_rois: int = 4) -> np.ndarray:
    """Create a simple label map with n_rois square ROIs."""
    labels = np.zeros((h, w), dtype=np.int32)
    sz = min(h, w) // (n_rois + 1)
    for i in range(n_rois):
        y0 = sz // 2 + i * (sz + 2)
        x0 = sz // 2
        if y0 + sz > h:
            break
        labels[y0 : y0 + sz, x0 : x0 + sz] = i + 1
    return labels


def _make_frame(labels: np.ndarray, roi_values: dict) -> np.ndarray:
    """Create a grayscale frame with specific values per ROI."""
    frame = np.zeros(labels.shape, dtype=np.float32)
    for rid, val in roi_values.items():
        frame[labels == rid] = val
    return frame


class TestRawExtraction(unittest.TestCase):
    def setUp(self):
        self.labels = _make_labels(64, 64, 4)
        self.ids = [1, 2, 3, 4]
        self.ext = TraceExtractor(self.labels, self.ids, prefer_gpu=False)

    def tearDown(self):
        self.ext.close()

    def test_uniform_frame(self):
        frame = np.full(self.labels.shape, 128.0, dtype=np.float32)
        means = self.ext.extract(frame)
        np.testing.assert_allclose(means, 128.0, atol=0.01)

    def test_per_roi_values(self):
        vals = {1: 50.0, 2: 100.0, 3: 150.0, 4: 200.0}
        frame = _make_frame(self.labels, vals)
        means = self.ext.extract(frame)
        for i, rid in enumerate(self.ids):
            self.assertAlmostEqual(means[i], vals[rid], places=1,
                                   msg=f"ROI {rid}: expected {vals[rid]}, got {means[i]}")

    def test_zero_background(self):
        vals = {1: 100.0}
        frame = _make_frame(self.labels, vals)
        means = self.ext.extract(frame)
        self.assertAlmostEqual(means[0], 100.0, places=1)
        for i in range(1, len(self.ids)):
            self.assertAlmostEqual(means[i], 0.0, places=1)

    def test_single_pixel_roi(self):
        labels = np.zeros((10, 10), dtype=np.int32)
        labels[5, 5] = 1
        ext = TraceExtractor(labels, [1], prefer_gpu=False)
        frame = np.zeros((10, 10), dtype=np.float32)
        frame[5, 5] = 42.0
        means = ext.extract(frame)
        self.assertAlmostEqual(means[0], 42.0, places=1)
        ext.close()

    def test_n_rois_property(self):
        self.assertEqual(self.ext.n_rois, 4)

    def test_backend_is_numpy(self):
        self.assertEqual(self.ext.backend, "numpy")


class TestNeuropilRings(unittest.TestCase):
    def setUp(self):
        self.labels = _make_labels(128, 128, 3)
        self.ids = [1, 2, 3]

    def test_ring_excludes_roi(self):
        npil = build_neuropil_labels(self.labels, self.ids, inner_gap=1, ring_width=5)
        for rid in self.ids:
            roi_mask = self.labels == rid
            overlap = npil[roi_mask]
            self.assertTrue(np.all(overlap == 0),
                            f"Neuropil ring overlaps ROI {rid}")

    def test_ring_has_pixels(self):
        npil = build_neuropil_labels(self.labels, self.ids, inner_gap=1, ring_width=5)
        for rid in self.ids:
            n_pixels = np.sum(npil == rid)
            self.assertGreater(n_pixels, 0,
                               f"Neuropil ring for ROI {rid} has no pixels")

    def test_subtraction_reduces_neuropil_bleed(self):
        r = 0.7
        ext = TraceExtractor(
            self.labels, self.ids, prefer_gpu=False,
            neuropil_r=r, neuropil_inner_gap=1, neuropil_ring_width=5,
        )
        ext_no = TraceExtractor(self.labels, self.ids, prefer_gpu=False)
        frame = np.full(self.labels.shape, 100.0, dtype=np.float32)
        frame[self.labels == 1] = 200.0
        means_sub = ext.extract(frame)
        means_raw = ext_no.extract(frame)
        self.assertAlmostEqual(means_raw[0], 200.0, places=1)
        self.assertGreater(means_sub[0], means_raw[0] - r * 200,
                           "Subtraction removed too much signal")
        ext.close()
        ext_no.close()


class TestDeltaFOverF(unittest.TestCase):
    def setUp(self):
        self.labels = _make_labels(32, 32, 2)
        self.ids = [1, 2]
        self.ext = TraceExtractor(self.labels, self.ids, prefer_gpu=False)

    def tearDown(self):
        self.ext.close()

    def test_dff_flat_baseline(self):
        baseline = np.full((20, 2), 100.0, dtype=np.float32)
        frame = _make_frame(self.labels, {1: 120.0, 2: 100.0})
        dff = self.ext.extract_dff(frame, baseline, percentile=20.0)
        np.testing.assert_allclose(dff[0], 0.2, atol=0.02,
                                   err_msg="ROI 1 dF/F should be ~0.2")
        np.testing.assert_allclose(dff[1], 0.0, atol=0.02,
                                   err_msg="ROI 2 dF/F should be ~0.0")

    def test_dff_empty_baseline(self):
        baseline = np.array([], dtype=np.float32).reshape(0, 2)
        frame = _make_frame(self.labels, {1: 120.0, 2: 100.0})
        dff = self.ext.extract_dff(frame, baseline, percentile=20.0)
        np.testing.assert_allclose(dff, 0.0, atol=0.01,
                                   err_msg="Empty baseline should return zeros")

    def test_dff_negative_transient(self):
        baseline = np.full((20, 2), 100.0, dtype=np.float32)
        frame = _make_frame(self.labels, {1: 80.0, 2: 100.0})
        dff = self.ext.extract_dff(frame, baseline, percentile=20.0)
        self.assertLess(dff[0], 0.0, "Negative dF/F for below-baseline")

    def test_dff_spike_detection(self):
        n_frames = 100
        baseline_vals = np.full((n_frames, 2), 100.0, dtype=np.float32)
        baseline_vals[:, 0] += np.random.normal(0, 2, n_frames)
        baseline_vals[:, 1] += np.random.normal(0, 2, n_frames)
        spike_val = 250.0
        frame = _make_frame(self.labels, {1: spike_val, 2: 100.0})
        dff = self.ext.extract_dff(frame, baseline_vals, percentile=20.0)
        self.assertGreater(dff[0], 1.0,
                           f"Spike dF/F should be >1.0, got {dff[0]:.3f}")
        self.assertAlmostEqual(dff[1], 0.0, delta=0.1,
                               msg="Non-spiking ROI should be near 0")


class TestSyntheticTimeSeries(unittest.TestCase):
    """Validate extraction over a time series of synthetic frames."""

    def test_known_calcium_transient(self):
        labels = _make_labels(32, 32, 2)
        ext = TraceExtractor(labels, [1, 2], prefer_gpu=False)
        n_frames = 60
        fps = 30.0
        tau_rise = 0.05
        tau_decay = 0.5
        t = np.arange(n_frames) / fps
        spike_time = 0.5
        transient = np.zeros(n_frames)
        mask = t >= spike_time
        dt = t[mask] - spike_time
        transient[mask] = (1 - np.exp(-dt / tau_rise)) * np.exp(-dt / tau_decay)
        transient *= 100.0
        baseline = 100.0
        raw_traces = np.zeros((n_frames, 2), dtype=np.float32)
        for i in range(n_frames):
            vals = {1: baseline + transient[i], 2: baseline}
            frame = _make_frame(labels, vals)
            means = ext.extract(frame)
            raw_traces[i] = means
        peak_idx = np.argmax(raw_traces[:, 0])
        self.assertGreater(raw_traces[peak_idx, 0], baseline + 20,
                           "Should detect transient peak")
        self.assertAlmostEqual(raw_traces[0, 0], baseline, delta=1.0,
                               msg="Pre-spike should be at baseline")
        np.testing.assert_allclose(raw_traces[:, 1], baseline, atol=1.0,
                                   err_msg="Non-spiking ROI should stay flat")
        ext.close()


if __name__ == "__main__":
    unittest.main()
