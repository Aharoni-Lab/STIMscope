# Citation

If you use STIMscope in your research, please cite the platform plus
the upstream / hardware-vendor references it depends on.

The machine-readable version lives at
[`CITATION.cff`](https://github.com/Aharoni-Lab/STIMscope/blob/main/CITATION.cff)
in the repo root; GitHub renders a "Cite this repository" button
in the sidebar that exposes it as BibTeX, APA, etc.

## Platform

```bibtex
@software{STIMscope,
  title        = {STIMscope: Spatio-Temporal Illumination Microscope},
  author       = {Aharoni Lab},
  organization = {UCLA Department of Neurology},
  year         = {2026},
  url          = {https://github.com/Aharoni-Lab/STIMscope},
  license      = {GPL-3.0}
}
```

## Hardware + standards referenced

- **TI DLP4710** DMD with **DLPC3479** controller — wire-level protocol
  per the TI **DLPU081A** datasheet (see Texas Instruments product
  documentation).
- **IDS Peak SDK** — IDS USB3 industrial camera SDK
  (<https://en.ids-imaging.com/download-peak.html>); see also IDS
  Peak documentation for the GenICam node semantics surfaced in the
  GUI's Sensor Settings dialog.
- **GenICam** standard — for the camera trigger / node-map abstraction.

## Upstream code attribution

See the [`NOTICE`](https://github.com/Aharoni-Lab/STIMscope/blob/main/NOTICE)
file at the repo root for upstream attributions and any vendored
dependencies.

## License

GPL-3.0 — see
[`LICENSE`](https://github.com/Aharoni-Lab/STIMscope/blob/main/LICENSE).
