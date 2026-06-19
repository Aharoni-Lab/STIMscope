# Figures — sources and licensing

The figure files in this directory are reproduced for documentation purposes
in accordance with their original licenses. They are **not** redistributable
under this repository's GPL-3.0 software license; the licenses below apply
independently to each image asset.

## Figures (`fig01*`, `fig04*`)

Source: Chorsi H. T., Soldado-Magraner J., Jin S., Soltanalipouryekesammak I.,
Zheng A., Markovic B., Geschwind D. H., Golshani P., Buonomano D. V., Aharoni D.
(2026). *STIMscope: A high-resolution, low-cost, optogenetic stimulation
platform for closed-loop manipulation of neural activity at the centimeter
scale.* bioRxiv, posted May 28, 2026.
DOI: [10.64898/2026.05.27.728160](https://www.biorxiv.org/content/10.64898/2026.05.27.728160v1).

Reproduced under the bioRxiv license:
**Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International
(CC BY-NC-ND 4.0)** — <https://creativecommons.org/licenses/by-nc-nd/4.0/>.

Filenames map to panels as follows:

| File | Panel | Caption |
|---|---|---|
| `fig01a_platform_photo.png` | Fig 1a | Photo of the implemented STIMscope platform in the inverted configuration |
| `fig01b_hardware_architecture.png` | Fig 1b | Hardware architecture for synchronization, control and communication between the image sensor, DMD projector, microcontroller and NVIDIA Jetson Orin in real-time |
| `fig01c_optical_layout.png` | Fig 1c | Schematic of the optical layout and main components, showing integration of a small pixel CMOS sensor with a low magnification large aperture relay |
| `fig04a_software_architecture.png` | Fig 4a | CRISPI software architecture — Initialization, Calibration, Central Real-Time, Inference, Real-Time Trace Extraction, and Visualization Dashboard modules |
| `fig04b_calibrated_projection.jpg` | Fig 4b | Mask / Projection / Overlay triptych demonstrating calibrated projector→camera registration |
| `fig04ef_latency.png` | Fig 4e + 4f | Latency distributions — trigger-to-photodiode (e, mean = 26.3 ms) and closed-loop end-to-end (f, mean = 91.6 ms) |

## Upstream repository figure (`upstream_*`)

Source: <https://github.com/Aharoni-Lab/STIMscope/tree/main/Images>.
Reproduced from the upstream STIMscope repository (Aharoni Lab, UCLA).
Subject to the licensing of that repository (GPL-3.0 at the time of fetch);
attribution: Aharoni Lab, UCLA.

| File | Upstream filename |
|---|---|
| `upstream_stimscope_inverted.jpg` | `UCLA-STIMscope_closed_loop.jpg` |
