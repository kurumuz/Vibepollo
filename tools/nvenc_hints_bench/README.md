# NVENC motion-hint benchmark

Synthetic game-like scenes with exact per-pixel motion, encoded by NVENC with
and without external motion-estimation hints derived from that motion the
way the host derives them from a game's DLSS motion vectors; per-region PSNR
from the encoder's reconstruction. Standalone (not part of the CMake build);
build in MSYS2 UCRT64 on the Windows host:

    g++ -std=c++17 -O2 -static -I../../third-party/nv-codec-headers/include/ffnvcodec main.cpp -o nvenc_hints.exe -ld3d11 -ldxgi -ldxguid

`nvenc_hints.exe --help` lists the options; `--selftest` checks the motion
model and the hint ABI without a GPU. Scenes (`--scene`, motion scale
`--speeds` in screen px/frame):

- `pan`, `orbit`, `parallax`, `objects`, `foliage`, `game` (parallax, roll,
  sprites and a HUD).
- `topdown`: a bird's-eye racer; the camera follows the player (held at the
  centre) along a weaving road with dashed lane lines (repetitive along the
  motion) and grass, clouds nearer the camera (1.7x parallax), other cars
  drifting relative to it.
- `chase`: a chase camera on a road in perspective, the ground streaming
  toward the camera (speed = the bottom row's px/frame), dashed lanes, posts,
  fog toward the horizon, gentle curves (yaw), cars ahead.

Hints (`--hints`): `off`; `on` (what game vectors know: surface motion, also
under the HUD); `verified` (`on`, each kept only if it beats zero motion on
the frames being encoded: what the host does); `oracle` (also static HUD and
disocclusion); `none`/`zero`/`empty`/`wrong` for checks.

Production-like settings: `--codec av1 --mode cbr --bitrate 40000000 --fps
120 --preset 4 --multipass off --split forced --sb-cu 16`.

Findings so far: see the memory note on motion-vector hints (2026-09-30):
hints pay off only where NVENC's own search fails (fine/repetitive texture
without coarse structure, fog, small fast objects at 16x16 CU hints).
