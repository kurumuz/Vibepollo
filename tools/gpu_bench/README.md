# GPU cost benchmarks

How much the streaming host's GPU work costs a GPU-bound game, piece by piece,
on the real machine. Standalone programs (not part of the CMake build); build
in MSYS2 UCRT64 on the Windows host:

    g++ -O2 -std=c++17 gpu_bench.cpp -o gpu_bench.exe -static -ld3d12 -ld3d11 -ldxgi -ld3dcompiler
    g++ -O2 -std=c++17 -I../../third-party/nv-codec-headers/include nvenc_bench.cpp -o nvenc_bench.exe -static -ld3d11 -ldxgi

`gpu_bench`:

- `caps`: the adapter's preemption granularity (RTX 4090 + HAGS: DMA buffer).
- `host-alone <shaders dir>`: the host's per-frame work timed alone, with its
  own shaders and dispatch/draw sequences (game frame -> capture image, the
  motion pass, the P010 conversion).
- `hook-alone`: the game hook's D3D12 copies (after DLSS, at Present).
- `game [--hook] <secs> <iter_dlss> <iter_main>`: a synthetic GPU-bound game
  (D3D12 compute, two frames in flight; `GAME_CHUNKS=180` splits its passes
  into short dispatches as a real game's), optionally with the hook's work in
  its frames; prints its frame GPU times and how many frames ran long.
- `host-load <shaders dir> <hz> <secs> [a b c] [nowait] [split]
  [cls=normal|above|high|realtime] [tp=none|7|abs20|abs30]`: the host's work
  at a rate, at a GPU scheduling class and context priority, to run beside
  `game` (from another shell); prints its submit -> done latency.

`nvenc_bench <hz> <secs> [1pass] [nosplit] [prio=high|realtime|normal]`: NVENC
as the host configures it (AV1 P4 ULL 10-bit, two-pass quarter, split-frame
forced), to run beside `game`.

Results on 2026-10-09 (RTX 4090, HAGS, synthetic game 11.33 ms a frame,
180 dispatches; host work 0.44 ms a frame alone):

| beside the game | game p50 | stalls > 5 ms / 30 s | host submit -> done p50/p99 |
|---|---|---|---|
| host work, REALTIME + absolute 30 (Sunshine's) | +0.77 ms | 7-12 (to 45 ms) | 1.0 / 3.3 ms |
| host work, HIGH + relative 7 (now the default) | +0.29 ms | 0 | 0.75 / 1.1 ms |
| NVENC (any mode) | +0.2 ms | 0 | encode 9.6 ms two-pass, 2.6 one-pass (noise) |
| all of it, new default | +0.43 ms | 0 | |
| hook (copies in the game's frame) | +0.05 ms | 0 | |
