# Inference options

Every option here keeps the reference pipeline available: with none of them set, `archive` and `live` run full Hydra outputs and the dynamic GRU. Each option is explained with what it does and how it was measured. [BENCHMARK_RESULTS.md](../BENCHMARK_RESULTS.md) has the evidence.

| Status | Options |
|---|---|
| **Fastest configuration** (very slight output changes) | `--hydra-compile`, `--gru-compile`, plus everything below |
| **Exact, and selected** (byte-identical outputs) | `--detector-batch-size 1`, `--crop-batch-size 1`, `--hydra-output-mode minimal`, `--gru-execution graph`, `--gru-check-interval 1`, `--hydra-execution graph`, `--detect-ahead`, `--archive-processes`, `--decode-workers` (archives), `--gpu-crops`, `--cuda-blocking-sync`, `--hydra-fp16-weights`, `--gru-graph-first-step` |
| **Not selected, kept for comparison** | `--gru-streams`, `--crop-streams`, `--starseg-execution`, `--channels-last`, `--starseg-half`, `--hydra-fp32`, `--hydra-batch-size`, `--archive-frame-window`, `staged-archive` |

Capture is lazy: the first input of each new shape or walker count runs warmup and capture. Keep the strict output comparison and measure target-GPU latency, throughput and memory before selecting a configuration.

## Hydra and GRU execution

- `--hydra-output-mode minimal` skips the activated Hydra outputs that nothing uses. Archives skip them entirely, and live replay activates and transfers only the support and mouth maps for arm association. For 512-pixel float32 outputs this removes an 18 MiB tensor per crop in archive mode and replaces it with 2 MiB in live mode. Raw GRU inputs and tracking thresholds are unchanged. The Python API uses `LiveConfig(hydra_output_mode="minimal")`, and `predict_tensor_batch` still returns raw and full activated fields.
- `--gru-execution fixed --gru-check-interval N` keeps one slot per seed, masks completed walkers and checks whether all walkers have stopped every N steps (1–128). The reference decoder is `--gru-execution dynamic`, and the report records both settings. Fixed slots reduce dynamic compaction and host synchronization, but completed slots still run neural layers and can add work. The Python API uses `LiveConfig(gru_execution="fixed", gru_check_interval=8)`.
- `--gru-execution graph` runs the fixed-slot rollout from CUDA graphs captured per walker count. Walker counts are never padded. The step index moves to the device, and loop-history slots outside the populated prefix are masked with +inf before an exact minimum. It also samples each image's dense patches with one `grid_sample` call over that image, instead of expanding the 18-channel field per walker; with cuDNN's grid sampler, the expanded form copied the whole field once per walker on every step. Tip and base peak candidates and seed-embedding windows are extracted on the GPU, while score ordering, suppression and window means still run in NumPy on identical float32 values. `--gru-check-interval` sets steps per replay.
- `--gru-graph-first-step` (requires `--gru-execution graph`) also replays each rollout's first step, which applies the invalid-seed rule, from its own captured graph instead of launching it eagerly.
- `--hydra-execution graph` replays minimal-mode Hydra forwards at repeated input shapes.
- `--hydra-fp16-weights` holds Hydra's convolution and linear weights in float16. Under autocast PyTorch otherwise casts them from float32 on every forward, hundreds of small copies per crop. Outputs stay byte-identical, and Hydra's GPU time falls about 9%.
- `--hydra-compile` (requires `--hydra-execution graph`) compiles the Hydra forward with `torch.compile` before capturing it. Inductor fuses the normalization, activation and layer-scale arithmetic between convolutions, which about halves Hydra's time and lowers its peak memory. Arithmetic changes slightly, so outputs are not byte-identical. Use `--hydra-fp16-weights` with it.
- `--gru-compile` (requires `--gru-execution graph`) compiles the GRU step the same way. Inductor fuses the step's elementwise and indexing work into a few Triton kernels; convolutions and matrix products still use cuDNN and cuBLAS. This roughly halved per-step GPU time on an RTX A6000. Outputs are not byte-identical. On the 478-image validation split every image kept the same matches, every arm the same steps and stop reason, polylines moved at most 0.005 px and state confidences at most 0.0007. On its own, against the eager reference, it occasionally flipped a stop decision on the 256-image archive (one arm 5.6 px shorter, confidence 0.48 lower), failing the strict gates, which is why the compiled configuration is judged by the relaxed gate instead.

Compiled Hydra and GRU steps use fixed kernel configurations rather than Inductor's speed-based autotuning, so repeated runs on one GPU type give the same bytes. Compilation happens on first use and is cached by Inductor. Triton builds one C helper module on first use; images without a C compiler can point `DEEPSTARNOODLES_TRITON_PREBUILT` at a manifest of prebuilt helpers, each pinned by source and binary SHA-256.

Do not compile StarSeg or run it in half precision. On the lab validation mix even small changes to detection boxes move crops, and star counts changed in 6 of 371 frames when compiled and in 10 of 371 at half precision.

## Scheduling

- `--detect-ahead N` runs decode, StarSeg and crop preparation up to N source groups ahead on a second CUDA stream. Hydra, the GRU and output writing stay in input order on the main stream.
- `--archive-processes N` (archive only; 1–16) shards frames round-robin across N resident processes on one GPU and merges their records back into input order. Each process applies its own allocator limit, so device memory is shared, not capped in total. Without CUDA MPS the processes take turns on the GPU. Start an MPS control daemon (`nvidia-cuda-mps-control -d`, stopped with `echo quit | nvidia-cuda-mps-control`) so their kernels run concurrently. On the RTX 5090, eight processes under MPS processed 64.5% more images per second than four without it, with identical predictions.
- `--decode-workers N` (archives; 1, 2 or 4; 0 keeps synchronous decoding) reads images in worker threads with the same OpenCV decoder and yields them in source order, preserving detector batches. Round 11's selection uses 2, with `--decode-prefetch-frames 4 --decode-prefetch-mib 256 --decode-max-frame-mib 64`. The queue reserves decoded-array bytes from image headers before submitting jobs, and it rejects frames larger than `--decode-max-frame-mib`. Reservations cover reader-owned queued and in-flight decoded arrays, **not** the current consumer batch, native codec scratch, compressed buffers or filesystem cache. `--decode-prefetch-frames` (at most 64) and `--decode-prefetch-mib` (at most 1024) bound extra retained work, and the process RSS guard still applies. Staged decoding closes its queue before loading Hydra and reports aggregate statistics across shards. The report's `archive_decode` statistics separate summed worker decode time from consumer wait. Profiling calls the wait stage `archive_decode_wait`; overlapping worker time is not a serial pipeline stage. No frames are prefetched beyond the selected CSV rows, and warmup gets a separate reader. Native filesystem or decoder calls can't be cancelled by Python threads, so a process deadline remains necessary for a blocked input. Benchmark input hashing warms the cache, so this doesn't establish cold-storage performance.
- `--processes N` is the same idea for `serve-videos`, where workers claim whole videos. It rejects `--decode-workers`, because each worker has its own decoder.
- `--gpu-crops` (not combined with `--crop-streams` or `--archive-frame-window`) uploads each source frame that has detections and cuts, pads and resizes its Hydra crops on the GPU. It reproduces Pillow's 8-bit bilinear resize exactly: the same 22-bit fixed-point coefficients, computed in float64 on the host, then both passes in int32 on the GPU, each rounded to uint8. Hydra inputs are therefore **byte-identical** to the CPU path. That held on all 454 archive crops and on test images from 16 to 7,838 px against Pillow 11.2.1 and 12.0.0. On the RTX 5090 it raised eight-process MPS archive throughput by 15.4% on its own.
- `--cuda-blocking-sync` makes host threads that wait for the GPU sleep instead of spinning. By default the CUDA driver spins a CPU core during every synchronization when a process sees more CPUs than it has CUDA contexts, and containers that show many CPUs but enforce a small CPU quota pay for every spinning waiter. Outputs are unchanged. Locally, eight archive processes under MPS used 26% less CPU time. That was 20% faster when the CPU quota was the limit, but 9% slower when the GPU was. On the RTX 5090 container it raised eight-process MPS throughput by 6.6% alone, and by 20.1% with `--gpu-crops`.

## Guards and profiling

- `--allocator-limit-gib` (default 7) caps each process's PyTorch allocator. It is a guard, not GPU emulation: a limit on a larger GPU doesn't reproduce a smaller GPU's speed or guarantee total memory fit. PyTorch caches freed blocks up to this limit.
- `--ram-limit-gib` (default 24) fails the run when a process's sampled peak RSS exceeds it.
- `--profile-stages` synchronizes each stage for per-stage timings. Use a separate run, because throughput runs must omit it.
- `--images` limits a run, and `--warmup` sets warmup images. `archive` and `live-replay` process every CSV row by default with no warmup. `benchmark` defaults to 128 measured images and eight warmup images, and both forms share the same inference path. Live replay measures ordered files, including decoding and JSONL output; it doesn't simulate camera arrival, dropped frames or queue latency.

## Not selected, kept for comparison

These work and have tests, but measurement didn't support them. Don't use them without new evidence.

- `--gru-streams N` (1–8; requires `--gru-execution graph`; not combined with `--crop-streams`) rolls out the GRUs of up to N crops at once. Each slot has its own stream, graphs and static buffers. One host thread replays every running slot and reads each slot's any-walker-active flag one step late, so the streams never wait for the host between steps. Hydra runs first for all crops in the group, and tracking and records keep crop order. With `--detect-ahead`, archive mode gathers crops across up to 8 queued frames. Faults were more common with it, at about 3 in 20 processes without MPS.
- `--crop-streams N` (1–4) runs up to N crop groups of one frame concurrently, each on its own CUDA stream with separate Hydra and GRU graph state. Tracking and records keep crop order. Graph captures exclude concurrent replays, because PyTorch registers its default CUDA generator with every capture.
- `--starseg-execution graph` captures StarSeg's backbone and neck once per input shape and replays them. Detections, masks and polygons were identical on all 256 archive frames. It keeps about 550 MiB of activations per process, and with the compiled configuration it showed no speed gain.
- `--channels-last` runs Hydra in channels-last memory layout. It adds nothing on top of `--hydra-compile`.
- `--starseg-half` and `--hydra-fp32` change model precision. Half-precision StarSeg changed star counts in 10 of 371 frames.
- `--hydra-batch-size 1 --crop-batch-size 4` runs Hydra separately for each crop, then passes the ordered group of up to four crops to the GRU. It can isolate dense-model batch effects, but it doesn't guarantee parity, higher speed or lower peak memory. The Python API uses `LiveConfig(hydra_batch_size=1, crop_batch_size=4)`.
- `--archive-frame-window` keeps detector calls at one full image while filling Hydra and GRU crop groups from successive images. The tested grouped configurations failed the trained-output parity checks. See [its details](#cross-frame-crop-accumulation).
- `staged-archive` runs the detector and dense/tracer stages separately through a disk spool to reduce simultaneous model residency. See [its details](#staged-archive).

### Cross-frame crop accumulation

```bash
deepstarnoodles-inference archive --models models/ --index inputs/archive.csv \
  --output outputs/crop-window --detector-batch-size 1 --crop-batch-size 8 \
  --hydra-batch-size 1 --hydra-output-mode minimal \
  --archive-frame-window 64 --archive-crop-buffer-mib 256
```

`--archive-frame-window 0` is the default. A positive window (at most 256) limits outstanding source records, and the crop queue fills up to `--crop-batch-size` (at most 128 in this mode). A full crop batch, frame-window limit, byte limit or end of input triggers a flush. Every source record stays in order, including empty detections, and source records wait until all their crops finish. It requires detector batch size 1 and is rejected for live, replay and staged commands.

The byte budget (1–1024 MiB, default 256) covers retained RGB crops **and owned mask polygon arrays**. Exact crop geometry and polygon bytes are checked before allocation, and an individually oversized crop fails explicitly. The current source frame, detector descriptors, model tensors, native scratch and partial output for one source are outside this budget, and the process RSS guard still applies.

`report.json` identifies `resident-cross-frame-crops`, queue peaks, flush reasons and the crop-batch histogram. Buffered `source_group_latency` measures source ingestion and any flush it triggers, excluding the final flush at the end of input, so it is **not per-image completion latency**.

### Staged archive

```bash
deepstarnoodles-inference staged-archive --models models/ --index inputs/archive.csv \
  --output outputs/staged --images 128 --detector-batch-size 1 --crop-batch-size 1 \
  --shard-size 32 --max-spool-gib 8
```

`--shard-size` bounds the number of source images processed before reloading stages. `--max-spool-gib` bounds live intermediate files and stops the run if exceeded. Stage initialization and spool I/O count toward job time. It defaults to 128 images, so set `--images` explicitly for a full workload.

## Camera and live details

- **Frames.** Camera capture uses a background reader with **one pending latest frame**. If inference falls behind, a new frame replaces it, and the report counts overwrites and shutdown discards. `--queue-policy sequential` disables the background reader. Video files always use sequential capture. `--max-frames` counts processed frames. Each new command starts fresh tracker state, and crop batches combine animals within the current frame.
- **Indexes and timing.** `source_index` counts successful capture reads, and `processed_index` counts inference calls. Temporal trackers age by processed observations even when the source sequence skips. Reported acquisition time is the monotonic timestamp immediately after OpenCV `read()` returns, **not sensor exposure time**. Queue wait, inference completion and total post-acquisition latency are measured separately. Video timestamps are best-effort OpenCV `POS_MSEC`. Timing means cover the full session, and quantiles cover the most recent 1,024 frames. Cold first inference is included.
- **Record time.** Every record ends with `time_s`, `fps` and `schema_version` (3). `time_s` is `frame / fps`, where `frame` is the video frame for `serve-videos`, `source_index` for `live` and the CSV row for image indexes (so an index of stills cut from mid-video starts at 0 s; add the cut's start time yourself). The rate comes from the video for `serve-videos` and `live --video` (OpenCV `CAP_PROP_FPS`). Image-index commands (`archive`, `live-replay`, `benchmark`, `staged-archive`) take it from `--fps F`, and sharded archives recompute `time_s` from the global frame after the merge. Without a rate, as for `live --camera`, both fields are `null`; a camera's nominal rate is not a clock, so use `source_timestamp_s` and `acquired_monotonic_ns` there. The Python API takes `predict(..., fps=F)`.
- **Shutdown.** End of file, Ctrl-C, SIGTERM and inference errors close the capture and write a final report. OpenCV can't always interrupt a blocking read: shutdown waits up to one second and reports `reader_stopped: false` if it is still blocked. A video read failure may be either end of file or a decoding failure, and the output names that ambiguity.
- **Your own frames.** An external acquisition loop may pass strictly increasing `source_index`, `acquired_monotonic_ns=time.monotonic_ns()` recorded after acquisition, and an optional `source_timestamp_s` to `predict()`. Concurrent calls on one pipeline are rejected.

The live profile (`--crop-batch-size 1 --hydra-output-mode minimal --gru-execution fixed --gru-check-interval 1`) passed all five warmed-replay comparisons. Mean per-run p95 latency fell 14–17% on the three positive ROV clips and 7% on the three-frame crowded probe. The detector-only negative clip was mixed and slower on average. These short replays don't establish sustained camera frame rates or reliable tail latency, and the camera command includes its first inference without warmup.
