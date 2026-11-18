# Inference benchmark results — completed pilot through Round 16, 28 October–8 November 2026 (UTC)

**0.5.0.dev11 and dev12 keep what later steps need from Hydra's maps, without changing any arm.** Those maps are about 2 MiB per crop, too large to save, so dev11 keeps each star's body origin, support area and centroid, and dev12 adds the mouth, madreporite and butthole as `[x, y, p]`. dev12 also adds `track-videos`, which gives `serve-videos` output star and arm IDs. On 1,435 frames with 3,502 stars every star matched dev10 apart from the added fields; see [the section](#hydra-fields-and-tracking-after-annotation-050dev11-and-dev12).

**Round 16 (0.5.0.dev10) measures the compiled pipeline on nine GPU types on RunPod's secure cloud. The RTX 5090 costs the least per frame.** It sustained **58.1 images/s** at $1.19/h, or $5.69 per million frames. The RTX 4090 is next, at 41.1 images/s and $6.02 per million frames. The RTX PRO 6000 Blackwell was fastest at **67.4 images/s**, at $10.26 per million frames. The 5090 was out of stock throughout the 31 October runs and was measured on 8 November with the same frozen bundle. All 18 sessions were steady by the rules below, and none faulted. The round also found and fixed GPU memory growing with every video, which had tripped the device guard in its first pass. See [its section](#round-16-speed-test-across-gpu-types).

**From Round 13 the benchmark is streamed video annotation.** Videos start on the workstation and are uploaded to the pod while it works. Only annotations come back, and throughput is the rate at which they arrive at the workstation, averaged over at least ten minutes of sustained work. On the synthetic lab videos, one RTX 5090 sustained about **36–39 images/s with eight processes and about 45 with compiled GRU steps** (100–122 star crops/s), with its GPU at the 575 W power limit. These are long-run estimates on the whole lab mix: Round 13's windows themselves were not sustained rates, as its section explains, and from Round 16 the videos are cut into two-minute segments so that the measured window is one. Upload from the workstation carried about 25 Mbit/s over one SSH connection and 50–52 Mbit/s over six. Six connections feed roughly one pod; one connection starved it.

**Round 11 (0.5.0.dev5) prepares Hydra crops on the GPU and lets threads sleep while they wait for it, raising eight-process archive throughput by 20% with unchanged predictions.** On the same image, runtime, weights and workload, eight processes under MPS with `--gpu-crops --cuda-blocking-sync` processed **52.25 / 52.21 images/s** against **43.38 / 43.61** for Round 7's eight processes (**+20.08%** ratio of means). GPU crops alone gave +15.43% and blocking waits alone +6.62%. Ten processes with both processed 53.06 / 53.35 images/s (+23.76% against their own eight-process baseline). Every run reproduced the Round 4 selected bytes. CPU per image fell from about 220 ms to about 125 ms and the CPU quota stopped throttling. The archive now uses about 7 of the 13.6 cores, so the GPU is the limit.

**Round 10 (0.5.0.dev4) compiles the GRU step and runs Hydra channels-last for live replay, judged by a relaxed agreement gate.** From this round on, very slight output differences are acceptable. The gate requires the same star count per frame, box IoU of at least 0.98 for 99.5% of paired stars, and 99% of reference arms within 2 source pixels. All four clips passed. On three clips every arm stayed within 0.5 px. On the sparse crowded probe one arm changed its step count and the candidate added one arm (86 against 85). Warm p95 changed by **−16.78%, −3.92%, −11.57% and +3.88%**. Compilation added 12–35 s to each process start, and outputs differed run to run. It is optional for long live sessions, not a default.

**Round 9 (0.5.0.dev4) finds the eight-process archive limited by the container's 13.6-core CPU quota; more processes or one OpenCV thread do not help.** Ten processes with one OpenCV thread processed **36.92 / 42.40 images/s** against **44.33 / 42.79** for eight (**−8.96%**, same bytes). They were throttled by the quota for 13.3–15.6 s per run, against 4.3–5.3 s for eight. Channels-last Hydra with crop batch 4 exceeded the 30 GiB sampled device guard at eight and ten processes. One OpenCV thread raised warm p95 on the crowded live probe by 22.36%.

**Round 8 (0.5.0.dev4) finds no further archive gain: eight processes under MPS remain the selection.** Eight processes reproduced Round 7 at **41.95 images/s** with the same bytes. Ten processes processed 40.12 images/s. A compiled GRU step (`--gru-compile`) processed 40.72 images/s, failed the strict gates against the eager reference, and was informational only. With MPS the archive is no longer limited by the GRU. The compiled step lowered warm live p95 on the sparse crowded probe by 15.78% (paired −25.00% / −3.71%) and changed the ROV clip by −2.46% (paired −9.07% / +5.26%), with outputs that differed run to run.

**Round 7 (0.5.0.dev3) runs eight archive processes under CUDA MPS for 1.65× the Round 6 archive throughput, with unchanged predictions.** On the same RTX 5090 image, runtime, weights and workload, eight processes under MPS processed **43.49 / 44.60 images/s** against **26.46 / 27.07** for Round 6's four processes (**+64.52%** ratio of means). Zero-warmup processing rose **+50.20%**. Every archive run reproduced the Round 4 selected predictions byte for byte. Round 6's plateau was not CPU: the container's 13.6-core quota was about half used. Without MPS, processes take turns on the GPU; with MPS their kernels run concurrently, and board power rose from about 314 W to 435 W.

**Round 6 (0.5.0.dev2, whole-GPU phase) runs four resident archive processes for about 2.5× the Round 5 archive throughput, with unchanged predictions.** Each process keeps the 7 GiB allocator limit; only the sampled whole-GPU guards were raised. Four processes processed **25.43 / 25.83 images/s** against **10.35 / 10.35** for the single-process Round 5 candidate (**+147.62%** ratio of means). Zero-warmup process wall fell from **34.2 / 33.9 s** to **24.7 / 24.1 s**. Two crop streams cut warm p95 on the sparse crowded probe by **33%**. Every archive run reproduced the Round 4 selected predictions byte for byte.

**The optimized decoders leave validation accuracy unchanged.** On the 478-image annotated validation split, every decoder scores the released tracer at **F1 0.8032** and **group-macro F1 0.7506**, with identical match counts on every image. The decoders tested were CORE's reference, the package's vendored copy, fixed slots (Round 4), CUDA graphs (Rounds 5–6) and four concurrent GRU streams. The recorded validation of the same checkpoint is F1 0.8034. That run had two more matched arms; it computed fields and decoded on an RTX 5090, where floating-point rounding differs (see [Validation accuracy](#validation-accuracy-of-the-optimized-profiles)).

**Round 5 (0.5.0.dev1) more than doubles archive throughput and roughly halves warm live p95, with predictions byte-identical to the Round 4 selection.** On the same RTX 5090 image, runtime, weights, workload and strict gates, the warmed archive candidate processed **9.05 / 10.59 images/s** against **4.72 / 4.49** for the Round 4 selection (**+113.29%** ratio of means). The zero-warmup whole job took **33.5 / 33.7 s** against **66.8 / 68.9 s** (process rate **+101.64%**). Warm live p95 fell **48–68%** on the three positive ROV clips and the sparse crowded probe; the detector-only negative clip is unchanged within its variation. The Round 4 text below is retained as measured.

**Round 4 — for archives, use singleton detector/crop batches, minimal Hydra outputs, fixed GRU with check interval 1, and two bounded decode workers.** The final same-build comparison improved zero-warmup whole-process throughput by **32.72% and 22.23%** in two paired runs (**27.48% ratio-of-means gain**); process wall fell **24.65% and 18.19%**. A separate comparison held decode workers fixed and measured the added minimal/fixed profile at **+6.76% / +7.18% processing throughput**. Strict output parity passed throughout.

**Round 4 — for live use, singleton minimal/fixed1 is a qualified warmed-replay candidate with synchronous decoding.** Both repeats improved p95 on every positive clip, including the sparse crowded probe. The detector-only negative clip varied sharply and regressed in one repeat. This supports testing the candidate on the target machine; it does not establish a uniform latency reduction, a camera FPS guarantee, or sustained queue behavior. Earlier zero-warmup live results remain variable and are retained below.

The pilot through Round 13 ran on an **RTX 5090**, with resource guards intended to leave headroom for an 8 GiB GPU and 32 GiB host, and Round 16 ran on nine GPU types, the 5090 among them. None of this establishes RTX 5060 performance or certifies physical 8 GiB fit. The trained cascade is **YOLO26-L segmentation → DINOv3 ConvNeXt-Base Hydra → GRU128**; the Hydra checkpoint is Base, not Large.

## Streaming benchmark (Round 13 on)

**Rule.** The source data are videos on the workstation. They are uploaded to the pod during the run, can be cached there on a rolling basis and are decompressed on the pod. Only annotations return, and a frame counts once its annotation has reached the workstation. Throughput is the average over at least ten minutes of sustained wall-clock annotation.

**Videos.** The lab recordings this simulates are 30 or 60 minutes long at 1 frame per second. `streaming/make_videos.py` builds one 30-minute, 1 fps video per lab camera setup in the StarSeg validation split: 12 setups, 371 images, from 640×480 to 5496×3672. Frames loop over that setup's images in a fixed shuffled order, each held for 5 s like a slowly changing lab scene, so every frame is still annotated in full. The H.264 encode targets 0.2 bits per pixel. Real 2600×2128 lab footage averages 0.16 with MPEG-4 part 2, while some 12–20 MP stills cost five times that at constant quality. The 12 videos total 2.2 GB, and the upload schedule weights each setup by its validation count, for 91 KB per frame on average.

**Sustained rate.** The measure must be a good proxy of the sustained rate, and with 30-minute videos a ten-minute window was not one. A one-star video takes a worker about 2.5 minutes, and a seven-star `coldbox` video about 15. Workers therefore finish the light videos first and pile up on the heavy ones, so images/s falls through a session while star crops/s stays flat, and a session ends before the mix reaches its long-run proportions. A processor-sharing simulation fitted to Round 13 reproduces the curve, with a −15% difference between the window's halves and up to ±8.3% from where the sequence starts. With two-minute segments the same simulation gives 0.0% and ±0.9%. So from Round 16:

- `streaming/make_segments.py` cuts the same videos at keyframes, without re-encoding, into 180 segments of about 120 frames (80–150). Sampled frames decode to identical pixels. The feeder plays each setup's segments in turn, at the setup weights. One file against the same frames in four segments showed no measurable boundary cost (15.4 and 16.0 images/s, within run-to-run noise).
- A window counts as a sustained rate only if it was steady: its second half is within 5% of its first, its stars per image are within 5% of the whole lab mix (2.76), and the pod's queue of videos stayed stocked, so the pod set the pace rather than the uplink. `ops/stream_analyze.py` and the dashboard apply these checks.

**Harness.** Scripts named under `ops/`, `streaming/` and `review/`, and results under `stream_results/`, belong to the campaign workspace and are not in this repository. Rounds 12, 14 and 15 were planned but never ran, because of a pod that never booted and a lack of RTX 5090 stock, and Round 16 replaced them.

- **Feeder** (`ops/stream_session.py`, workstation): keeps 12 videos waiting in the pod's `ready/` directory. Each video is re-sent under a new sequence-numbered name every time it is scheduled, as unseen data would be.
- **Service** (`deepstarnoodles-inference serve-videos`, pod): eight or ten resident processes under MPS claim whole videos, decode them and annotate every frame. Each process deletes a video once it has finished it.
- **Collector** (workstation): `stream_results.py` streams each record back over SSH as soon as it is written. The collector stamps it with the workstation's clock and stores it locally.
- **Sessions:** each configuration runs as a twelve-minute session. `ops/stream_analyze.py` reports the ten-minute window that starts two minutes after the session's start marker arrives, plus a per-minute series. It also compares frames annotated in more than one session: exact configurations must match byte for byte, and compiled ones are judged by the relaxed gate.

## Round 16: speed test across GPU types

Round 16 installs `deepstarnoodles_inference-0.5.0.dev10` (sha256 `445e299b…`). It adds no new speed option. Each GPU type ran the compiled configuration (float16 Hydra weights, compiled Hydra and GRU, first-step graphs) for two 14-minute sessions at different process counts, on one secure pod. Sessions used a 48-segment subset of the two-minute segments, uploaded once and replayed from a cache on the pod, so the uplink set no limit. Windows start four minutes into each session. The round spent $4.89 under dev10, plus $0.90 on the earlier pass described below. The RTX 5090 had no secure stock during those runs. It ran on 8 November with the same frozen bundle, wheel, segments and window rules, for $0.87.

| GPU | $/h | Processes | Images/s | Star crops/s | Stars per image | $ per million frames | Peak device memory, MiB | GPU busy | Mean board power |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RTX 5090 | 1.19 | 10 | 57.84 | 162.1 | 2.80 | 5.71 | 21,719 | 99% | 573 W |
| | | 12 | 58.10 | 163.2 | 2.81 | 5.69 | 25,793 | 99% | 572 W |
| RTX 4090 | 0.89 | 8 | 41.06 | 115.2 | 2.81 | 6.02 | 16,058 | 99% | 428 W |
| | | 10 | 40.94 | 114.1 | 2.79 | 6.04 | 18,928 | 100% | 431 W |
| RTX 3090 | 0.50 | 6 | 20.59 | 58.5 | 2.84 | 6.74 | 10,833 | 99% | 334 W |
| | | 8 | 20.93 | 58.9 | 2.81 | 6.63 | 14,354 | 99% | 335 W |
| RTX PRO 4500 Blackwell | 0.72 | 6 | 27.55 | 77.4 | 2.81 | 7.26 | 11,225 | 99% | 164 W |
| | | 8 | 27.18 | 76.3 | 2.80 | 7.36 | 14,619 | 99% | 165 W |
| RTX A6000 | 0.59 | 6 | 20.95 | 59.4 | 2.84 | 7.82 | 10,932 | 97% | 293 W |
| | | 8 | 22.17 | 62.2 | 2.81 | 7.39 | 14,468 | 99% | 297 W |
| L40S | 1.09 | 8 | 40.61 | 113.3 | 2.79 | 7.46 | 16,591 | 99% | 328 W |
| | | 10 | 40.24 | 112.9 | 2.80 | 7.52 | 19,834 | 99% | 337 W |
| RTX 6000 Ada | 0.99 | 8 | 36.38 | 101.2 | 2.78 | 7.56 | 15,953 | 100% | 297 W |
| | | 10 | 35.72 | 99.5 | 2.79 | 7.70 | 19,452 | 100% | 298 W |
| A40 | 0.59 | 6 | 19.19 | 53.4 | 2.78 | 8.54 | 10,941 | 98% | 296 W |
| | | 8 | 19.26 | 54.8 | 2.85 | 8.51 | 14,489 | 99% | 296 W |
| RTX PRO 6000 Blackwell | 2.49 | 10 | 67.38 | 188.3 | 2.80 | 10.26 | 22,834 | 100% | 544 W |
| | | 14 | 66.17 | 185.1 | 2.80 | 10.45 | 30,957 | 99% | 550 W |

Every window's halves were within 3.8% of each other and its stars per image within 3.1% of the mix, and the queue never ran dry, so all 18 are sustained rates. All GPUs were busy throughout, and the second process count changed the rate by less than 2% except on the A6000 (+5.8%).

**GPU memory grew with every video, and 0.5.0.dev10 fixes it.** The first pass, on dev9, stopped the RTX 3090's sessions at the 96% device guard after 4 minutes, and the eight-process sessions on the 46 GB A6000 and A40 after 7–8 minutes. Single workers on a local A6000 under Torch 2.8 showed why: reserved memory rose about 506 MiB per video, and allocated memory about 32 MiB per video, whatever the video's resolution. `serve-videos` made a new detect-ahead stream for each video. The caching allocator reuses free blocks only within their stream, and cuBLAS keeps a 32 MiB workspace per handle and stream, so each new stream stranded a working set. PyTorch's pool of 32 streams was the only bound. dev10 keeps one stream for the worker's life. Reserved memory then stays at 1.29 GiB and peak allocated at 0.85 GiB from video to video, up to 5496×3672, and allocator limits as low as 1.6 GiB ran without error. Outputs were byte-identical on all 19 videos compared. The dev9 runs are kept under `stream_results/round16c_*_stranded`; their six-process rates on the A6000 and A40 were 20.0 and 18.7 images/s, against 20.95 and 19.19 here. The speed test also now sets each process's allocator limit to its share of the GPU, at most 7 GiB.

**Faults depend on MPS.** The same local workers, without MPS and time-slicing one GPU, hit the known illegal memory access 3 times in about 1.1 process-hours. Each was an Xid 31 MMU fault (`GPCCLIENT_GCC`, `FAULT_PDE`, `ACCESS_TYPE_VIRT_READ`), mid-video at unrelated frames. A worker forcing a cache release once a second did not fault sooner than the others, and another faulted without ever reaching its allocator limit, so cached-block release isn't the trigger. Under MPS there were none in the 31.4 worker-hours of the 31 October runs, or the 13.9 streamed before them. The RTX 5090's two sessions on 8 November also ended normally, with no video errors. The root cause remains unknown.

**Outputs depend slightly on the GPU's chip family, with no visible bias.** Each type's two process counts agreed on every shared frame, with arms within 0.01 px. Between types, on the 5,730 frames all of them annotated (`review/round16_cross_gpu.py`), the A6000 and A40 were identical, the 4090 and RTX 6000 Ada had no arm outside 2 px, and the PRO 4500 and PRO 6000 agreed on 99.998% of arms. The L40S kept 99.7% of the 4090's arms within 2 px, and the 3090 kept 99.3% of the A6000's. Across families the figures are 98.4–98.5% with 99.7–99.8% of frames at the same star count, just under the relaxed gate's 99% line. The RTX 5090, measured later, kept 98.4% of the A6000's arms within 2 px on 5,850 shared frames, with 99.8% of frames at the same star count, in line with the other cross-family pairs. Across the eight types run on 31 October, stars per frame stayed within 2.398–2.400, arms per star within 9.89–9.90, and mean arm length within 100.15–100.18 px.

## Hydra fields and tracking after annotation (0.5.0.dev11 and dev12)

`serve-videos` annotates frames independently and drops Hydra's fields after the GRU. From dev11 each star also keeps the body origin that arm tracking measures arms from, and the support area and centroid (`body_fields.py`). dev12 adds each landmark heatmap's maximum with its probability. They are summarized on the GPU from the maps inference already holds, and read once the GRU has synchronized. `track-videos` then runs BoT-SORT and the per-star arm tracker over the saved records.

- **Arms unchanged.** On twelve two-minute lab segments on an RTX A6000, in the compiled configuration, all 3,502 stars in 1,435 frames matched the dev10 output byte for byte apart from the added fields.
- **Checked against a second implementation.** A separate implementation of the body origin, written independently of dev11's, agreed on all 3,502 stars. A version that took landmarks from the logits instead of the probabilities agreed on every landmark.
- **Cost.** Computing these on the GPU took about 0.22 ms per crop, in a microbenchmark on an RTX A6000 that another job was also using. That is under 1% of a crop's time. Compressed output grew by 68 bytes per star, about 1%.
- **Same tracker inputs.** Given the same arms, tracking from a saved body origin assigns the same IDs as tracking from the maps.
- **Same tracking quality.** On the first 240 frames of a real 1 fps lab clip, live replay and `track-videos` both kept one star ID throughout. An arm changed ID between consecutive frames 5.44% of the time in live replay (113 times) and 5.62% after annotation (117 times). Their arms lay a median 0.87 px apart, because live replay crops from BoT-SORT's smoothed boxes rather than the detections, so the two don't make the same ID decisions at ambiguous frames. Seven detections that BoT-SORT never confirmed went to `untracked_stars`, and `representations` read the tracked file without `--untracked`.

On these clips the mouth was rarely visible: its median probability was 0.007, so 88% of body origins came from the support centroid. The butthole's median probability was 0.68, and the madreporite's 0.08.

## Round 13: first streamed annotation round

Round 13 installs `deepstarnoodles_inference-0.5.0.dev7` (sha256 `2daf033d…`), which adds `serve-videos`. Sessions ran in A-B-C-A order. The pod booted in 26 minutes; the round's lease cost at most $1.72.

| Session (10-min window) | Images/s | Star crops/s | Stars per image | Upload in window, Mbit/s | Long-run estimate, images/s | Change between the window's halves |
|---|---:|---:|---:|---:|---:|---:|
| A1: eight processes, Round 11 selection; feeder on one SSH connection | 38.07 | 100.2 | 2.63 | 18 | 36.1 | −43% |
| B1: ten processes; feeder on six connections | (62.31) | (118.2) | 1.90 | 50 | 41.9 | −22% |
| C1: eight processes, compiled GRU steps and first-step graphs | 46.30 | 122.4 | 2.64 | 52 | 44.7 | −16% |
| A2: eight processes; feeder on six connections | 36.16 | 111.9 | 3.09 | 26 | 39.1 | −12% |

Upload in window is spread over each transfer's duration. **The windows are not sustained rates**, because images/s fell through each one as workers piled up on heavy videos. The long-run estimate is the rate on the whole lab mix (2.76 stars per image), from `review/round13_long_run_estimate.py`: the mean of a per-minute GPU budget fitted to images and crops, and of per-setup worker time reweighted to the mix. The two agree within 2%. A1's estimate reads low because its feeder kept only 6.9 of 8 workers busy; B1's skips the replay burst described below. Three effects make the window numbers hard to compare directly.

- **A1's feeder starved it.** Every upload in A1 shared one multiplexed SSH connection, about 25 Mbit/s in total. A1 ran at about 60 images/s while the eight pre-uploaded videos lasted. Its per-minute rate then fell to 19–21 with the queue empty, and recovered to about 36. The feeder was switched to one connection per upload, six at a time, from B1 on. The workstation's uplink then carried 50–52 Mbit/s while uploads were needed. A2 needed only 26 Mbit/s, because its queue stayed full.
- **B1's window is inflated.** The restarted collector first replayed A1's records, so B1's records arrived late and then in a burst. B1 ran at 48–65 images/s through most of the session.
- **Content varied.** Each session drew a different slice of the video sequence, and stars per image ranged from 1.9 to 3.1. Twenty-megapixel `snackpack` videos with many stars per frame cost far more GPU time than one-star 1280×720 videos. Star crops per second, which track GPU work, varied less than images per second.

Throughout, the GPU was 99–100% busy at 520–579 W, against the RTX 5090's 575 W limit, while the container used about five of its 13.6 CPU cores. Ten processes peaked at 30,650 MiB of device memory, just under the 30 GiB guard.

Outputs matched the eight-process A1 on every frame both sessions annotated: 15,352 frames for B1 and 14,822 for A2. The compiled C1 passed the relaxed gate on 18,105 shared frames, with 99.999% of arms within 2 px both ways and one arm's step count changed.

## Round 11: GPU crop preparation and blocking CUDA waits

Round 11 installs `deepstarnoodles_inference-0.5.0.dev5` (sha256 `5a41a3ef…`, base commit `3e2eccad`). Round 9 had shown eight archive processes saturating the container's 13.6-core CPU quota. Local profiling found two large CPU costs:

- **Spinning waits.** When a process has fewer CUDA contexts than visible CPUs, the driver spins a core for every synchronization. The pod shows 128 CPUs. One thread waiting 1.9 s for a GPU kernel used 1.9 s of CPU time, and 0.07 s with blocking synchronization.
- **Crop preparation.** Each crop was converted to RGB, copied into Pillow and resized on the CPU: 8.7 s of CPU for the archive's 454 crops. Crop sides reach 7,838 px, and single large crops took up to about 400 ms.

Two exact options address them:

- `--gpu-crops` uploads each frame that has detections and reproduces Pillow's 8-bit bilinear resize on the GPU. It uses the same 22-bit fixed-point coefficients, computed in float64 on the host, then both passes in int32, each rounded to uint8. It matched the CPU path on all 454 crops, and Pillow 11.2.1 and 12.0.0 on test images from 16 to 7,838 px.
- `--cuda-blocking-sync` sets the primary context's blocking-sync flag through the driver API, and creates host-waited events as blocking events.

Image decode was not a lever: about 95% of its time is the archive's 5–20 MP PNGs, which nvJPEG cannot decode and Pillow decodes 15% more slowly than OpenCV.

Same container memory (57.7 GiB) and CPU quota as Round 10. No case faulted. All 16 optimized archive runs produced the Round 4 selected bytes (`0e5e4e99…`), and the eager reference reproduced Round 4's eager bytes. Each block compares a candidate (B) with eight processes under MPS (A), in A1, B1, B2, A2 order.

| Candidate (B) | A repeats, images/s | B repeats, images/s | Ratio of means | Paired |
|---|---:|---:|---:|---:|
| GPU crops and blocking waits | 43.381 / 43.613 | 52.250 / 52.213 | +20.08% | +20.45% / +19.72% |
| GPU crops | 44.268 / 45.456 | 51.947 / 51.625 | +15.43% | +17.35% / +13.57% |
| Ten processes, GPU crops and blocking waits | 45.031 / 40.950 | 53.057 / 53.353 | +23.76% | +17.82% / +30.29% |
| Blocking waits | 38.925 / 42.040 | 43.276 / 43.045 | +6.62% | +11.18% / +2.39% |

| Configuration | CPU per image, ms | Throttled per run, s | Device peak, MiB |
|---|---:|---:|---:|
| Eight processes (A runs) | 214–248 | 3.4–5.1 | 21,575–21,653 |
| Eight, GPU crops and blocking waits | 121–127 | 0 | 21,643–21,707 |
| Eight, GPU crops | 220–222 | 5.1–6.6 | 21,709–21,715 |
| Ten, GPU crops and blocking waits | 137–139 | 1.3–1.6 | 27,076–27,142 |
| Eight, blocking waits | 162–181 | 0 | 21,581–21,649 |

CPU per image is process-tree CPU time over the timed window divided by images processed. GPU crops alone leave it unchanged, because spinning waiters absorb the freed cycles. They still raise throughput by 15%, because large crops no longer stall each process's pipeline. In the fastest runs, shards finished 3.6–4.9 s after the common start, so the round-robin split of this 256-image job loses about 7% to the slowest shard. Round 11 used one secure RTX 5090 lease of 0.30 h (at most $0.36).

```bash
# Round 11 archive selection: Round 7 plus GPU crop preparation and blocking waits, under a running MPS daemon.
nvidia-cuda-mps-control -d
deepstarnoodles-inference benchmark --mode archive \
  --models /models/manifest.json --index /inputs/archive.csv \
  --output /outputs/round11-archive --images 256 --warmup 4 \
  --detector-batch-size 1 --crop-batch-size 1 \
  --hydra-output-mode minimal --gru-execution graph --gru-check-interval 1 \
  --hydra-execution graph --detect-ahead 2 --archive-processes 8 \
  --gpu-crops --cuda-blocking-sync \
  --decode-workers 2 --decode-prefetch-frames 4 \
  --decode-prefetch-mib 256 --decode-max-frame-mib 64 \
  --archive-frame-window 0 --archive-crop-buffer-mib 256 \
  --allocator-limit-gib 7 --ram-limit-gib 24
echo quit | nvidia-cuda-mps-control
```

## Round 10: compiled GRU step and channels-last Hydra for live replay

Round 10 reuses Round 8's image, overlay wheel (0.5.0.dev4), weights and clips. Each block compares a reviewed live baseline with the same settings plus `--gru-compile --channels-last`, in A1, B1, B2, A2 order. The baselines are the Round 6 two-crop-stream candidate on the sparse crowded probe and the Round 5 graph candidate on three ROV clips. Both changes alter arithmetic slightly, so candidates are judged by the relaxed agreement gate (`relaxed_gate.py`) against the block's first baseline. Every baseline reproduced its earlier live bytes. Container: 57.7 GiB memory, 13.6-core CPU quota; the compile probe used the prebuilt Triton helper. No case faulted.

| Clip (warm live, ABBA) | A p95, ms | B p95, ms | Ratio of means | Paired | Images/s, ratio of means |
|---|---:|---:|---:|---:|---:|
| Sparse crowded, two crop streams | 165.5 / 160.5 | 146.0 / 125.3 | −16.78% | −11.78% / −21.93% | +22.14% |
| ROV 2003, tracks 10–25 | 82.1 / 97.9 | 77.3 / 95.7 | −3.92% | −5.88% / −2.28% | +3.99% |
| ROV 2004, tracks 1–14 | 66.0 / 57.3 | 54.8 / 54.2 | −11.57% | −16.90% / −5.44% | +12.73% |
| ROV 2008, track 12 | 76.7 / 76.4 | 88.3 / 70.7 | +3.88% | +15.15% / −7.44% | −7.53% |

| Clip | Frames, same star count | Paired stars, min IoU | Arms, reference → candidate | Within 0.5 px | Steps changed |
|---|---:|---:|---:|---:|---:|
| Sparse crowded | 3 / 3 | 14, 1.000 | 85 → 86 | 84 | 1 |
| ROV 2003 | 16 / 16 | 14, 1.000 | 237 → 237 | 237 | 0 |
| ROV 2004 | 16 / 16 | 10, 1.000 | 45 → 45 | 45 | 0 |
| ROV 2008 | 16 / 16 | 3, 1.000 | 49 → 49 | 49 | 0 |

Both candidate repeats of every clip gave the same agreement, but different bytes. The gate counts reference arms, so the sparse probe's extra candidate arm does not count against it. Compiling in each process raised process wall time by 125–282% (for example 8–10 s to 20–44 s). Round 10 used one secure RTX 5090 lease of 0.47 h (at most $0.55).

## Round 9: OpenCV thread caps, ten processes and relaxed Hydra settings

Round 9 reuses Round 8's image, overlay wheel, weights and workload, on a host with 116 GiB container memory and the same 13.6-core CPU quota. The Runner now records the cgroup's CPU throttling counters around every case. All 11 optimized archive runs produced the Round 4 selected bytes, and the eager reference reproduced Round 4's eager bytes. No case faulted.

| Configuration (all under MPS) | Images/s | Strict gates | Throttled, s |
|---|---:|---|---:|
| Eight processes (validation) | 37.75 | pass | 4.7 |
| Eight processes, one OpenCV thread | 37.69 | pass | 4.5 |
| Ten processes, one OpenCV thread | 39.72 | pass | 15.6 |
| Eight / ten processes, channels-last Hydra, crop batch 4 | — | stopped at the 30 GiB device guard (31.6 / 31.1 GiB) | — |

Throttled time is the cgroup's `throttled_usec` over the whole case. The scheduler took the fastest valid screen into an ABBA against eight processes (A). Order A1, B1, B2, A2; both sides reproduced the Round 4 selected bytes.

| Block | A repeats | B repeats (ten processes, one OpenCV thread) | Ratio of means | Paired |
|---|---:|---:|---:|---:|
| Warmed processing, images/s | 44.335 / 42.792 | 36.921 / 42.403 | −8.96% | −16.72% / −0.91% |
| Zero-warmup processing, images/s | 30.602 / 34.474 | 32.631 / 32.605 | +0.25% | +6.63% / −5.42% |
| Zero-warmup whole process, images/s | 12.923 / 13.959 | 12.814 / 13.168 | −3.35% | −0.85% / −5.67% |

During the timed window, eight processes used 8–10 of the 13.6 cores on average, about 210–225 ms of CPU per image, and the quota still throttled them in bursts. On live replay, one OpenCV thread raised p95 on the sparse crowded probe from 114.7 / 115.8 ms to 143.7 / 138.4 ms (+22.36%). It changed the ROV 2003 clip by −22.26% (paired +3.93% / −39.02%), a ratio dominated by one slow baseline repeat. Round 9 used one secure RTX 5090 lease of 0.48 h (at most $0.55).

## Round 8: ten processes and a compiled GRU step

Round 8 installs `deepstarnoodles_inference-0.5.0.dev4` (sha256 `9c7efc95…`, base commit `3e2eccad`). It adds `--gru-compile`, which captures a `torch.compile`d GRU step in the same CUDA graphs, and allows up to 16 archive processes. The image has no C compiler. Triton needs one C helper module (`cuda_utils`), so the bundle carries a prebuilt copy, pinned by source and binary SHA-256. A probe confirmed the pod compiled through it. Same container memory (57.7 GiB) and 13.6-core CPU quota as Round 7; no case faulted.

| Configuration (all under MPS) | Images/s | Strict gates | Bytes |
|---|---:|---|---|
| Eight processes (Round 7 selection) | 41.95 | pass | Round 4 selected |
| Ten processes | 40.12 | pass | Round 4 selected |
| Eight processes, compiled GRU step | 40.72 | **fail** | differ |

Ten processes did not beat eight, so no archive ABBA followed. The compiled run stayed within the 1 px polyline and 0.01 confidence tolerances: maxima 0.050 px and 0.0007. It still failed the strict gates, which also require equal steps and stop reasons for every arm. In local rehearsals the compiled step also flipped one archive stop decision (an arm 5.6 px shorter). Each process autotunes its reduction kernels, so repeated compiled runs produced different outputs. It is not a selectable candidate.

| Clip (warm live, ABBA) | A p95, ms | B p95 (compiled), ms | Ratio of means | Paired |
|---|---:|---:|---:|---:|
| Sparse crowded, two crop streams | 199.2 / 152.1 | 149.4 / 146.5 | −15.78% | −25.00% / −3.71% |
| ROV A, single crop | 104.3 / 89.4 | 94.9 / 94.1 | −2.46% | −9.07% / +5.26% |

Both live blocks passed the strict gates against their own baselines, but the two compiled repeats of each clip produced different bytes. Round 8 used one secure RTX 5090 lease of 0.41 h (about $0.49).

## Round 7: CUDA MPS, more processes and concurrent GRU streams

Round 7 installs `deepstarnoodles_inference-0.5.0.dev3` (sha256 `9c678be4…`, base commit `3e2eccad`) into the unchanged Round 4 image during preflight. Runtime, weights, workload and strict tolerances are unchanged. The Runner adds three things:

- An optional CUDA MPS control daemon per case, started before the case and stopped after it. A probe first checks that a client connects through it, and each MPS case must show a running MPS server.
- Process-tree CPU sampling alongside the GPU samples.
- One rerun, under a new name, of any case that aborts with a CUDA illegal memory access (see the README's known issue).

No case faulted on the lease. The container reported 128 CPUs but a cgroup quota of **13.6 cores**, and 57.7 GiB of memory.

**Output identity.** The eager reference reproduced Round 4's eager bytes. All 14 optimized archive runs produced the Round 4 selected bytes (`0e5e4e99…`): validations, screens and both ABBA blocks, with and without MPS. The crowded live candidate matched its baseline bytes. All 19 cases passed.

### Screens (single observations, warmup 4)

| Configuration | Images/s | GPU busy | CPU cores used |
|---|---:|---:|---:|
| Four processes (Round 6 selection) | 26.37 | 80% | 6.4 |
| Four processes, four GRU streams each | 27.89 | 71% | 6.5 |
| Four processes, MPS | 31.87 | 64% | 6.5 |
| Six processes, MPS | 34.71 | 70% | 7.0 |
| Eight processes, MPS | 42.25 | 69% | 9.3 |
| Four processes, GRU streams, MPS | 30.07 | 65% | 6.3 |

GPU busy is the mean share of samples with any kernel running during the timed window. CPU cores used are process-tree CPU seconds per wall second over the same window. The scheduler selected eight processes under MPS.

### Archive: four processes (A) versus eight processes under MPS (B)

Order A1, B1, B2, A2.

| Block | A repeats | B repeats | Ratio of means | Paired |
|---|---:|---:|---:|---:|
| Warmed processing, images/s | 26.464 / 27.075 | 43.485 / 44.597 | +64.52% | +64.32% / +64.72% |
| Zero-warmup processing, images/s | 23.793 / 23.747 | 36.077 / 35.328 | +50.20% | +51.63% / +48.77% |
| Zero-warmup whole process, images/s | 11.109 / 11.115 | 12.697 / 12.839 | +14.90% | +14.30% / +15.51% |
| Zero-warmup process wall, s | 23.05 / 23.03 | 20.16 / 19.94 | −12.97% | −12.51% / −13.43% |

Eight processes load their models and warm up before the shared start, so for this 256-image job the whole-process gain is smaller; it grows with archive length. In the warmed blocks, mean board power was about 314 W for A and 435 W for B, against the RTX 5090's 575 W rating. B's sampled device peak was 21,581 MiB with 21.4–21.7 GiB process-tree RSS.

### Live: GRU streams on the crowded probe

A is Round 6's crowded live candidate (two crop streams); B uses four concurrent GRU streams instead. Order A1, B1, B2, A2.

| Clip | A p95, ms | B p95, ms | Ratio of means | Paired |
|---|---:|---:|---:|---:|
| Sparse crowded | 148.6 / 137.2 | 178.6 / 182.6 | +26.40% | +20.22% / +33.10% |

GRU streams do not help live replay; crop streams remain the crowded-scene live option.

Round 7 used one secure RTX 5090 lease of 0.70 h (about $0.84), including a slow container start on this host.

```bash
# Round 7 archive candidate: the Round 6 command with eight processes, under a running MPS daemon.
nvidia-cuda-mps-control -d
deepstarnoodles-inference benchmark --mode archive \
  --models /models/manifest.json --index /inputs/archive.csv \
  --output /outputs/round7-archive --images 256 --warmup 4 \
  --detector-batch-size 1 --crop-batch-size 1 \
  --hydra-output-mode minimal --gru-execution graph --gru-check-interval 1 \
  --hydra-execution graph --detect-ahead 2 --archive-processes 8 \
  --decode-workers 2 --decode-prefetch-frames 4 \
  --decode-prefetch-mib 256 --decode-max-frame-mib 64 \
  --archive-frame-window 0 --archive-crop-buffer-mib 256 \
  --allocator-limit-gib 7 --ram-limit-gib 24
echo quit | nvidia-cuda-mps-control
```

## Validation accuracy of the optimized profiles

The released tracer (`e6c15060…`) was rescored on the 478-image annotated validation split: 180 split groups and 4,801 annotated arms. Scoring follows `SCRIPTS/evaluate_tracer_split.py` in the main repository: arms are matched one to one at a symmetric Chamfer distance of at most 8 crop pixels, with pooled F1 and group-macro F1. The recorded validation of this checkpoint ran on node1's RTX 5090 on 28 October. Its frozen field cache was not on this host, so `cache_fixed_tracer_split.py` rebuilt it from the same Hydra checkpoint, index, images and annotations. All 478 cache identities match the recorded manifest. The rerun used an RTX A6000 with PyTorch 2.6.0+cu124.

| Path | F1 | Group-macro F1 | Precision | Recall | TP / FP / FN |
|---|---:|---:|---:|---:|---:|
| Recorded validation (RTX 5090 fields) | 0.8034 | 0.7507 | 0.8364 | 0.7730 | 3,711 / 726 / 1,090 |
| Rebuilt cache, each of six decoders | 0.8032 | 0.7506 | 0.8365 | 0.7725 | 3,709 / 725 / 1,092 |
| End to end: reference, Round 4 and Round 5–6 profiles | 0.8034 | 0.7508 | 0.8365 | 0.7728 | 3,710 / 725 / 1,091 |

**Decoders on the rebuilt cache.** Six decoders scored the same fields:

- CORE's `trace_arms_noodle_batch` with four images per call (the recorded procedure)
- the same with one image per call
- the package's vendored copy (the campaign reference profile)
- fixed slots with check interval 1 (Round 4)
- the CUDA-graph rollout with device peak candidates (Rounds 5–6)
- four concurrent GRU streams (`--gru-streams 4`)

All but the first ran one crop per call, as deployed. All six gave the same match counts on every image, and their paired group-bootstrap F1 difference is 0 with 95% interval [0, 0]. Fixed-slot, CUDA-graph and concurrent-stream arms were byte-identical on all 478 images, as were the CORE and vendored dynamic decoders. Fixed-slot and dynamic arms differ by at most 0.0086 crop pixels, and four-image versus single-crop calls by at most 0.023 pixels; both differences are floating-point rounding.

**End to end.** The package's Hydra recomputed the fields from the cache builder's exact inputs. At the cache's batch of four it reproduced the stored float16 fields bit for bit on all 478 images. At one crop per forward, CUDA-graph replay matched eager Hydra bit for bit on all 478 images. Each profile then decoded its own fields, using the Hydra input as the tracer's RGB as the cascade does, and the three profiles scored identically, with zero F1 difference in all 2,000 group resamples.

**No drift attributable to the optimizations.** The three paths differ by at most two matched arms out of 4,801. Rounding on a different GPU and software stack (in both the Hydra fields and decoding), batch size and the cache's 8-bit RGB storage explain that spread. No decoder or launch option contributes to it. Single timing observations on the shared A6000, decoding all 478 crops: dynamic 36.0 s, fixed slots 32.4 s, CUDA graph 8.5 s and four GRU streams 6.2 s.

## Round 6: whole-GPU archive processes and concurrent crop streams

Round 6 is the whole-GPU phase. It installs `deepstarnoodles_inference-0.5.0.dev2` (sha256 `3a6642a2…`, base commit `3e2eccad`) into the unchanged Round 4 image during preflight. Runtime, weights, workload and strict tolerances are those of Round 5. Each process keeps the 7 GiB allocator and 24 GiB RSS limits. The sampled guards were raised to 30 GiB device memory and 64 GiB process-tree RSS for the whole GPU. Two options are new:

- `--archive-processes N` shards source frames round-robin across N resident processes on one GPU, each running the Round 5 archive candidate (G). It merges their records back into input order.
- `--crop-streams 2` runs a live frame's crop groups concurrently on two CUDA streams with separate Hydra/GRU graph state.

**Output identity.** The eager reference reproduced Round 4's eager bytes (`88f86553…`). The G validation, the 2-, 3-, 4- and 6-process screens and both archive ABBA blocks all produced the Round 4 selected bytes (`0e5e4e99…`; 454 crops, 4,554 arms). The crowded live candidate matched its baseline bytes. All 18 cases passed with no deferrals.

### Process-count screen (single observations, warmup 4)

| Processes | Images/s | Sampled device peak, MiB | Process-tree RSS, GiB |
|---:|---:|---:|---:|
| 1 (G validation) | 11.35 | 2,710 | 3.09 |
| 2 | 16.34 | 5,409 | 6.33 |
| 3 | 22.35 | 8,108 | 9.31 |
| 4 | 26.12 | 10,804 | 11.88 |
| 6 | 26.05 | 16,250 | 16.49 |

Four processes were selected; six added memory without adding throughput.

### Archive: Round 5 G (A) versus four processes (B)

B is G with `--archive-processes 4`. Order A1, B1, B2, A2.

| Block | A repeats | B repeats | Ratio of means | Paired |
|---|---:|---:|---:|---:|
| Warmed processing, images/s | 10.348 / 10.350 | 25.426 / 25.825 | +147.62% | +145.72% / +149.53% |
| Zero-warmup processing, images/s | 9.944 / 9.709 | 22.762 / 22.605 | +130.83% | +128.89% / +132.81% |
| Zero-warmup whole process, images/s | 7.479 / 7.562 | 10.366 / 10.608 | +39.45% | +38.61% / +40.29% |
| Zero-warmup process wall, s | 34.23 / 33.85 | 24.70 / 24.13 | −28.28% | −27.85% / −28.72% |

Process wall also includes each process's imports and model loading, the record merge and exit. Outside timed processing, these took about 13 s with four processes against about 8 s with one, so the whole-process gain grows with archive length.

### Live: two crop streams on the sparse crowded probe

A is the Round 5 live candidate L; B adds `--crop-streams 2`. The probe's three frames (14 crops) replay after a three-frame warmup. Order A1, B1, B2, A2.

| Clip | A p95, ms | B p95, ms | Ratio of means | Paired |
|---|---:|---:|---:|---:|
| Sparse crowded | 250.9 / 239.9 | 166.0 / 163.7 | −32.82% | −33.85% / −31.74% |

The ROV clips are single-star, with at most one crop per frame, so crop streams cannot overlap anything there. They were not rerun.

### Memory, faults and lease

B's sampled device peak was 10,806 MiB with 11.9–12.1 GiB process-tree RSS, against 2,710 MiB and 3.1 GiB for A. The six-process screen peaked at 16,250 MiB and 16.5 GiB. No case faulted on the lease. Locally, multi-process detect-ahead runs occasionally abort with a CUDA illegal memory access, about 1 in 48 processes (see the README's known issue); run multi-process jobs under a retry. Round 6 used one secure RTX 5090 lease of 0.32 h (about $0.38).

```bash
# Round 6 archive candidate (B): the Round 5 G command plus four processes.
deepstarnoodles-inference benchmark --mode archive \
  --models /models/manifest.json --index /inputs/archive.csv \
  --output /outputs/round6-archive --images 256 --warmup 4 \
  --detector-batch-size 1 --crop-batch-size 1 \
  --hydra-output-mode minimal --gru-execution graph --gru-check-interval 1 \
  --hydra-execution graph --detect-ahead 2 --archive-processes 4 \
  --decode-workers 2 --decode-prefetch-frames 4 \
  --decode-prefetch-mib 256 --decode-max-frame-mib 64 \
  --archive-frame-window 0 --archive-crop-buffer-mib 256 \
  --allocator-limit-gib 7 --ram-limit-gib 24

# Round 6 crowded live candidate: the Round 5 live command plus two crop streams.
deepstarnoodles-inference live-replay \
  --models /models/manifest.json --index /inputs/clip.csv \
  --output /outputs/round6-crowded-live --warmup 3 \
  --detector-batch-size 1 --crop-batch-size 1 \
  --hydra-output-mode minimal --gru-execution graph --gru-check-interval 1 \
  --hydra-execution graph --crop-streams 2 --decode-workers 0 \
  --allocator-limit-gib 7 --ram-limit-gib 24
```

## Round 5: CUDA-graph replay, device peak candidates and detect-ahead

Round 5 installs a hash-pinned overlay wheel, `deepstarnoodles_inference-0.5.0.dev1` (sha256 `057a167a…`, base commit `3e2eccad`), into the unchanged Round 4 image during preflight. Image, runtime (PyTorch 2.8.0+cu128, cuDNN 91002, Ultralytics 8.4.51), weights, the 256-image archive and the five clips are those of Round 4. Detector and crop batches stay 1, Hydra outputs stay minimal, and precision, thresholds and strict tolerances are unchanged. The three new options change only how the same kernels are launched and scheduled:

- `--gru-execution graph` replays the fixed-slot rollout from CUDA graphs captured per walker count. It also samples each image's dense patches with one `grid_sample` call over that image. In the reference, the expanded 18-channel field made cuDNN's grid sampler copy the whole field once per walker on every step — about 80% of step GPU time at 37 walkers on an RTX A6000. Tip/base peak candidates and seed windows are gathered on the GPU; ordering, suppression and window means stay in NumPy on identical float32 values.
- `--hydra-execution graph` replays the Hydra forward at its fixed 512-pixel shape.
- `--detect-ahead 2` runs decode, StarSeg and crop preparation for the next source group on a second CUDA stream while Hydra and the GRU run in input order.

**Runtime continuity and output identity.** On this lease the eager reference reproduced Round 4's eager predictions byte for byte (`88f86553…`), and the Round 4 minimal/fixed1 profile reproduced Round 4's selected predictions (`0e5e4e99…`). Every Round 5 archive run — validation, both ABBA blocks and the attribution runs — produced those same `0e5e4e99…` bytes, and every live block's candidate matched its baseline bytes. All 33 cases passed with no deferrals.

### Archive: Round 4 selection (D) versus Round 5 (G)

D is minimal outputs, fixed GRU with check interval 1 and two decode workers. G adds `--gru-execution graph --hydra-execution graph --detect-ahead 2`. Order D1, G1, G2, D2.

| Block | D repeats | G repeats | Ratio of means | Paired |
|---|---:|---:|---:|---:|
| Warmed processing, images/s | 4.720 / 4.491 | 9.052 / 10.595 | +113.29% | +91.79% / +135.89% |
| Zero-warmup processing, images/s | 4.339 / 4.191 | 9.774 / 9.643 | +127.62% | +125.26% / +130.07% |
| Zero-warmup whole process, images/s | 3.834 / 3.717 | 7.631 / 7.594 | +101.64% | +99.04% / +104.33% |
| Zero-warmup process wall, s | 66.78 / 68.88 | 33.55 / 33.71 | −50.42% | −49.76% / −51.06% |

Single attribution observations (warmup 4, not paired claims): the Round 4 profile validated at 4.53 images/s; GRU graph replay with device peaks reached 8.14; adding Hydra graph replay reached 8.94; adding detect-ahead (the G validation) reached 10.51. In that G validation the Hydra/GRU consumer waited about 10.6 s of the roughly 24-second measurement for the detect-ahead producer, so StarSeg and crop preparation now limit single-process archive throughput on this host.

### Warm live replay: Round 4 live candidate (B) versus Round 5 (L)

L adds GRU and Hydra graph replay to B; decoding stays synchronous and detect-ahead is not used. Each clip warms its whole clip, resets tracking and replays it; order B1, L1, L2, B2. Negative changes mean lower per-run p95.

| Clip | B p95, ms | L p95, ms | Ratio of means | Paired |
|---|---:|---:|---:|---:|
| ROV A | 183.2 / 218.5 | 84.5 / 90.1 | −56.54% | −53.90% / −58.76% |
| ROV B | 151.8 / 177.0 | 55.4 / 57.2 | −65.74% | −63.49% / −67.68% |
| ROV C | 153.9 / 138.0 | 73.5 / 77.0 | −48.42% | −52.20% / −44.21% |
| Negative ROV (detector only) | 28.6 / 23.3 | 22.9 / 28.6 | −0.69% | −19.80% / +22.79% |
| Sparse crowded | 699.9 / 599.8 | 202.4 / 210.7 | −68.21% | −71.08% / −64.87% |

The negative clip calls neither Hydra nor the GRU, so its changes are timing variation. These remain warm-cache short replays, not sustained camera latency.

### Memory and lease

G's sampled device peak was 2,710 MiB against 4,004 MiB for D (PyTorch reserved 1,910 versus 3,336 MiB). All-case sampled peaks were 4,344 MiB device memory and 3.09 GiB process-tree RSS, under the unchanged 7 GiB allocator and sampled 7 GiB device / 24 GiB RSS guards. Round 5 used one secure RTX 5090 lease of 0.51 h (about $0.61). As before, these measurements do not establish RTX 5060 performance or certify physical 8 GiB fit.

```bash
# Round 5 archive candidate (G); use --warmup 0 for the whole-job comparison.
deepstarnoodles-inference benchmark --mode archive \
  --models /models/manifest.json --index /inputs/archive.csv \
  --output /outputs/round5-archive --images 256 --warmup 4 \
  --detector-batch-size 1 --crop-batch-size 1 \
  --hydra-output-mode minimal --gru-execution graph --gru-check-interval 1 \
  --hydra-execution graph --detect-ahead 2 \
  --decode-workers 2 --decode-prefetch-frames 4 \
  --decode-prefetch-mib 256 --decode-max-frame-mib 64 \
  --archive-frame-window 0 --archive-crop-buffer-mib 256 \
  --allocator-limit-gib 7 --ram-limit-gib 24

# Round 5 live candidate (L); --warmup 3 for the three-frame probe.
deepstarnoodles-inference live-replay \
  --models /models/manifest.json --index /inputs/clip.csv \
  --output /outputs/round5-warm-live --warmup 16 \
  --detector-batch-size 1 --crop-batch-size 1 \
  --hydra-output-mode minimal --gru-execution graph --gru-check-interval 1 \
  --hydra-execution graph --decode-workers 0 --allocator-limit-gib 7 --ram-limit-gib 24
```

**Rounds 1–4 are complete:** the pilot, Round 2, Round 3 and Round 4 contain **43, 112, 80 and 30 recovered, hash-verified cases**. Round 4 has no deferred cases, 24 strict parity passes and six uncompared references; both archive blocks and all five warmed live blocks qualified. Its final evidence was verified on 2026-10-29 at 01:52:57 UTC.

## Round 4: combined archive configuration and warmed live verification

The source and container are identical to Round 3. Detector/crop batch stays **1**; default Hydra batching therefore executes one crop at a time. Weights, precision/layout settings, thresholds, seed rules and tolerances are unchanged. Decode prefetch retains at most four frames within a 256 MiB decoded-array budget, with a 64 MiB maximum frame. Cross-frame crop buffering and staged execution remain disabled.

### Added profile: decode workers held fixed at two

C uses full Hydra outputs and dynamic GRU; D uses minimal outputs and fixed GRU with check interval 1. Each warms four frames (eight crops and 78 arms), then measures the same 256-image workload. Chronological order is C1, D1, D2, C2.

| Order | Profile | Processing images/s | Process wall, s | Benchmark job, s |
|---|---|---:|---:|---:|
| C1 | Full / dynamic | 4.431 | 67.909 | 63.745 |
| D1 | Minimal / fixed1 | 4.730 | 64.197 | 60.416 |
| D2 | Minimal / fixed1 | 4.761 | 63.708 | 59.945 |
| C2 | Full / dynamic | 4.442 | 66.701 | 62.944 |

The added profile improves processing throughput **6.97%** by ratio of means, paired **+6.76% / +7.18%**. Candidate rates span **4.730–4.761 images/s**, versus **4.431–4.442** for the control. Process wall improves **4.98%** and benchmark-job time **4.99%** by ratio of means. This isolates the added profile while prefetch stays enabled; it does not measure prefetch alone.

### Combined archive configuration: zero-warmup whole jobs

A is full/dynamic with sequential decoding. D combines minimal/fixed1 with two decode workers. All runs measure the same 256 inputs with zero warmup, ordered A1, D1, D2, A2.

| Order | Configuration | Process wall, s | Whole-process images/s | Benchmark job, s |
|---|---|---:|---:|---:|
| A1 | Eager / decode0 | 83.921 | 3.050 | 80.001 |
| D1 | Combined / decode2 | 63.230 | 4.049 | 59.267 |
| D2 | Combined / decode2 | 68.669 | 3.728 | 64.731 |
| A2 | Eager / decode0 | 83.935 | 3.050 | 80.052 |

Whole-process throughput improves **27.48%** by ratio of means, paired **+32.72% / +22.23%**. Process wall changes **−21.42%**, paired **−24.65% / −18.19%**; benchmark-job time changes **−22.53%**, paired **−25.92% / −19.14%**. Combined rates span **3.728–4.049 whole-process images/s**, versus approximately **3.050** in both eager repeats. Two-repeat ranges are descriptive, not confidence intervals.

Every archive run produces **454 crops and 4,554 arms** and passes unchanged gates when compared with the reference. Outputs repeat byte-identically within each configuration; fixed versus dynamic outputs differ within tolerance. The eager output matches prior rounds byte for byte. Each archive repeat peaks at **4,004 MiB sampled device memory**; the largest tree RSS across these blocks is **3.12 GiB**. The initial unpaired no-prefetch singleton validation was slightly slower than the first eager observation (3.484 versus 3.576 images/s); the recommendation rests on the repeated comparisons above.

### Warmed live replay: positive clips improve, negative control varies

Both A (full/dynamic) and B (minimal/fixed1) warm the **entire clip**, then reset tracking and replay it: 16 frames per ROV clip and three for the sparse crowded probe. Decode stays synchronous and detector/crop batch stays 1. Each clip has its own A1, B1, B2, A2 block. Negative percentages mean lower per-run p95.

| Clip | Reference p95 A1 / A2, ms | Candidate p95 B1 / B2, ms | Mean-ratio change | Paired changes |
|---|---:|---:|---:|---:|
| ROV A | 204.7 / 192.5 | 173.4 / 154.9 | -17.35% | -15.28% / -19.56% |
| ROV B | 174.2 / 200.0 | 156.9 / 152.2 | -17.42% | -9.94% / -23.94% |
| ROV C | 147.9 / 178.7 | 135.7 / 144.3 | -14.28% | -8.23% / -19.28% |
| Negative ROV | 23.9 / 22.6 | 22.7 / 38.6 | +31.71% | -5.21% / +70.79% |
| Sparse crowded | 620.8 / 702.6 | 581.2 / 648.2 | -7.11% | -6.37% / -7.75% |

All five blocks pass strict parity. Positive ROV clips detect **14, 10 and three crops** (237, 45 and 49 arms); the crowded probe detects **6 / 4 / 4 animals** in its three frames (85 arms). Warmup exercises the same detections and arms as the timed pass. Each configuration repeats byte-identically. The negative clip has no detections and calls neither Hydra nor GRU; its **−5.21% / +70.79%** paired changes demonstrate timing variation and cannot be credited to their altered execution.

The positive clips support minimal/fixed1 as the next live configuration to validate, while the negative result prevents a uniform speedup claim. These are warm-cache, short, same-shape replays on a 5090, with only three crowded frames. They do not measure sustained camera input, queue backlog, dropped frames, or a stable long-run tail. Earlier-round differences cannot isolate warmup as a cause because host and profile also differ.

## Round 3: decode prefetch with unchanged model execution

A uses sequential decoding; B uses **two decode workers**, four prefetched frames, a 256 MiB retained decoded-array budget and a 64 MiB maximum frame. Both keep detector/crop batch 1, full Hydra outputs, dynamic GRU, default Hydra batching and unchanged precision/layout. No crop buffer is enabled. Execution order is A1, B1, B2, A2; all runs use the same 256 inputs.

### Warmed processing: four warmup frames

| Order | Decode workers | Processing images/s | Process wall, s | Benchmark job, s |
|---|---:|---:|---:|---:|
| A1 | 0 | 3.229 | 89.306 | 85.731 |
| B1 | 2 | 4.520 | 64.578 | 61.029 |
| B2 | 2 | 5.556 | 53.567 | 50.148 |
| A2 | 0 | 3.679 | 77.086 | 73.698 |

Processing throughput’s B/A ratio of means is **1.4586 (+45.86%)**; paired ratios are **1.4000 / 1.5101**. Reference repeats span **3.229–3.679 images/s** and candidate repeats **4.520–5.556**. This variation is material: two-repeat ranges are descriptive, not confidence intervals. Process-wall and benchmark-job ratios of means are 0.7100 (−29.00%) and 0.6973 (−30.27%), but these timings include four warmup images.

### Whole jobs: zero warmup

| Order | Decode workers | Process wall, s | Whole-process images/s | Benchmark job, s |
|---|---:|---:|---:|---:|
| A1 | 0 | 69.136 | 3.703 | 65.177 |
| B1 | 2 | 53.244 | 4.808 | 50.486 |
| B2 | 2 | 59.566 | 4.298 | 55.522 |
| A2 | 0 | 66.055 | 3.876 | 62.685 |

Whole-process throughput’s B/A ratio of means is **1.2016 (+20.16%)**, with paired gains **+29.85% / +10.89%**. Process wall is **−16.56%** by ratio of means, with paired changes **−22.99% / −9.82%**; benchmark-job time is **−17.09%**, paired **−22.54% / −11.43%**. Both pairs favor prefetch, while the spread limits precision. The warmed and zero-warmup effects should not be collapsed into one speedup percentage.

All eight archive repeat files are byte-identical to their same-build reference and contain 454 detected crops and 4,554 arms. Sampled device peaks stay at 4,004 MiB. The warmed block’s largest tree RSS is 3.14 GiB; the zero-warmup block’s is 3.10 GiB. Single full 256-frame promotions also passed with one and two workers, but only two workers received ABBA confirmation. This does not prove two workers are faster than one.

The prefetch reader preserves input order and bounds retained decoded arrays. In the full 256-frame promotion it peaked at four pending frames and 110.84 MiB reserved decoded arrays. Worker decode overlaps inference; worker elapsed time is not additive to consumer-stage timings or guaranteed time saved. Filesystem-cache and storage limits described below still apply.

### Live replay: parity passed, latency mixed

The Round 3 live candidate uses minimal Hydra outputs, Hydra batch 1, dynamic GRU and crop batch 1, with sequential decoding and zero warmup. A uses full outputs and default Hydra batching; the singleton crop keeps effective Hydra execution at one. This is distinct from Round 2’s fixed-GRU candidate. Each clip gets its own A1, B1, B2, A2 block and tracking reset. Negative percentage means lower p95 latency.

| Clip | Reference p95 A1 / A2, ms | Candidate p95 B1 / B2, ms | Mean-ratio change | Paired changes |
|---|---:|---:|---:|---:|
| ROV A | 569.4 / 485.7 | 559.3 / 380.5 | -10.93% | -1.78% / -21.65% |
| ROV B | 439.4 / 762.0 | 453.3 / 638.9 | -9.09% | +3.16% / -16.15% |
| ROV C | 506.6 / 538.4 | 409.5 / 578.0 | -5.50% | -19.16% / +7.35% |
| Negative ROV | 127.0 / 153.2 | 103.7 / 106.1 | -25.12% | -18.35% / -30.74% |
| Sparse crowded | 1308.8 / 1700.1 | 1936.0 / 1542.1 | +15.59% | +47.92% / -9.29% |

All five blocks passed the unchanged gates, and each block’s four outputs are byte-identical. ROV B and C each have one regression; the sparse crowded mean p95 is **15.59% slower**, with paired changes **+47.92% / −9.29%**. The negative clip invokes neither Hydra nor GRU, so its apparent improvement cannot be attributed to them. These are per-run short-replay p95 estimates, not pooled percentiles or sustained-camera latency. No broad live speedup is established.

### Screen rejections

Holding Hydra at batch 1 did not make crop/GRU batches 2/4/8 equivalent: the 64-frame dynamic screens still failed strict parity. Fixed GRU with check interval 1 at crop batch 2 and minimal outputs at crop batch 2 also failed. Buffered crop batches 8/16/32 and staged batches 8/16/32 each failed, so neither schedule was promoted. All 11 archive rejections recorded ten parity events apiece: confidence, length, point-count and steps; boxes and compared identities matched. Actual buffered groups reached the requested 8/16/32 sizes within their configured queue limits. Buffered ingestion p95 is not frame-completion latency and is not used for promotion claims.

A separate matched-seed diagnostic on the verified 64-frame outputs localizes the material difference to the **same two arms taking one extra step** across grouped dynamic crop batches 2/4/8 and the minimal-output/fixed-slot crop-2 variants. All 110 crops, 1,103 seed identities and tip scores match. This is consistent with batch-shape numerical sensitivity amplified by a discrete stopping decision, but intermediate logits were not captured and the first divergent kernel is not established. The strict gates remain unchanged and these grouped cases remain rejected. The numeric evidence records the diagnostic digest and its correspondence to the final recovered case hashes.

The crowded live probe failed at crop batches 2/4/8 even with Hydra batch 1, with 17 events each including arm tracking IDs. Singleton full/dynamic Hydra controls are baseline-equivalent and were excluded from optimization selection. Prefix screens and their fastest observations are not the report’s main conclusion.

Round 4 completed the same-build singleton combination and warmed-live follow-up; its repeated results and remaining limits appear above.

## Round 2: repeated resident measurements

The reference uses full Hydra outputs and dynamic GRU; the candidate uses minimal outputs, fixed GRU and check interval 1. Both use detector/crop batch 1 with identical weights and precision.

Each run processes the first four inputs once as warmup, then times all 256 inputs, including those same four frames. A is the reference and B the candidate; execution order is A1, B1, B2, A2.

| Execution order | Configuration | Processing images/s | Child-process seconds | Child-process images/s | Benchmark job seconds | Device peak MiB | RSS peak GiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | Reference | 3.584 | 80.318 | 3.187 | 76.838 | 4004 | 3.06 |
| B1 | Candidate | 3.733 | 77.044 | 3.323 | 73.978 | 4004 | 2.97 |
| B2 | Candidate | 3.759 | 76.591 | 3.342 | 73.345 | 4004 | 3.07 |
| A2 | Reference | 3.559 | 80.756 | 3.170 | 77.452 | 4004 | 3.05 |

The two within-pair processing-rate changes are **+4.16% / +5.61%**. Child-process rates changed **+4.25% / +5.44%**, with process time **−4.08% / −5.16%**. The candidate’s two processing rates span 3.733–3.759 images/s, versus 3.559–3.584 for the reference. These two repeats describe observed variation; they are not confidence intervals. The [numeric evidence](BENCHMARK_EVIDENCE.json) retains unrounded paired observations and verified-manifest hashes. Every repeat passed parity and produced 454 detected star crops and 4,554 arm records. Predictions were byte-identical within each configuration’s repeats; candidate/reference files differ but satisfy the numeric and structural gates.

Three singleton-crop configurations reached full-256 validation: minimal outputs with fixed GRU check intervals 1, 4 and 8. Their initial processing rates were 3.747, 3.621 and 3.614 images/s, respectively, and all passed. Only interval 1 received the repeated comparison above. The singleton reference remains the fallback where a candidate lacks a demonstrated benefit.

**Timing definitions matter.** Processing images/s covers the timed inference loop, excluding initialization and warmup. “Benchmark job seconds” runs from the benchmark’s main function to its final timing sample, including model initialization, warmup, inference and prediction output (and staged spool cleanup); earlier Python/module imports and final report serialization are excluded. “Child-process seconds” is the monitor’s launch-to-observed-exit time, including startup and polling overhead. Its rate is 256 divided by those seconds; for the table above, the wall time also includes four warmup frames. It is therefore not a zero-warmup job comparison.

## Round 2 and pilot: staged versus resident whole jobs

The next table uses the selected candidate with **zero warmup**, the same 256 frames, and detector/crop batch 1. A is resident execution; B stages 64-frame shards through the models. Both pairs passed parity.

| Execution order | Schedule | Child-process seconds | Whole-process images/s | Benchmark job seconds | Device peak MiB | RSS peak GiB |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | Resident | 76.006 | 3.368 | 72.998 | 4004 | 2.83 |
| B1 | Staged, 64 frames | 88.759 | 2.884 | 85.606 | 4228 | 3.40 |
| B2 | Staged, 64 frames | 86.564 | 2.957 | 83.313 | 4228 | 3.41 |
| A2 | Resident | 75.987 | 3.369 | 72.916 | 4004 | 2.98 |

Staging reduced whole-process throughput by **14.37% / 12.22%** and increased wall time by **16.78% / 13.92%**. It also had higher sampled device memory here. The resident schedule is the better measured throughput choice for this workload. Processing-only rates are intentionally omitted: staging’s model reload and spool costs make those rates a different comparison scope. Each schedule’s two output files were byte-identical; staged/resident outputs passed parity without being byte-identical. Optional Round 2 shard-32 and shard-128 runs also passed: they took 99.317 / 84.011 process seconds (2.578 / 3.047 images/s), with sampled device peaks 3,058 / 3,706 MiB and RSS peaks 3.44 / 3.43 GiB. Those sizes have one observation each, not paired confirmation.

The completed pilot also tested shard sizes 32/64/128, each once with the original reference and zero warmup:

| Pilot schedule | Process seconds | Whole-process images/s | Sampled device peak MiB | Sampled RSS GiB |
| --- | ---: | ---: | ---: | ---: |
| Resident | 80.179 | 3.193 | 4,004 | 3.00 |
| Staged, 32 frames | 99.675 | 2.568 | 3,058 | 3.43 |
| Staged, 64 frames | 93.590 | 2.735 | 4,228 | 3.40 |
| Staged, 128 frames | 88.454 | 2.894 | 3,706 | 3.39 |

All four passed parity. The 32-frame schedule traded 23.6% lower sampled device peak for 19.6% lower whole-process throughput. Each staged job wrote about 12.38 GiB cumulatively; peak live spool was 1.35/2.48/4.78 GiB for shards 32/64/128, then cleaned to zero. This is a memory/throughput tradeoff, not a throughput win. Do not compare pilot and Round 2 timings as a cross-build speedup.

## Round 2: repeated per-frame latency

Each clip starts a fresh process and tracking state, with zero warmup. The same singleton-crop candidate passed every frame of all five clips in both repeats. Pair 1 runs reference then candidate; pair 2 reverses the order. Rates below are measured processing rates; p95 is observed frame-completion latency within each short replay. Negative percentage means lower latency.

| Clip | Pair | Reference p95 ms | Candidate p95 ms | p95 change | Reference processing images/s | Candidate processing images/s |
| --- | --- | --- | --- | --- | --- | --- |
| ROV A | 1 | 501.7 | 491.6 | -2.01% | 5.245 | 5.675 |
| ROV A | 2 | 513.3 | 495.1 | -3.54% | 5.108 | 5.560 |
| ROV B | 1 | 508.8 | 482.6 | -5.14% | 5.906 | 6.387 |
| ROV B | 2 | 594.7 | 506.1 | -14.90% | 5.574 | 6.194 |
| ROV C | 1 | 494.1 | 502.6 | +1.71% | 8.964 | 9.506 |
| ROV C | 2 | 519.9 | 512.6 | -1.39% | 8.824 | 9.411 |
| Negative ROV | 1 | 169.6 | 141.3 | -16.67% | 16.845 | 19.099 |
| Negative ROV | 2 | 185.4 | 138.1 | -25.50% | 15.967 | 19.165 |
| Sparse crowded | 1 | 1789.1 | 1484.0 | -17.06% | 1.039 | 1.199 |
| Sparse crowded | 2 | 1576.2 | 1649.9 | +4.67% | 1.133 | 1.131 |

ROV A/B/C each contain 16 frames at 224×224, with 14/10/3 detected-positive frames and at most one detected star per frame. The negative ROV clip contains 16 frames and runs no Hydra/GRU inference; its apparent improvement cannot be attributed to those models. The sparse crowded clip contains only three 960×600 frames, with 6, 4 and 4 detected stars. It is not a continuous camera stream. Every configuration’s predictions were byte-identical across its two repeats.

ROV C and the crowded clip each have one latency regression. The crowded processing-rate change is +15.41% in pair 1 and −0.22% in pair 2. These tiny replays are useful for checking behavior and exposing regressions, but do not establish sustained FPS, queue behavior, dropped frames, capture-to-display latency, or a reliable latency tail. The live result remains workload-specific; retain the reference for unproven conditions. Earlier single-screen latency improvements are not used as the repeated conclusion.

## Why larger batches were not selected

In the completed pilot, crop batches 2/4/8 reached 4.005/3.913/4.342 warmed images/s but failed parity with 129/92/142 issue events. Their detector boxes and raster masks matched, while downstream arm counts, trace steps, confidence, identities or geometry changed. Faster execution with changed outputs does not qualify as an equivalent inference optimization. Issue counts are events, not failed-frame counts.

Round 2’s 64-frame archive screen had four passing singleton-crop variants, 20 parity-failing batched variants, and eight detector-batch-16 variants stopped by the sampled device-memory guard. The latter reached 7,568 MiB, above the 7 GiB target. Four crop-batch-4 live variants passed the single-star ROV clips but failed the crowded probe; those ROV passes never exercised a crop batch above one. No failed batch was promoted. The complete pilot likewise promoted none of its eight archive alternatives and no live alternative across all five probes.

The unchanged gate requires structural agreement, including counts, trace steps and stopping reasons; minimum mask IoU 0.99; box/polyline/length differences within 1 pixel in their specified coordinate spaces; identity/confidence differences within 0.01; and source-mapping difference within 0.01 pixel. This is runtime parity relative to the pinned eager reference, not a measure of biological accuracy.

## Workload and memory limits

The archive has 256 unique frames from 14 sources and 82 capture groups: 136 lab, 60 field and 60 web images, stratified across domain and crowding rather than weighted to production frequency. Annotations mark 177 positive and 79 negative frames, 426 animals and a maximum of 16 per frame; the reference actually detects 454 crops, at most nine in a frame. Native widths range from 224 to 5,496 and heights from 224 to 3,672. The archive and live workloads together have 291 unique images; 32 live images also occur in the archive.

All inputs come from StarSeg validation. Known Hydra test source/group matches were excluded, but the archive includes 93 Hydra training matches, one train-and-validation match, 19 validation matches and 143 unverified cases. An absent match does not prove independence. These data support runtime comparison, not accuracy or generalization claims.

Archive image files total 598.65 MiB and fit in host RAM. Hashing inputs before inference warms the filesystem cache. This is not a cold-disk, remote-storage or uncached large-archive benchmark. Output flushing does not include fsync, so the reported rate is not durable-storage throughput.

All 30 Round 4 cases have verified resource samples: **1,741 samples**, no sampling errors, missing device/RSS observations, guard exceedances or stop reasons. All-case sampled peaks are **4,344 MiB device and 3.115 GiB tree RSS**. The maximum sampling gap is **4.329 seconds**, in one archive profile candidate run; this is materially longer than earlier rounds and further limits peak-memory assurance. Sampling does not prove the absence of brief higher peaks.

All 80 Round 3 cases have verified resource samples: **2,783 samples**, no sampling errors, missing device/RSS observations, guard exceedances or stop reasons. The maximum observed gap is 0.659 s; all-case peaks are 4,344 MiB device and 3.934 GiB process-tree RSS. These are sampled peaks, not hard upper bounds.

All 112 recovered Round 2 cases have verified resource samples: 4,342 samples, no missing device observations or sampling-error entries, and eight guard stops. The maximum observed sample gap is 0.911 seconds. The selected archive repeats stayed at 4,004 MiB sampled device peak and about 2.97–3.07 GiB sampled process-tree RSS; accepted live repeats reached 4,344 MiB and about 2.62 GiB. The eight rejected large-detector batches raised the all-case device maximum to 7,568 MiB. All-case RSS peaked at 3.44 GiB. The completed pilot had 3,159 samples, no guard stops, a 4,602 MiB device maximum and 3.43 GiB RSS maximum.

The campaign combines a 7 GiB PyTorch allocator limit with sampled 7 GiB whole-device and 24 GiB process-tree RSS guards. Sampling can miss brief peaks, and an allocator cap does not constrain all GPU allocations. These guards leave the RTX 5090’s compute, bandwidth and physical memory available; they are neither an RTX 5060 emulator nor a hard 32 GiB host-memory limit. Verification on the actual 8 GiB target and sustained representative live inputs is still required.

## Reproduce the final Round 4 configuration

Use the immutable Round 4 image in the build table and the same model manifest, image bytes and input order for each comparison. Set OMP_NUM_THREADS=8, MKL_NUM_THREADS=8 and OPENBLAS_NUM_THREADS=1. Choose a fresh output directory each time.

```bash
# Combined archive candidate: zero-warmup whole job.
deepstarnoodles-inference benchmark --mode archive \
  --models /models/manifest.json --index /inputs/archive.csv \
  --output /outputs/round4-archive --images 256 --warmup 0 \
  --detector-batch-size 1 --crop-batch-size 1 \
  --hydra-output-mode minimal --gru-execution fixed --gru-check-interval 1 \
  --decode-workers 2 --decode-prefetch-frames 4 \
  --decode-prefetch-mib 256 --decode-max-frame-mib 64 \
  --archive-frame-window 0 --archive-crop-buffer-mib 256 \
  --allocator-limit-gib 7 --ram-limit-gib 24

# Warmed live replay candidate; use --warmup 3 for a three-frame probe.
deepstarnoodles-inference live-replay \
  --models /models/manifest.json --index /inputs/clip.csv \
  --output /outputs/round4-warm-live --warmup 16 \
  --detector-batch-size 1 --crop-batch-size 1 \
  --hydra-output-mode minimal --gru-execution fixed --gru-check-interval 1 \
  --decode-workers 0 --allocator-limit-gib 7 --ram-limit-gib 24
```

For the archive whole-job reference, use full Hydra outputs, dynamic GRU, check interval 8 and decode workers 0. To isolate the added profile, keep decode workers 2 on both sides and use warmup 4. For the live reference, change only the three profile flags to full/dynamic/8; keep full-clip warmup and synchronous decode. Default Hydra batching is unchanged. Use the public compare command against a same-build reference before adopting a candidate. Private benchmark images are not distributed; use representative local inputs for target-machine validation.

## Reproduce Round 3 and Round 2

For the Round 3 archive result, use the immutable Round 3 image in the build table. Keep the models, input CSV and mounted images identical between A and B. Match `OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=8` and `OPENBLAS_NUM_THREADS=1`, and choose a new output directory each time.

```bash
# Round 3 B: warmed processing; use --warmup 0 for the whole-job comparison.
deepstarnoodles-inference benchmark --mode archive \
  --models /models/manifest.json --index /inputs/archive.csv \
  --output /outputs/round3-decode2 --images 256 --warmup 4 \
  --detector-batch-size 1 --crop-batch-size 1 \
  --hydra-output-mode full --gru-execution dynamic --gru-check-interval 8 \
  --decode-workers 2 --decode-prefetch-frames 4 \
  --decode-prefetch-mib 256 --decode-max-frame-mib 64 \
  --archive-frame-window 0 --archive-crop-buffer-mib 256 \
  --allocator-limit-gib 7 --ram-limit-gib 24
```

For Round 3 A, change only `--decode-workers 2` to `--decode-workers 0` and choose a fresh output directory. Its live comparison uses zero-warmup `live-replay`, sequential decode, crop batch 1, `--hydra-batch-size 1 --hydra-output-mode minimal --gru-execution dynamic --gru-check-interval 8`; it is a measured parity-valid configuration, not a general latency recommendation.

Use the immutable Round 2 image below and the pinned model manifest. These commands assume its CLI is available in the container, with models and CSV/image inputs mounted read-only at `/models` and `/inputs`, and a writable `/outputs`. Set `OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=8` and `OPENBLAS_NUM_THREADS=1` to match the worker. Choose an unused output-directory path for each execution; the CLI creates it and rejects existing directories. The private workload is not distributed with this report, so substitute your own representative CSV when it is unavailable.

```bash
# Archive candidate, zero-warmup complete-job configuration.
deepstarnoodles-inference archive \
  --models /models/manifest.json --index /inputs/archive.csv \
  --output /outputs/archive-candidate --warmup 0 \
  --detector-batch-size 1 --crop-batch-size 1 \
  --hydra-output-mode minimal --gru-execution fixed --gru-check-interval 1 \
  --allocator-limit-gib 7 --ram-limit-gib 24

# Archive repeated-processing configuration: same flags, 256 frames + 4 warmup.
deepstarnoodles-inference benchmark --mode archive \
  --models /models/manifest.json --index /inputs/archive.csv \
  --output /outputs/archive-warmed --images 256 --warmup 4 \
  --detector-batch-size 1 --crop-batch-size 1 \
  --hydra-output-mode minimal --gru-execution fixed --gru-check-interval 1 \
  --allocator-limit-gib 7 --ram-limit-gib 24

# Live replay: chronological frames; run each clip separately to reset tracking.
deepstarnoodles-inference live-replay \
  --models /models/manifest.json --index /inputs/clip.csv \
  --output /outputs/live-replay --warmup 0 \
  --detector-batch-size 1 --crop-batch-size 1 \
  --hydra-output-mode minimal --gru-execution fixed --gru-check-interval 1 \
  --allocator-limit-gib 7 --ram-limit-gib 24
```

For the corresponding reference, replace the three candidate flags with `--hydra-output-mode full --gru-execution dynamic --gru-check-interval 8`. Archive mode treats frames independently; live replay maintains tracking state within each clip. The `live` camera command is a separate path and was not benchmarked here. The CLI’s allocator/RSS flags do not reproduce the campaign’s external whole-device/process-tree sampler by themselves.

## Exact builds and checkpoints

| Measured build | Source commit | Immutable image |
| --- | --- | --- |
| Completed pilot | [`2b37bb6888179a19fd8c02071b7cd4bc7bdfcf5f`](https://github.com/weertman/deepstarnoodles-inference/tree/2b37bb6888179a19fd8c02071b7cd4bc7bdfcf5f) | `ghcr.io/weertman/deepstarnoodles-inference@sha256:48840df16d7b3e638eca357e34c045e4c4e84ae1951388974502ca65acd5e12b` |
| Completed Round 2 | [`e5dcb7c1040c8ba8be9c1d01a96698074981b36f`](https://github.com/weertman/deepstarnoodles-inference/tree/e5dcb7c1040c8ba8be9c1d01a96698074981b36f) | `ghcr.io/weertman/deepstarnoodles-inference@sha256:fb2364ba364a0577a33607930caa0152c530fd43b0f311a91be290a9eb70bfec` |
| Completed Round 3 | [`fe68dfe3732b878a3520f1bc36e5058dae913612`](https://github.com/weertman/deepstarnoodles-inference/tree/fe68dfe3732b878a3520f1bc36e5058dae913612) | `ghcr.io/weertman/deepstarnoodles-inference@sha256:931c8f145c2553d167bdc627ef6dba054adf9b5e500c537c19565762bd787e64` |
| Completed Round 4 | [`fe68dfe3732b878a3520f1bc36e5058dae913612`](https://github.com/weertman/deepstarnoodles-inference/tree/fe68dfe3732b878a3520f1bc36e5058dae913612) | `ghcr.io/weertman/deepstarnoodles-inference@sha256:931c8f145c2553d167bdc627ef6dba054adf9b5e500c537c19565762bd787e64` |
| Completed Round 5 | Base `3e2eccad861c25db84c037977c3eab4006d9c6f8` plus overlay wheel `deepstarnoodles_inference-0.5.0.dev1` (sha256 `057a167ade192fb3f3d5d31316e78ac0bc72d30425897c533d9c0d1d62c46787`), installed during preflight | Same as Round 4 |
| Completed Round 6 | Base `3e2eccad861c25db84c037977c3eab4006d9c6f8` plus overlay wheel `deepstarnoodles_inference-0.5.0.dev2` (sha256 `3a6642a2b186dea06ba118909ed85d7b044224321b5d006760863c9d85768657`), installed during preflight | Same as Round 4 |
| Completed Round 7 | Base `3e2eccad861c25db84c037977c3eab4006d9c6f8` plus overlay wheel `deepstarnoodles_inference-0.5.0.dev3` (sha256 `9c678be4d5fb365151547377e45cda39c9f2aafddf0c3d6225eeed757ab8f83e`), installed during preflight | Same as Round 4 |
| Completed Round 8 | Base `3e2eccad861c25db84c037977c3eab4006d9c6f8` plus overlay wheel `deepstarnoodles_inference-0.5.0.dev4` (sha256 `9c7efc953e1279928289db158f7886608f2c2724ae778831cdd873cfd097f70e`), installed during preflight | Same as Round 4 |

Runtime: PyTorch 2.8.0+cu128, CUDA 12.8, cuDNN 91002, Ultralytics 8.4.51, NumPy 2.2.6, OpenCV 4.11.0, eight PyTorch CPU threads. Separate hosts/builds are not timing controls for one another. Independent comparisons established identical pilot/Round 2 and Round 2/Round 3 reference predictions on the same ordered 256 inputs and checkpoint hashes; that validates baseline output continuity only. Round 3 used a separate host. Its faster initial reference is not itself an optimization gain. Round 4 uses the same source/image as Round 3 on a separate lease and reproduces its eager reference bytes; only within-round pairs support timing claims.

| Checkpoint | SHA-256 |
| --- | --- |
| StarSeg, YOLO26-L segmentation | `5929d8a9c1ae3cf9a8b5c3ab9b1372221fe9f8d6cadd5a94f020aac8c9c3fb53` |
| Hydra, DINOv3 ConvNeXt-Base, EMA export | `aa056f933c1a0994fb299203aa20423c5208637d2eb4ede7e942abcc3aa91d7c` |
| Arm tracer, GRU hidden size 128 | `e6c1506067fc5864ab1d0d648dec2f7932632017d952753cb68f80fed2c4812f` |

The GRU traces arms within an image; live temporal tracking is separate. The [model release](https://github.com/weertman/deepstarnoodles-inference/releases/tag/models-seed0-20261028) supplies the portable manifest and checkpoints.

The completed measurements are bounded to the configurations and workload above. The numeric evidence preserves exact source/image pins, raw repeats, signed effects, resource coverage and verified final-manifest hashes.
