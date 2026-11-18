# DeepStarNoodles inference

Portable StarSeg → Hydra → GRU128 inference for video files, image archives, cameras and live replay. The wheel contains the runtime, crop and tracking configuration, and DINOv3 ConvNeXt source, so it needs no research checkout. Checkpoints and images stay external.

The trained models are YOLO26-L segmentation, **DINOv3 ConvNeXt-Base** and a GRU with 128 hidden units, which follows each arm within an image. Live modes add BoT-SORT animal IDs and per-animal arm association. Archive images are independent.

| Document | What it covers |
|---|---|
| This README | Installing, getting the models and the commands to run |
| [docs/options.md](docs/options.md) | Every speed, scheduling and live option, including the ones measurement rejected |
| [docs/representations.md](docs/representations.md) | Turning output into the paper's polyline representations |
| [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) | Every measurement, with rates per GPU type, output agreement and memory |

**Version note.** This branch is 0.5.0.dev12. It writes output schema 4. Schema 3 (0.5.0.dev11) added each star's body origin, support area and centroid and each frame's time, along with the `representations` command. Schema 4 adds each star's mouth, madreporite and butthole positions and probabilities, and 0.5.0.dev12 adds the `track-videos` command. The published container image is still 0.4.0, so it lacks `serve-videos`, `track-videos`, `representations` and most options in the commands below. Install from source or build the image from this repository, as shown next.

## Install

Use Python 3.11 or newer. Install the target PyTorch wheels first, then the package:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -c constraints-cuda128.txt -r requirements-runtime.txt
python -m pip install .
deepstarnoodles-inference smoke --device cuda:0
```

The smoke command builds the full architecture with random weights and runs a small synthetic forward pass. It checks the installation, not accuracy or speed. `--device cpu` works for development.

Or build the container, which pins PyTorch 2.8.0, torchvision 0.23.0, CUDA 12.8 and Ultralytics 8.4.51:

```bash
docker build -t deepstarnoodles-inference .
```

Inside the container the same commands run as `docker run --rm --gpus all --shm-size=1g -v "$PWD:/work" deepstarnoodles-inference <command>`, without the `deepstarnoodles-inference` prefix. For RunPod's SSH worker mode, set the container argument to `runpod-worker`, expose TCP port 22, and supply your SSH **public** key in `PUBLIC_KEY`. Host keys are created at startup, and password login is disabled.

## Get the models

The checkpoints are published in [models-seed0-20261028](https://github.com/weertman/deepstarnoodles-inference/releases/tag/models-seed0-20261028), about 793 MB in total. This downloads them and verifies their SHA-256 hashes:

```bash
curl -fL -o manifest.json \
  https://github.com/weertman/deepstarnoodles-inference/releases/download/models-seed0-20261028/manifest.json
deepstarnoodles-inference fetch-models --models manifest.json --destination models
deepstarnoodles-inference verify-models --models models/
```

`models/` must not already exist. Nothing is downloaded implicitly during inference. Checkpoint loading uses PyTorch serialization, so only load trusted weights whose hashes match the manifest.

## Annotate videos

`serve-videos` is a resident service for annotating many videos. This is the configuration measured across GPU types:

```bash
nvidia-cuda-mps-control -d
deepstarnoodles-inference serve-videos --queue queue --output out --models models/manifest.json --processes 8 \
  --device cuda:0 --detector-batch-size 1 --crop-batch-size 1 --hydra-output-mode minimal \
  --gru-execution graph --gru-check-interval 1 --hydra-execution graph \
  --detect-ahead 2 --decode-prefetch-frames 4 --gpu-crops --cuda-blocking-sync \
  --hydra-fp16-weights --hydra-compile --gru-compile --gru-graph-first-step
```

Workers annotate each frame on its own, so arms are numbered per frame and stars have no IDs. Each star keeps what later steps need from Hydra's maps, which are too large to save; see [Output format](#output-format). [Track stars and arms afterwards](#track-stars-and-arms-afterwards) gives them IDs.

Move each finished video into `queue/ready/` with a rename. Each worker claims whole videos, decodes every frame and annotates each frame independently. Records are appended to `out/<video>.jsonl.part`, which becomes `out/<video>.jsonl` when the video is done, and the video is then deleted, so the queue works as a rolling cache. A `STOP` file in the queue directory ends every worker after its current frame, and undecodable videos move to `queue/failed/`. `--claim-prefix P` limits workers to names starting with `P`.

On nine GPU types this sustained 19–67 images/s per GPU on a mix of lab videos with about 2.8 stars per frame. The RTX 5090 cost the least per frame, at 58 images/s and $1.19/h, and the RTX 4090 was next, at 41 images/s. [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md#round-16-speed-test-across-gpu-types) has the table.

- **Always run several processes under CUDA MPS** (see [Known issues](#known-issues)). Keep `CUDA_MPS_PIPE_DIRECTORY` short, because the control socket path must fit the 108-byte Unix socket limit.
- **Six to ten processes** saturated every GPU tested. Each holds about 2 GiB of GPU memory, flat from video to video, at every resolution tested up to 5496×3672.
- **`--allocator-limit-gib`** defaults to 7 per process, so lower it if the processes together could fill the card. Setting it to `(0.9 × GPU MiB ÷ processes − 400) ÷ 1024` leaves room for each process's CUDA context, and 1.6 was enough in testing.

Compiled outputs are not byte-identical to the uncompiled pipeline, but the difference is very small. On the 478-image validation split F1 is 0.8033 against 0.8034, with one extra false positive in 4,801 arms. On the lab mix 99.6% of arms stay within 2 px, and star counts match in all 371 frames. For byte-identical outputs, leave out `--hydra-compile --gru-compile`, which cost about a quarter of the speed in a local test.

## Track stars and arms afterwards

`track-videos` gives the stars and arms in `serve-videos` records persistent IDs, as live replay does during inference. BoT-SORT follows each frame's star boxes. Then each star's arm tracker follows its arms, measured from the star's saved `body_origin_model`. It needs no GPU, model weights or video, and it runs one video per process:

```bash
deepstarnoodles-inference track-videos out/ --output tracked/ --jobs 8
```

- **Inputs:** `.jsonl` or `.jsonl.zst` files, or directories, which are searched recursively and mirrored under `--output`. Compressed files need `zstd` on `PATH`.
- **Resuming:** videos already tracked are skipped, so a rerun picks up where it stopped.
- **What it writes:** each star gets `star_track_id`, and each arm gets `track_id`, scoped to its star. `arm_index` stays.
- **Unconfirmed stars:** live replay only crops stars BoT-SORT has confirmed, which leaves out each star in the first frame it appears in after a video's first frame. Those detections move to the frame's `untracked_stars`, so `stars` matches live replay, nothing is lost, and the output feeds `representations` without `--untracked`.
- **No camera-motion correction:** BoT-SORT normally corrects for camera motion using the frames, which aren't available here. With a fixed camera that changes little.

On a 4-minute, 1 fps lab clip, both ways kept one star ID throughout. An arm changed ID between frames 5.6% of the time afterwards against 5.4% live. Records from before 0.5.0.dev11 have no body origin, so the command refuses them.

## Annotate images

`archive` annotates a CSV of images with an `image_path` column; relative paths resolve against the CSV. Under MPS, this configuration reached about 52 images/s on an RTX 5090, and its predictions are byte-identical to the earlier Round 4 selection, which passed the strict parity gates against the reference:

```bash
nvidia-cuda-mps-control -d
deepstarnoodles-inference archive --models models/ --index inputs/archive.csv --output outputs/archive \
  --detector-batch-size 1 --crop-batch-size 1 --hydra-output-mode minimal \
  --gru-execution graph --gru-check-interval 1 --hydra-execution graph \
  --detect-ahead 2 --archive-processes 8 --gpu-crops --cuda-blocking-sync \
  --decode-workers 2 --decode-prefetch-frames 4 --decode-prefetch-mib 256 --decode-max-frame-mib 64
```

Output directories must be new. Results are `predictions.jsonl` and a `report.json` with checkpoint hashes, hardware, runtime details, crop counts, timing and memory peaks. Images with no detections still get a record.

## Annotate a camera or a live stream

```bash
deepstarnoodles-inference live --models models/ --camera 0 --output outputs/camera \
  --max-seconds 60 --crop-batch-size 1 --hydra-output-mode minimal --gru-execution fixed --gru-check-interval 1
deepstarnoodles-inference live --models models/ --video inputs/clip.mp4 --output outputs/clip \
  --crop-batch-size 1 --hydra-output-mode minimal --gru-execution fixed --gru-check-interval 1
```

Camera mode keeps one pending frame and reports dropped frames. A video is processed frame by frame in order, and it doesn't simulate camera timing. `live-replay` does the same for a chronological CSV of images, one stream per command. In Docker, add `--device=/dev/video0:/dev/video0` to use a camera.

The same resident pipeline accepts frames from your own code:

```python
from deepstarnoodles_inference.live import LiveConfig, LivePipeline

pipeline = LivePipeline.from_manifest("models/", LiveConfig(device="cuda:0", crop_batch_size=1))
result = pipeline.predict(frame_bgr)  # H x W x 3 numpy.uint8, BGR order
pipeline.reset()  # New independent stream; models stay loaded.
```

Call `predict()` synchronously and don't change the frame until it returns. After a failed inference, call `reset()` before continuing.

Latency on short warmed replays improved with this profile, but those replays don't establish sustained camera frame rates. Test it on your own stream. [docs/options.md](docs/options.md#camera-and-live-details) has more on timing and shutdown.

## Output format

Each line of a `.jsonl` file is one frame:

- A frame has `frame`, `image_path`, `source_shape` and `stars`, plus `video` from `serve-videos`. `frame` is the video frame for `serve-videos` and `live --video` and the CSV row for image indexes.
- A star has `detection_index`, `star_track_id`, `crop` (`left`, `top`, `side`, `confidence`, `bbox_xyxy`), `mask_polygon_source` and `arms`.
- An arm has `polyline_source`, `polyline_crop`, `length_px`, `steps`, `stop_reason`, `state_confidence`, `tip_score`, `base_score` and an `identity` vector. Archive arms use `arm_index`, and live arms use `track_id` scoped to their `star_track_id`.
- After its arms, a star has `body_origin_model` (`[x, y]`), the runtime body centre that arm association uses, and `body_origin_code`, which says where it came from: 0 for the mouth peak, 1 for the support > 0.3 centroid and 2 for the crop centre. It also has `support_area_model_px`, the number of support > 0.5 pixels, and `support_centroid_model`, their mean `[x, y]`, which is `null` when the area is 0.
- Last, a star has `landmarks_model`, which maps `mouth`, `madreporite` and `butthole` to `[x, y, p]`. Each is the maximum of Hydra's heatmap for that landmark, with its probability. Nothing is thresholded, so a landmark the model can't see has a low `p`, and its position means little. A non-finite map gives `null`.
- Each frame ends with `time_s`, `fps` and `schema_version` (4). `time_s` is `frame / fps`. The rate comes from the video for `serve-videos` and `live --video`, and from `--fps` for image indexes. Without a rate, as for a camera, both are `null`. For image indexes `time_s` counts from the first CSV row, not from the start of the video: stills cut from mid-video need their start time added.

Coordinates are in source pixels unless the name says crop or model. Model coordinates are pixels of the 512-pixel crop.

Schemas 3 and 4 only append keys: every earlier key keeps its value and position, and each file still has exactly one line per frame. Readers should detect the new fields by their presence, since records from 0.5.0.dev10 and earlier lack them all, and those from 0.5.0.dev11 lack the landmarks. `compare` checks the new fields only when both files have them. It compares a landmark's position only where both runs give it a `p` of at least 0.5.

## Representations

`representations` turns output files into the polyline representations of the DeepStarNoodles paper figures, using `noodle_polylines` vendored from DeepStarNoodles. It needs no GPU, PyTorch or OpenCV:

```bash
deepstarnoodles-inference representations outputs/live/predictions.jsonl --output reps/ --csv
deepstarnoodles-inference representations stream_out/ --output reps/ --untracked single --preset fig6_flipping
```

It writes one container per star, each preset's outputs, optional CSV tables and a manifest. Archive and `serve-videos` records carry no star IDs. Run [`track-videos`](#track-stars-and-arms-afterwards) on them first, or pass `--untracked single` for one animal per video, or `--untracked detections`. Records from 0.5.0.dev10 and earlier give only the outputs that need no body origin. [docs/representations.md](docs/representations.md) covers presets, R0 context and the output layout.

## Known issues

- **Worker faults without MPS.** When several processes time-slice one GPU, graph execution with `--detect-ahead` occasionally aborts a process with a CUDA illegal memory access (Xid 31, a read fault through the constant cache). `serve-videos` without MPS faulted 3 times in about 1.1 process-hours. Under CUDA MPS none has occurred in about 45 streamed process-hours on rented pods, or in the archive rounds. The root cause has not been isolated. A fault ends the process, and no run has produced different predictions. Under MPS a fatal fault in one client ends them all, so wrap long jobs in a retry.
- **Outputs vary slightly by GPU chip family.** Runs on one GPU type repeat byte for byte. Between Ampere, Ada and Blackwell, about 1.5% of arms lack a match within 2 px and 0.2–0.3% of frames change star count. Star counts, arm counts and arm lengths agree within 0.1%, so it looks like noise rather than bias.
- **Larger batches fail output parity.** Keep detector and crop batches at one.
- **No RTX 5060 or 8 GB measurement exists.** An allocator limit on a larger GPU doesn't reproduce a smaller GPU's speed or guarantee that it fits.

## Compare two runs

`compare` streams two prediction files through fixed parity gates. It needs no GPU or model weights:

```bash
deepstarnoodles-inference compare --reference outputs/a/predictions.jsonl \
  --candidate outputs/b/predictions.jsonl --output outputs/comparison.json
```

It exits 0 only for a pass, 1 for a failed comparison or a file error, and 2 for invalid arguments. `--output` must name a new file. Source paths must match, so mount inputs at the same path in both runs. A pass doesn't establish accuracy, memory fit or a speedup.

## Development

```bash
python -m pip install ".[test]"
python -m pytest
```

Tests that need CUDA skip themselves on CPU. The GitHub Actions workflow tests the installed wheel on CPU and builds the CUDA image on every pull request. It publishes `ghcr.io/weertman/deepstarnoodles-inference` on pushes to `main`, version tags and manual dispatch, and it records the immutable image digest. Use that digest for experiments. A commit with `[skip ci]` in its message updates `main` without publishing an image.

Licensing: project runtime code is MIT; bundled DINOv3 and BoT-SORT material retains its own terms. See `THIRD_PARTY_NOTICES.md` and `licenses/`.
