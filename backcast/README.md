# DeepStarNoodles backcast

Annotate a long archive of 1 fps sunflower-star video with the StarSeg → Hydra → GRU pipeline, on a fleet of GPUs rented from RunPod. It runs from one lab machine, which sends each video to a pod and brings the annotations home. It spends real money, so it has a hard budget and rents pods only when you run `run`.

## Get it going on the lab machine

You need Python 3.11 or newer, a RunPod API key, the footage readable from this machine, and outbound SSH to RunPod's high TCP ports.

```bash
git clone https://github.com/weertman/DeepStarNoodles.git
cd DeepStarNoodles/backcast

# System tools. Without sudo: conda install -c conda-forge rsync zstd ffmpeg openssh
sudo apt install -y rsync zstd ffmpeg openssh-client

# RunPod key: runpod.io > Settings > API Keys. Every shell that runs the fleet needs it, tmux included.
export RUNPOD_API_KEY=...

# Point it at your footage: edit roots, output dir, pods and budget.
cp fleet.example.toml fleet.toml
```

The controller itself needs only the Python standard library. If this machine has an older Python, `conda create -n backcast python=3.11` is enough.

Then, in this order:

```bash
python3 -m backcast doctor     # checks this machine, the key, the footage and current GPU stock; rents nothing
python3 -m backcast plan       # counts frames and terabytes, estimates cost and time; rents nothing
```

```bash
# Pilot: about 20 videos evenly spaced through the archive, on 2 pods, capped at $5.
python3 -m backcast run --limit-videos 20 --pods 2 --max-usd 5
python3 -m backcast report     # your footage's real images/s, cost per million frames and projected total
```

Look at some output before going further. This pilot is also the runner's first run on real pods: the pipeline and pod setup ran on eight GPU types in the speed test, but the runner itself has so far been tested only with fake pods and a local GPU standing in for a pod. Watch `python3 -m backcast status` while it runs.

Then start the full job inside `tmux` so it survives a dropped connection:

```bash
tmux new -s backcast
python3 -m backcast run
```

The pilot's finished videos count; the full run does not repeat them.

## Watching and stopping it

```bash
python3 -m backcast status     # progress, rate, spend, each pod, failed videos
python3 -m backcast stop       # finish what is on the pods, then end
python3 -m backcast terminate  # end every pod right now; unfinished videos go back in the queue
```

- **Resuming:** rerun `python3 -m backcast run`. It skips finished videos, picks up new ones, and re-adopts pods that are still alive. If the controller dies, pods keep billing until `lease_hours` at most, so rerun it or run `terminate`.
- **Ctrl-C:** terminates every pod. Unfinished videos stay in the queue.
- **Budget:** at 97% of `max_total_usd` it stops sending new work; at 100% it terminates everything. Spend is estimated from each pod's hourly rate. To continue, rerun with `--max-usd` higher.
- **Failed videos:** a video that cannot be decoded is set aside and listed by `status`. `retry-failed` puts them back in the queue.

## What you get

For each video, `<output dir>/<root name>/<relative path>.jsonl.zst` holds one JSON record per frame. `_manifest.jsonl` records which GPU annotated each video.

```bash
zstd -dc annotations/cam1/day1.mp4.jsonl.zst | head -1 | python3 -m json.tool
```

Each record is one frame. It has `video`, `frame`, `source_shape` and `stars`, and it ends with `time_s`, `fps` and `schema_version` (4).

- **Star:** its `crop` (`bbox_xyxy`, `confidence`), `mask_polygon_source` and `arms`. After the arms come what later steps need from Hydra's maps:
  - `body_origin_model`: the body center that arm tracking measures arms from;
  - `support_area_model_px` and `support_centroid_model`;
  - `landmarks_model`: `mouth`, `madreporite` and `butthole` as `[x, y, p]`.
- **Arm:** `polyline_source`, `polyline_crop`, `length_px`, `tip_score`, `base_score`, `stop_reason`, an 8-number `identity`, and `arm_index`, which numbers it within its frame only.

Stars have no IDs yet, because each frame is annotated on its own; see [Tracking and representations](#tracking-and-representations). Coordinates are in source pixels unless the name says crop or model. Model coordinates are pixels of the 512-pixel crop Hydra sees, so a model point maps to source pixels as `x * side / 512 + left`. Expect about 17 KB per frame compressed, or 78 KB raw, on the lab mix: about 0.6 TB per camera-year. `inference/README.md` lists every field.

Records written before 8 November 2026 (inference 0.5.0.dev10) have none of the Hydra fields or times, so they can't be tracked. Annotate those videos again.

## Tracking and representations

The fleet annotates each frame on its own. Star and arm IDs, and the paper's polyline representations, come afterwards on the lab machine. Neither step needs a GPU, model weights or the videos.

Set up once, in its own environment. CPU PyTorch is enough:

```bash
python3 -m venv ~/dsn-analysis
. ~/dsn-analysis/bin/activate
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
pip install -c inference/constraints-cuda128.txt -r inference/requirements-runtime.txt
pip install --no-deps ./inference
pip install "zstandard>=0.22"
```

Then:

```bash
deepstarnoodles-inference track-videos /path/to/annotations --output /path/to/tracked --jobs 8
deepstarnoodles-inference representations /path/to/tracked --output /path/to/representations --csv
```

`track-videos` gives each star a `star_track_id` and each arm a `track_id`, as live tracking would have during inference. It writes a new copy and skips videos it has already done, so you can rerun it as more annotations arrive. Detections it couldn't confirm move to each frame's `untracked_stars`. `representations` writes the paper's figure outputs per star; `inference/docs/representations.md` explains its presets.

`docs/figures/polyline_transforms.py` draws, on one frame of any tracked video, how `representations` turns each traced arm into its output columns; `docs/figures/polyline_transforms_video.py` animates the same steps.

`o` is the body origin: the mouth, or mid-body when the mouth is hidden. `r` is the radius of a disc with the star's area; columns compared across frames use `R0` instead, the median `r` over a baseline period. Curvature is how far the arm's direction turns, divided by its length. For shape change, the arm from the previous frame is shifted and rotated onto it; `arm_deformation` is the frame's median RMS gap per second.

## Choosing GPUs and the number of pods

`docs/speed_test.md` has measured rates and costs per GPU type, on the lab mix of about 2.8 stars per frame. The cheapest measured was the RTX 5090:

| GPU | Images/s | $/h | $ per camera-year | Days per camera-year, 5 pods |
|---|---:|---:|---:|---:|
| RTX 5090 | 58 | 1.19 | 179 | 1.3 |
| RTX 4090 | 41 | 0.89 | 190 | 1.8 |
| RTX 3090 | 21 | 0.50 | 209 | 3.5 |
| RTX PRO 4500 Blackwell | 28 | 0.72 | 229 | 2.6 |
| RTX A6000 | 22 | 0.59 | 233 | 3.3 |
| L40S | 41 | 1.09 | 235 | 1.8 |
| RTX 6000 Ada | 36 | 0.99 | 238 | 2.0 |
| A40 | 19 | 0.59 | 268 | 3.8 |
| RTX PRO 6000 Blackwell | 67 | 2.49 | 324 | 1.1 |

The RTX 5090 was out of stock during the first test on 31 October and was measured on 8 November. Its stock showed Low then, so `fleet.example.toml` lists it first and the other types after it.

- **More pods cost about the same and finish sooner.** Each extra pod adds only about 5–10 minutes of setup.
- **List several GPU types in `fleet.toml`.** RunPod stock changes by the minute and it reports only Low, Medium or High, so the fleet rents the first type in your list that is in stock at or below its price cap, and replaces any pod that dies.
- **Chip families can mix.** Outputs differ slightly between them, with no visible bias, so the default `gpu_family = "any"` uses whatever is in stock. Set it to `"ada"`, `"ampere"` or `"blackwell"` to rent only that family. The cheapest type is the 5090 (Blackwell) and the next is the 4090 (Ada). Set it before the full run, because it doesn't redo finished videos.
- **Check the upload.** `plan` prints what your upload limit can carry. On video at 91 KB per frame, 400 Mbit/s carries about 550 images/s, or about 13 RTX 4090 pods or 9 RTX 5090 pods.

## How it works

The controller keeps a work list in SQLite (`run_dir/state.sqlite`). For each pod it installs the pinned runtime, fetches the public model weights directly from GitHub and starts `serve-videos` under CUDA MPS. It keeps a few videos waiting on the pod's local disk, polls every 20 seconds, compresses each finished annotation file on the pod, downloads it, checks the record count, and only then marks the video done and deletes the pod's copy. A pod that dies or expires gives its unfinished videos back to the queue. RunPod also ends every pod after `lease_hours`, as a backstop if the controller dies.

`inference/` is a vendored copy of the `deepstarnoodles-inference` package at 0.5.0.dev12, which adds `serve-videos` and the compile options that the published image lacks. `inference/BENCHMARK_RESULTS.md` is the full measurement history.

The pods fetch the model set `models-seed0-20261028` (`MODELS_URL` in `backcast/config.py`): the published Hydra and tracer with an earlier one-class detector of the same architecture and size, the detector of the paper's throughput tests. The paper's models are the release [`models-20261117`](https://github.com/weertman/deepstarnoodles-inference/releases/tag/models-20261117). Its manifests set the tracer's invalid-seed threshold to 0.6, which inference 0.5.0.dev14 or later applies; 0.5.0.dev12 ignores the field and traces every seed.

## Accuracy

The pipeline runs Hydra and the GRU in compiled form, about 40% faster overall. Outputs are not byte-identical to the eager pipeline. On the 478-image validation split, F1 is 0.8033 against 0.8034, with one extra false positive in 4,801 arms. On the lab mix, 99.6% of arms stay within 2 px and star counts match in all 371 frames. Compiled outputs repeat byte for byte on one GPU type. Between chip families, about 1.5% of arms lack a match within 2 px and 0.2–0.3% of frames change star count, with no visible bias in star counts, arm counts or arm lengths. The fleet mixes families by default; set `gpu_family` in `fleet.toml` to rent only one.

## Tests

```bash
python3 -m unittest discover tests    # fake pods: no cloud, no GPU, no money
```

## More detail

- [`docs/speed_test.md`](docs/speed_test.md): rates and costs per GPU, how to choose, process counts, memory, and how outputs differ between GPUs.
- [`fleet.example.toml`](fleet.example.toml): every setting, with comments.
- [`inference/README.md`](inference/README.md) and [`inference/BENCHMARK_RESULTS.md`](inference/BENCHMARK_RESULTS.md): the pipeline and its full measurement history.
