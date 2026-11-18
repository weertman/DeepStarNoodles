# GPU speed test

Measured 30–31 October 2026 on RunPod's secure cloud, one pod of each type in stock, with the pipeline in this repository (`inference/`, 0.5.0.dev10). The RTX 5090 was out of stock then and was measured on 8 November 2026 with the same harness. All 18 sessions passed the steadiness checks below, and no worker failed.

| GPU | Memory | $/h | Best processes | Images/s | Images/s per $/h | $ per million frames | $ per camera-year | Days per camera-year, 1 pod | 3 pods | 5 pods | 8 pods |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RTX 5090 | 32 GB | 1.19 | 12 | 58.1 | **48.8** | 5.69 | 179 | 6.3 | 2.1 | 1.3 | 0.8 |
| RTX 4090 | 24 GB | 0.89 | 8 | 41.1 | 46.1 | 6.02 | 190 | 8.9 | 3.0 | 1.8 | 1.1 |
| RTX 3090 | 24 GB | 0.50 | 8 | 20.9 | 41.9 | 6.63 | 209 | 17.4 | 5.8 | 3.5 | 2.2 |
| RTX PRO 4500 Blackwell | 32 GB | 0.72 | 6 | 27.6 | 38.3 | 7.26 | 229 | 13.2 | 4.4 | 2.6 | 1.7 |
| RTX A6000 | 48 GB | 0.59 | 8 | 22.2 | 37.6 | 7.39 | 233 | 16.5 | 5.5 | 3.3 | 2.1 |
| L40S | 48 GB | 1.09 | 8 | 40.6 | 37.3 | 7.46 | 235 | 9.0 | 3.0 | 1.8 | 1.1 |
| RTX 6000 Ada | 48 GB | 0.99 | 8 | 36.4 | 36.7 | 7.56 | 238 | 10.0 | 3.3 | 2.0 | 1.3 |
| A40 | 48 GB | 0.59 | 8 | 19.3 | 32.6 | 8.51 | 268 | 19.0 | 6.3 | 3.8 | 2.4 |
| RTX PRO 6000 Blackwell | 96 GB | 2.49 | 10 | 67.4 | 27.1 | 10.26 | 324 | 5.4 | 1.8 | 1.1 | 0.7 |

A camera-year is 31.5 million frames: 1 fps for 365 days. Rates are on the lab mix of about 2.8 stars per frame, and they hold only for that mix: frames with fewer stars run faster. Prices were RunPod's secure on-demand rates on the day, and they change. A pilot on your footage gives the real rate for your archive (`python3 -m backcast report`).

**The RTX 5090 was out of stock for the whole first test.** It was measured on 8 November with the same harness, segments and checks: 58.1 images/s at $1.19/h, close to the 60 expected from its chip. RunPod showed its stock as Low that day.

## What to rent

- **Cheapest per frame: RTX 5090, when it is in stock.** It costs 6% less per frame than the 4090 and runs 40% faster per pod. It was out of stock for the whole first test and showed Low stock on 8 November, so don't count on it alone.
- **Next: RTX 4090.** It also runs twice as fast per pod as the 3090, so the same number of pods finishes in half the time.
- **Then the 3090, then whichever of the others is in stock.** Per frame, the 3090 costs 10% more than the 4090, the PRO 4500, A6000, L40S and Ada 6000 cost 21–26% more, and the A40 costs 41% more. List them all in `fleet.toml`: the 3090, 4090, A6000, PRO 4500 and Ada 6000 all showed Low stock after the test, and five of one type may not be available.
- **RTX PRO 6000 only for a hard deadline.** It's the fastest pod and costs 70% more per frame than the 4090, so it's off by default in `fleet.example.toml`.
- **The upload isn't the limit for a few pods.** At 91 KB per frame, a 4090 pod needs about 30 Mbit/s up and 6 Mbit/s down. A 400 Mbit/s limit feeds about 13 of them. A 5090 pod needs about 42 Mbit/s up, so the same limit feeds about 9.

## Process counts

Adding processes past the table's count did not help. Every session ran the GPU at 97–100%, and the second process count on each card was within 2% of the first, except on the A6000, where 8 beat 6 by 6%. The CPU was not the limit: the workers used about 2–7 cores.

| GPU | Processes: images/s |
|---|---|
| RTX 5090 | 10: 57.8, 12: 58.1 |
| RTX 4090 | 8: 41.1, 10: 40.9 |
| RTX 3090 | 6: 20.6, 8: 20.9 |
| RTX PRO 4500 Blackwell | 6: 27.6, 8: 27.2 |
| RTX A6000 | 6: 21.0, 8: 22.2 |
| L40S | 8: 40.6, 10: 40.2 |
| RTX 6000 Ada | 8: 36.4, 10: 35.7 |
| A40 | 6: 19.2, 8: 19.3 |
| RTX PRO 6000 Blackwell | 10: 67.4, 14: 66.2 |

## GPU memory

Each process holds about 2 GiB, and it stays flat from video to video at every resolution up to 5496×3672. The highest peaks in the test were 11.0 GiB with 6 processes, 16.2 GiB with 8, 22.3 GiB with 10, 25.2 GiB with 12 and 30.2 GiB with 14. So 8 processes leave plenty of room on a 24 GB card, and `serve.sh` caps each process at its share of the card regardless.

## Do outputs depend on the GPU?

On one GPU type, outputs don't change with the process count: arm positions matched to within 0.01 px on about 5,800 frames and 138,000 arms for every type. Between GPU types they differ slightly, mostly by chip family. On the 5,730 frames every type annotated, taking the lower of the two directions for arms:

| Compared | Frames with the same star count | Arms with a match within 2 px |
|---|---:|---:|
| A6000 and A40 (both Ampere) | 100% | 100% |
| 4090 and RTX 6000 Ada (both Ada) | 100% | 100% |
| PRO 4500 and PRO 6000 (both Blackwell) | 100% | 100.0% |
| 4090 and L40S (both Ada) | 100% | 99.7% |
| A6000 and 3090 (both Ampere) | 99.8% | 99.3% |
| A6000 and 4090 (Ampere and Ada) | 99.7% | 98.5% |
| A6000 and PRO 6000 (Ampere and Blackwell) | 99.8% | 98.4% |
| 4090 and PRO 6000 (Ada and Blackwell) | 99.8% | 98.4% |
| A6000 and 5090 (Ampere and Blackwell) | 99.8% | 98.4% |

The 5090, measured later, is compared on the 5,850 frames it shared with the A6000.

Pairs across families miss the 99% line that this project's relaxed agreement gate uses. The differences look like noise rather than bias: across the eight types of the first test, stars per frame stayed within 2.398–2.400, arms per star within 9.89–9.90, and mean arm length within 100.15–100.18 px. A different star count came from 11–17 stars out of about 13,700, and every star found by both GPUs overlapped at IoU 0.98 or better.

So a mixed fleet shouldn't shift counts or lengths, but individual arms can differ between videos annotated on different families. `_manifest.jsonl` records the GPU behind every video, so you can check this in your analysis. Mixing is the default. To annotate everything on one family, set `gpu_family` in `fleet.toml` to `"ada"` (4090, L40S, RTX 6000 Ada), `"ampere"` (3090, A6000, A40) or `"blackwell"` (5090, RTX PRO cards) before the full run. It doesn't redo finished videos. Validation accuracy hasn't been measured on the Ada family. On the 478-image split, the uncompiled pipeline scored F1 0.8034 on both an A6000 and a 5090, and the compiled one scored 0.8033 on an A6000.

## Method

Each GPU type ran two 14-minute sessions at different process counts: a 4-minute ramp, then a 10-minute measured window. Annotations were counted when they reached the controlling machine.

The videos were two-minute segments cut from synthetic 1 fps lab videos, weighted to the lab mix of 2.76 stars per frame. Each was uploaded once and replayed from a cache on the pod, so these rates show what the GPU can do, not the uploader's network.

A window counts as a sustained rate only if all of these hold:

- its second half is within 5% of its first half;
- its stars per frame are within 5% of the lab mix;
- the pod's video queue never ran dry.

Times in the table assume the rate holds for the whole job and that pods are working all the time. Each new pod adds about 5–10 minutes of setup, and pods are replaced every `lease_hours`.

## Why segments

With 30-minute videos, images/s fell steadily through every session while the GPU's star-crop rate stayed flat. Light videos finish first, then workers pile up on heavy ones. A real multi-day run averages this out. A short benchmark does not, which is why the test used two-minute segments.
