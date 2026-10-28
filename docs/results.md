## Policies (eval half, all categories pooled)

| policy | recall | miss rate | false-alarm rate | human review rate | thresholds |
|---|---|---|---|---|---|
| stage1_tau | 94.0% | 6.0% | 34.6% | 0.0% |  |
| 3way_human | 97.7% | 2.3% | 1.7% | 40.9% |  |
| 3way_vlm[img] | 68.2% | 31.8% | 8.7% | 0.0% |  |
| 3way_vlmP[img] | 94.3% | 5.7% | 38.8% | 0.0% | t=0.01 |
| hybrid[img] | 97.7% | 2.3% | 5.0% | 37.7% | t_ok=0.0, t_def=0.8 |
| 3way_vlm[zoom] | 68.7% | 31.3% | 3.7% | 0.0% |  |
| 3way_vlmP[zoom] | 97.3% | 2.7% | 46.1% | 0.0% | t=0.05 |
| hybrid[zoom] | 97.3% | 2.7% | 3.7% | 34.8% | t_ok=0.05, t_def=0.5 |
| 3way_vlm[zoom_ref] | 65.0% | 35.0% | 2.5% | 0.0% |  |
| 3way_vlmP[zoom_ref] | 97.5% | 2.5% | 46.7% | 0.0% | t=0.05 |
| hybrid[zoom_ref] | 97.5% | 2.5% | 3.9% | 35.5% | t_ok=0.05, t_def=0.4 |
| 3way_vlm[zoom_ref_heat] | 73.3% | 26.7% | 6.6% | 0.0% |  |
| 3way_vlmP[zoom_ref_heat] | 97.7% | 2.3% | 46.1% | 0.0% | t=0.1 |
| hybrid[zoom_ref_heat] | 97.7% | 2.3% | 2.9% | 35.2% | t_ok=0.1, t_def=0.6 |

## VLM reviewer inside the review zone (eval half)

| variant | n | verdict accuracy | defect recall | normal clear rate | P(defect) AUROC | stage-1 score AUROC | distinct self-confidence values | parse fails | latency (s, under load) | prompt tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| img | 442 | 52.3% | 20.3% | 84.5% | 0.602 | 0.639 | 3 | 0.0% | 1.66 | 415 |
| zoom | 442 | 58.4% | 21.6% | 95.5% | 0.694 | 0.639 | 3 | 0.0% | 3.16 | 997 |
| zoom_ref | 442 | 54.8% | 11.7% | 98.2% | 0.653 | 0.639 | 2 | 0.0% | 3.68 | 1524 |
| zoom_ref_heat | 442 | 61.5% | 34.2% | 89.1% | 0.665 | 0.639 | 3 | 0.0% | 4.85 | 1797 |

## Per category (stage 1, eval half)

| category | image AUROC | ms/image | review rate | stage-1 recall | stage-1 false alarms |
|---|---|---|---|---|---|
| candle | 0.970 | 34 | 30.0% | 94.0% | 20.0% |
| capsules | 0.700 | 36 | 82.5% | 98.0% | 96.7% |
| cashew | 0.974 | 30 | 12.0% | 96.0% | 28.0% |
| chewinggum | 0.996 | 29 | 8.0% | 98.0% | 0.0% |
| fryum | 0.938 | 25 | 42.7% | 74.0% | 8.0% |
| macaroni1 | 0.882 | 35 | 59.0% | 90.0% | 40.0% |
| macaroni2 | 0.595 | 64 | 90.0% | 92.0% | 90.0% |
| pcb1 | 0.954 | 27 | 33.0% | 96.0% | 26.0% |
| pcb2 | 0.939 | 35 | 42.0% | 94.0% | 36.0% |
| pcb3 | 0.952 | 35 | 40.6% | 98.0% | 39.2% |
| pcb4 | 0.992 | 27 | 22.8% | 100.0% | 3.9% |
| pipe_fryum | 0.989 | 27 | 14.7% | 98.0% | 4.0% |
