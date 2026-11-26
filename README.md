# Vehicle Detection, Tracking and License Plate Recognition

A computer-vision pipeline that tracks vehicles in traffic video, finds the license plate of each vehicle, reads it several times, and reports one final plate text per vehicle. If it cannot detect or read a plate, it says so instead of guessing.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AwsHamed/vehicle-tracking-alpr/blob/main/notebooks/01_vehicle_tracking_alpr.ipynb)

---

## Overview

OCR on a single frame of a moving vehicle is unreliable: angle, motion blur and small plate size all cause errors. This project handles plate reading **per vehicle** instead of per frame:

1. Detect and track vehicles, so each physical vehicle keeps one ID.
2. Detect the plate inside each vehicle's bounding box.
3. Keep the best plate crops of that vehicle (largest, not cut off).
4. Read several crops with a plate-specific OCR model.
5. Choose one final text from the reads, and write it to `plates.txt` and to an annotated video.

## Pipeline

```
Video -> OpenCV -> YOLO11 vehicle detection -> ByteTrack (one ID per vehicle)
      -> plate detector (inside each vehicle crop) -> best plate crops per vehicle
      -> plate OCR (fast-plate-ocr) -> consensus read -> plates.txt + annotated video
```

The notebook follows these stages: setup, helper functions, **Stage B** (track vehicles, detect plates, store crops), **Stage C** (read crops, choose one text, write `plates.txt`), **Stage D** (render the annotated video).

## Results

All numbers come from one test video and from my own manual checks. Treat them as indicative, not as a benchmark.

**Test video:** a road-traffic clip of 1285 frames. It is not included in this repository.

### Tracking and coverage

| Item | Value |
|---|---|
| Vehicles tracked (final run) | 29 |
| Vehicles on the main road (manual count) | 24 |
| Vehicles on a side road (approximate manual count) | 5 |
| Main-road vehicles with a plate text in the output | 24 of 24 |
| Side-road vehicles | `NO_PLATE_DETECTED` (3) or `UNREADABLE` (2) |

### Plate reading accuracy

I read the plates by eye from the video for 20 of the 24 main-road vehicles. The other 4 could not be verified, so they are not scored.

| Metric (20 hand-labeled plates) | Value |
|---|---|
| Exact match (every character correct) | 8 / 20 |
| At most 1 wrong character | 18 / 20 |
| At most 2 wrong characters | 20 / 20 |
| Mean character error rate (CER) | 0.11 |

A plate counts as wrong here if **any** character differs. Most errors are a single character (for example `0` vs `O`, `8` vs `B`, `5` vs `S`).

### What each choice changed

Measured on an earlier run of the same video and the same 20 labeled plates. Lower CER is better.

| OCR setup | Single read CER | Final (consensus) CER |
|---|---|---|
| EasyOCR | 0.44 | 0.39 |
| fast-plate-ocr | 0.26 | 0.23 |
| fast-plate-ocr + prefix rule | 0.15 | 0.12 |

- Switching from a general OCR (EasyOCR) to a plate-specific OCR gave the largest gain.
- Combining several reads helped, but only modestly (0.15 to 0.12).
- Image enhancement before OCR (upscale, CLAHE, sharpen, denoise) gave no clear gain. Differences were 0.02 CER or less, which is within noise on 20 plates, so no enhancement is used.
- One box per object (`agnostic_nms=True`): without it, the vehicle detector often returned the same object twice with two classes (for example car and truck). Frames with duplicate boxes dropped from 183 to 6, and the number of track IDs from 90 to 32 (earlier run).

## Method and design decisions

- **YOLO11n for vehicles** (COCO classes car, motorcycle, bus, truck): small and fast on a free Colab GPU.
- **ByteTrack** with `track_buffer = 90`: it also matches low-confidence detections to existing tracks, so a partly hidden vehicle keeps its ID longer. It uses motion and overlap only (no appearance model), so ID switches are possible when vehicles cross.
- **Plate detection inside each vehicle crop:** this links every plate to its vehicle ID directly and keeps the search area small.
- **Best crops, not all crops:** crops are ranked by size. Plates that touch the edge of the vehicle crop (cut off), are shorter than 15 px, or have an unusual width/height ratio are ignored. The 5 best crops of each vehicle are read.
- **Consensus read:** from the reads of one vehicle, the text with the smallest total Levenshtein distance to the other reads is chosen. This is not character-by-character voting.
- **Plate-specific OCR (fast-plate-ocr):** clearly better than EasyOCR on these crops.
- **Prefix rule:** the plates in my test video start with the letter `B`, so a wrong or missing first character is corrected. This rule is specific to this video and is switched off with `PLATE_PREFIX = None`.
- **No guessing without evidence:** a vehicle with no plate detection gets `NO_PLATE_DETECTED`, and one with no usable read gets `UNREADABLE`.

## Evaluation methodology

Ground truth was written by hand from the video. A `?` marks a character I could not read with certainty and counts as a wildcard when scoring. Character error rate is the edit distance between the output and the ground truth, divided by the length of the ground truth.

Ground truth is tied to the **track IDs of one run**. IDs change when tracking settings change, so it must be checked again after any change.

The real plate texts are **not published** in this repository. The evaluation cells read a local `ground_truth.csv` (columns `track_id,plate_text`, for example `1,AB123`) and are skipped when the file is missing.

## Limitations

- One short video, one camera angle. The numbers are not a benchmark.
- The same 20 plates were used to choose the OCR engine, the prefix rule and the enhancement mode. There is no separate test set.
- Exact match is only 8 / 20: most plates are almost right, but one wrong character makes the plate wrong.
- 4 main-road vehicles have an output text that I could not verify by eye.
- The plate OCR confidence is not used (`avg_conf` in `plates.txt` is always 0.00). A read is rejected only when there is no usable crop or text, never because of low quality, so unreliable reads can still get a text.
- The prefix rule fits only the plates of this video.
- ByteTrack has no appearance model, so ID switches are possible when vehicles cross.
- Side-road vehicles are far from the camera: their plates were not detected or not readable.

## Possible improvements

- Use the OCR confidence to reject weak reads and to weight the consensus.
- Define a region of interest so only the main road is processed and counted.
- Pick crops from different moments of the track instead of the largest ones, so errors are less correlated.
- Evaluate on a second video with new hand-labeled plates.

## How to run

1. Click the **Open in Colab** badge.
2. `Runtime` -> `Change runtime type` -> select a GPU.
3. Upload your video to Colab as `/content/traffic.mp4` (or change `VIDEO_PATH` in the config cell).
4. Run sections 0-6 once, then Stage B, Stage C and (optionally) Stage D.
5. Before rendering a video you want to share, set `MASK_PLATES = True` in the config cell.

## Repository structure

```
notebooks/   01_vehicle_tracking_alpr.ipynb
README.md
requirements.txt
LICENSE
```

## Data, models, licenses and privacy

- **Test video:** a road-traffic clip. It is not included in this repository and is not redistributed.
- **Vehicle detector:** YOLO11n weights from Ultralytics (Ultralytics is licensed AGPL-3.0).
- **Plate detector:** `Nuurrr66/license-plate-detector` from Hugging Face. It is a third-party model used as it is. Its model card does not state a license or the training data. It is downloaded when the notebook runs and is not redistributed here.
- **Plate OCR:** [fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr) (MIT) with the `cct-s-v2-global-model` pretrained model. See the project's model zoo for the model's terms. EasyOCR is used only for the comparison.
- **Privacy:** license plates can identify people. This repository contains no real plate texts and no plate images. Any video or image shared from this project should be rendered with `MASK_PLATES = True` and checked by eye, because masking only covers plates the detector found.

## Acknowledgments

- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)
- ByteTrack: Zhang et al., *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*, ECCV 2022
- [fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr) by ankandrew
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)

## Author

**Aws Hamed** - [GitHub](https://github.com/AwsHamed) - [LinkedIn](https://www.linkedin.com/in/aws-hamed-080244433)

## License

MIT for the code in this repository. Models, data and libraries keep their own licenses.