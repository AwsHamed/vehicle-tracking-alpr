# Vehicle Detection, Tracking and License Plate Recognition

An end-to-end pipeline that takes a traffic video and, for every vehicle, produces a stable tracking ID and the text of its license plate. It runs in Google Colab and uses pretrained models (no training was done in this project).

**Output for each run**

- `plates.txt`: one line per tracked vehicle (`id,plate,valid_reads,avg_conf`)
- `output_final.mp4`: the input video with a box around each vehicle and its ID and plate text above the box

## How it works

```
Video
  -> OpenCV             read frames
  -> YOLO11n            detect vehicles (car, motorcycle, bus, truck)
  -> ByteTrack          give each vehicle a persistent ID
  -> plate detector     find the plate inside each vehicle crop
  -> store crops        keep every plate crop under the vehicle's ID
  -> select + read      keep the best crops, read them with an OCR model
  -> vote               pick the final text per vehicle
  -> plates.txt + final video
```

Key design decisions:

1. **Plate detection on vehicle crops, not on the full frame.** Plates are small, so the plate detector only sees the crop of each tracked vehicle.
2. **One final reading per vehicle, not per frame.** A single frame is unreliable (blur, small plates). All plate crops of a vehicle are collected first; the final text is decided after the whole video is processed.
3. **Crop selection.** A stored crop is read only if it is not cut off by the vehicle-crop border, is at least 15 px tall, and has a width/height ratio between 2.0 and 5.5. The 5 largest remaining crops are read.
4. **Voting by agreement.** Among the readings of a vehicle, the one with the smallest total edit distance to all the others is chosen (a median-string choice), which removes random single-frame errors.
5. **Class-agnostic NMS.** With default settings the detector often returned two boxes on the same vehicle (for example `car` and `truck`), and the tracker gave each box its own ID. Enabling `agnostic_nms=True` fixes this (see Results).
6. **Final video rendered in a separate pass.** The final plate text is only known after voting, so the tracker output (boxes per frame) is stored and the video is rendered afterwards. No model is re-run, so IDs stay identical.

## Models

| Task | Model |
|---|---|
| Vehicle detection | YOLO11n (Ultralytics, COCO classes 2, 3, 5, 7) |
| Tracking | ByteTrack (Ultralytics), `track_buffer = 90` |
| Plate detection | `Nuurrr66/license-plate-detector` (Hugging Face, YOLO11-based, class `License_Plate`) |
| Plate OCR (default) | `fast-plate-ocr`, model `cct-s-v2-global-model` |
| Plate OCR (alternative) | EasyOCR (`OCR_ENGINE = "easyocr"`) |

## Results

Evaluated on one traffic video (1,285 frames). All numbers below were measured on this video only.

**Tracking**

| Setting | Tracked IDs | Frames with overlapping boxes (IoU > 0.5) |
|---|---|---|
| Default NMS | 90 | 183 |
| `agnostic_nms=True` | **32** | **6** |

Changing only the NMS setting removed 58 of the 90 IDs: they came from duplicate boxes on the same vehicle (for example a `car` box and a `truck` box).

**Plate output for all 32 tracked vehicles** (this includes vehicles on the side of the road, which are outside the evaluation scope below)

| Outcome | Vehicles |
|---|---|
| Plate text produced | 25 |
| Plate detected but not readable (`UNREADABLE`) | 4 |
| No plate detected (`NO_PLATE_DETECTED`) | 3 |

**Evaluation scope and manual verification**

The evaluation covers the vehicles on the **main road**, which is the area of interest of this project. It does not cover vehicles passing on the side of the road: they appear very small and only for a short time, so a human cannot read their plates in the video and the OCR output for them cannot be verified. They are tracked and may receive a plate text, but they are not counted in any number below.

The author compared the system output with the plates visible in the video:

- 22 main-road vehicles with plates legible to a human observer were evaluated.
- **17 of these 22 (77%)** were read exactly.
- The other 5 were close but not identical (a small number of characters differ).

How to read these numbers:

- The scope is defined by where the vehicle is and by whether a human can read its plate, not by what the system output.
- The sample is small (22 plates), it comes from the same video that was used during development, and there is no separate test set.
- The 77% is the accuracy on legible main-road plates. It is not an accuracy figure for every tracked vehicle.
- A character-level error rate was not recomputed after the labels were corrected.

## Limitations

- **Resolution is the main limit.** Many plate crops are only 15 to 55 px tall. Vehicles whose plates are smaller than that end up `UNREADABLE`.
- **`PLATE_PREFIX = "B"` is specific to this video.** The plates here start with `B`; a rule that drops a junk character before the `B` or adds a missing `B` is applied to each reading. For any other camera or country, set `PLATE_PREFIX = None`.
- **Single video, small evaluation** (see above).
- **Side-of-road vehicles are not evaluated.** They are tracked and may get a plate text, but that text is unverified.
- **Possible ID fragmentation.** A vehicle that disappears for a long time can still get a new ID, which creates a second line in `plates.txt`.
- **`avg_conf` is `0.00`** with the default OCR engine (its confidence is not extracted). It is only populated with EasyOCR.
- Models are pretrained and were not fine-tuned on this camera.

## Run it

1. Open the notebook in Google Colab and select a GPU runtime (Runtime, Change runtime type, T4).
2. Upload your video and set `VIDEO_PATH` in the config cell (the video is not included in this repository).
3. Run the cells in order: setup, imports, config, tracker config, models, helpers, then:
   - **Stage B**: track vehicles, detect plates, store crops (slowest step)
   - **Stage C**: read the crops, vote, write `plates.txt`
   - **Stage D**: render `output_final.mp4`
4. Re-run the vehicle-model cell (`model = YOLO(...)`) before every new run, because the tracker state is stored inside the model object.

Dependencies:

```
pip install ultralytics lap easyocr rapidfuzz "fast-plate-ocr[onnx]"
```

## Main settings

| Variable | Value | Meaning |
|---|---|---|
| `CONF_THRESHOLD` | 0.3 | Vehicle detection confidence |
| `TRACK_BUFFER` | 90 | Frames a lost vehicle is remembered (about 3 s at 30 FPS) |
| `AGNOSTIC_NMS` | `True` | One box per object even if two classes were predicted |
| `PLATE_CONF` | 0.25 | Plate detection confidence |
| `OCR_ENGINE` | `"fast"` | `"fast"` (plate-specific) or `"easyocr"` |
| `OCR_TOP_K` | 5 | Plate crops read per vehicle |
| `MIN_PLATE_H` | 15 | Minimum plate height in pixels |
| `MIN_RATIO` / `MAX_RATIO` | 2.0 / 5.5 | Accepted plate width/height ratio |
| `MIN_TEXT_LEN` | 4 | Shortest accepted OCR reading |
| `PLATE_PREFIX` | `"B"` | Dataset-specific format rule (`None` to disable) |
| `ENHANCE_MODE` | `"none"` | Optional image enhancement before OCR (not evaluated) |

## Output format

`plates.txt`:

```
id,plate,valid_reads,avg_conf
24,B065FY,5,0.00
30,UNREADABLE,0,0.00
45,NO_PLATE_DETECTED,0,0.00
```

`valid_reads` is the number of readings that passed the length filter before voting; low values (1 to 3) mean the plate is less reliable.

## Future work

- Evaluate on a held-out set of plates that were not used during development.
- Recompute a character-level error rate and compare OCR engines again using the corrected labels.
- Fine-tune the plate OCR on more labeled plates from this camera.
- Reduce ID fragmentation (for example by merging IDs that read the same plate).
- Add per-class statistics (vehicle counts by type).

## Privacy and licenses

License plates can identify people. Before sharing the output video or screenshots publicly, blur the plates or crop them out. Check the license of the input video and of the third-party models (YOLO11 / Ultralytics, the Hugging Face plate detector, `fast-plate-ocr`, EasyOCR) before redistributing anything.
