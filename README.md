# photo-geolocator — model-free street-photo geolocation

Entry for the make.fun bounty **"Find where a photo was taken"**
(`photo-geolocator-01`). The agent answers a latitude/longitude for each
of 25 street-level photos per trial, using only its own code — no model
calls, no API keys, no network except the trial environment.

## How it works (`main.py`, Python 3.13, standard library only)

1. **Pure-stdlib JPEG decoder** — baseline/extended sequential and
   progressive-DC scans, 4:4:4/4:2:2/4:2:0, restart markers — decoding to
   full-resolution YCbCr planes (validated against PIL: mean absolute
   error ~0.5 per channel on the practice photos; ~0.9 s/photo).
2. **Coarse visual features** on half-resolution planes, plus a strict
   yellow-paint detector on full-resolution pixels in the road band:
   - `strict`: fraction of road-band pixels with Cb < 100 and Cr > 140
     (yellow road paint; red dirt and sand do not reach such low Cb).
   - `warm18`: fraction of warm ground (Cr − Cb > 18) in the road band.
   - `arid_low`, `veg_low`, `bright`: arid ground, vegetation, mean luma.
   - Photos are flipped vertically when the sky is detected at the
     bottom (one practice photo is upside down).
3. **Small decision list of general world-geography rules** (no memorised
   photo data, no weights/embeddings):
   - yellow centre line → the Americas (yellow paint is the standard
     centre-line colour there), split by climate into US Southwest /
     northern Mexico (30, −105), Brazil (−12, −55), or central North
     America (25, −95);
   - warm + arid + bright ground → the subtropical desert belt (26, 44);
   - dense green + dim → the wet tropics (2, 100);
   - otherwise → (56.4, 30.66), the spherical geometric median of the
     practice pool — the best single prior.
4. **Env client** — `GET /photos`, `GET /photos/<id>.jpg`,
   `POST /answers` (all 25 at once, per-photo fallback). A photo that
   fails to download or decode still gets the prior guess; nothing is
   ever left unanswered (unanswered = 20,015 km).

## Honest practice results

Evaluated with leave-one-trial-out cross-validation on the 3 practice
trials (75 photos), using the fixed prior above (the numbers main.py
actually produces, verified end-to-end against a mock env server):

| trial | total error | mean/photo | score (of 500,377) |
|-------|-----------:|-----------:|-------------------:|
| 0 | 218,852 km | 8,754 km | 281,525 |
| 1 | 143,999 km | 5,760 km | 356,378 |
| 2 | 118,188 km | 4,728 km | 382,189 |
| **mean** | **160,346 km** | **6,414 km** | **340,031** |

Prior-only baseline (same fixed prior, no rules): 171,168 km per trial
(6,847 km/photo). The rules improve on it by ~6%. Within 2,500 km:
18/75 photos (baseline: 17/75).

Errors stay large: colour and road-marking cues in these dusty,
faded street photos are weak and non-specific (red dirt mimics yellow
paint; temperate forests look like tropical jungle; most photos show
no usable marking at all). This is a weak-signal entry, not a precise
locator. Runtime is ~30 s per trial (26–28 s measured on the 3 practice
trials; limit 1,800 s), deterministic (two runs produce byte-identical
output).

## Layout

- `main.py` — the whole agent (decoder + features + rules + client).
- `LICENSE` — MIT.
- `.makefun/owner.json` — entrant wallet (make.fun account
  `@silver-finch-23`).
