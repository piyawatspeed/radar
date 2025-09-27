# Code Review

## Summary
This review captures issues discovered while inspecting the radar-fusion training pipeline in `ddp.py`.

> **Update:** The training dataset now enforces exact 15-minute alignment by intersecting radar timestamps and discarding any
> center frame lacking the full +/-15 minute context, preventing zero-filled supervision from being generated. See
> `TwoRadarFusionDataset` for the implementation details.【F:ddp.py†L1267-L1311】【F:ddp.py†L1466-L1497】

## Major Issues
1. **Central frames are paired using loose timestamp tolerance rather than exact matches.** The dataset builds its sample index from the union of all timestamps across both radars and then, for each radar independently, chooses the closest frame within a tolerance that is *at least* five minutes—even for the "current" frame (`offset_min == 0`).【F:ddp.py†L1269-L1303】 This means the supposedly simultaneous inputs can actually be drawn from different observation times (e.g., Radar A at 12:00 and Radar B at 12:04) and still be treated as a synchronized pair. The product spec calls for combining two radars from the *same time*, so this tolerance should either be zero for the central frame or the dataset should intersect timestamps so only true coincidences are used. As written, the training target is built on misaligned meteorology.

2. **Samples proceed even when one or both radars are missing the central frame.** After the tolerant lookup, if both centers are missing the code searches neighboring timestamps, but if it still fails it simply fabricates zeroed tensors for the missing views and continues.【F:ddp.py†L1474-L1539】 That violates the requirement to combine two contemporaneous radars and silently injects empty targets/inpaint masks, which will teach the model that "all zeros" is an acceptable supervision signal. The pipeline should instead drop such timestamps (or at least flag them) once a synchronized pair cannot be found.

## Recommendations
- Restrict the tolerance for the central frame to exact matches (or a user-configurable threshold that defaults to zero) and consider constructing `ts_list` from the intersection of radar timestamps.
- Treat timestamps lacking both central frames as invalid and skip them, rather than emitting zero-filled samples.

Addressing these issues will ensure the network actually learns from synchronized dual-radar observations, matching the intended training objective.
