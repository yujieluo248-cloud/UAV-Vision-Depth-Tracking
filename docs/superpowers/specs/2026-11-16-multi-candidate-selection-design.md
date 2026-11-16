# Multi-Candidate Selection V2 Design

## Intent

Extend the HSV baseline so every valid orange connected component is exposed,
then select one target using simple shape gates and optional one-frame temporal
continuity. The selected contour produces the mask that a future depth stage
would consume.

## Scope and constraints

- Preserve `ColorDetector.detect()` for existing static tools and tests.
- Keep HSV thresholding, morphology, and `min_area` behavior unchanged.
- Do not modify any HSV configuration.
- Do not add Depth, alignment, 3D, ROS, PX4, YOLO, or a full tracker.
- Do not commit or push.

## Approaches

1. Put temporal state inside `ColorDetector`: compact, but makes image
   detection stateful and couples camera behavior to reusable segmentation.
2. Add `ColorCandidate` generation plus a stateless `CandidateSelector`, with
   the camera tool owning the tiny history: selected because responsibilities
   stay clear and existing static callers remain compatible.
3. Add a tracker abstraction now: rejected because prediction/filter state is
   explicitly outside this iteration.

## Design

`ColorDetector.detect_candidates()` returns a full cleaned `color_mask` and all
`ColorCandidate` objects sorted by descending contour area. Each candidate
contains its contour, integer bounding box, floating-point center, area, aspect
ratio, and fill ratio.

`CandidateSelector.select()` first applies baseline gates for excessive bbox
coverage, low fill, invalid aspect ratio, and—when history exists—center jump
and area change. Acquisition selects the largest remaining valid candidate.
Association minimizes a transparent weighted score based primarily on
normalized center distance, then logarithmic area change and a small shape
penalty. These defaults are starting values, not calibrated final parameters.

`ColorDetector.detect()` remains a compatibility acquisition path: it generates
candidates, uses `CandidateSelector` without history, draws only the selected
contour into `target_mask`, and returns the existing `ColorDetection` fields.
The cleaned full segmentation remains separately available as `color_mask`.

The RealSense preview owns `previous_center`, `previous_area`, and
`missed_frames`. A successful selection updates history; a miss retains it for
a few frames; exceeding `max_missed_frames` resets acquisition. Candidate boxes
are drawn thin, while the selected target is emphasized.

## Known limitation

At first acquisition, color and simple geometry cannot reliably distinguish
two genuinely similar orange objects. Largest-valid is only a baseline fallback
and does not solve target identity.
