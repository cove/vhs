VHS Plain HTML Wizard
=====================

This app is a plain HTML/CSS/JS wizard with a small Python HTTP server.
It avoids UI framework lock-in and reuses `libs/vhs_tuner_core.py` for core logic.

Run (Windows venv example)
--------------------------

1. Activate your environment.
2. Start the server:
   `venv-win\\Scripts\\python apps\\plain_html_wizard\\server.py`
3. Open:
   `http://127.0.0.1:8092`

Wizard Flow
-----------

1. Load chapter
   - choose archive and chapter
   - set frame span, sample rate (1/N), and bad batch proximity
2. Review frames
   - all sampled frames are shown in a grid
   - adjust IQR `k` to change the automatic threshold
   - click any frame to flip its manual good/bad override
3. Summary and save
   - review settings/stats
   - save BAD frame selections back to chapter metadata

How To Use The Tuner
--------------------

Step 1: Load chapter

1. Select an archive on the left.
2. Select a chapter in the chapter list.
3. Confirm `Start Frame` and `End Frame (exclusive)`.
4. Set `Sample Rate (1 / N)`:
   - `1` means inspect every frame.
   - `10` means inspect every 10th frame.
5. Set `Bad Batch Proximity`:
   - controls how much context around detected bad frames is shown.
6. Click `Next: Load Frames`.

Step 2: Review frames + threshold

- Frame colors:
  - green border = currently good
  - red border = currently bad
- Click a frame card to toggle its manual override.
- Use the `IQR k` slider to adjust auto-threshold:
  - lower `k` marks more frames as bad
  - higher `k` marks fewer frames as bad
- Use the sparkline to inspect scores:
  - drag on the sparkline to jump through the frame list
  - use the play button to auto-advance the current viewport
- Use `Fullscreen` when you need a larger review area.

Step 3: Summary + save

1. Click `Next: Summary`.
2. Confirm counts and settings.
3. Click `Save and Return to Chapters`.
4. The tuner saves updated `BAD_FRAMES` to:
   - `metadata/<archive>/chapters.ffmetadata`
   - the render pipeline then uses AviSynth `FreezeFrame` to replace those bad frames with nearby good frames.

How `BAD_FRAMES` maps to AviSynth `FreezeFrame`
-----------------------------------------------

- The tuner writes metadata only; it does not modify video pixels directly.
- During render, the AviSynth script reads chapter `BAD_FRAMES` and converts them into repair ranges.
- Each repair range becomes one or more `FreezeFrame(start, end, source)` calls.
- `source` is selected automatically from nearby clean frames:
  - prefer the next clean frame after a bad range
  - fall back to a previous clean frame if needed
- Contiguous/near-contiguous bad frames are merged so replacements are stable across bursts.

Status and cancellation
-----------------------

- During frame loading, the overlay shows progress and estimated ready time.
- Use `Cancel` on the loading overlay to stop a long load.

Practical workflow tips
-----------------------

- Start with a coarser sample (`1/5` or `1/10`) to find problem regions quickly.
- Reduce sample stride (`1/1` or `1/2`) for chapters with dense damage.
- Tune IQR first, then do manual frame-by-frame overrides.

Notes
-----

- Render extract behavior is enabled in this plain wizard for frame-exact sampling.
- This is intended for local single-user operation.
