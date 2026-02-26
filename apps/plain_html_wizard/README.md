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
   - set IQR `k`, apply, and toggle frame labels between good/bad
3. Summary and save
   - review settings/stats
   - save BAD frame selections back to chapter metadata

Notes
-----

- `Use Step6 extract` behavior is enabled in this plain wizard for frame-exact sampling.
- This is intended for local single-user operation.
