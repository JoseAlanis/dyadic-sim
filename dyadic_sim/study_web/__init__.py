"""Offline clinician rating packages for simulated sessions.

One self-contained HTML file per clinician, opened from disk in a browser,
rated, saved to a JSON file and sent back. No server, no login, no model
calls. See study_web/README.md.
"""

APP_VERSION = "0.1.0"
SCHEMA_VERSION = 1
PACKAGE_FORMAT = "transcript-rating-package"
RETURN_FORMAT = "transcript-rating-return"
