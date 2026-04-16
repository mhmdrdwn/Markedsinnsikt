"""Markedsinnsikt AI — entry point.

Run locally:
    python main.py

Production (gunicorn):
    gunicorn app.main:server --workers 1 --threads 4 --bind 0.0.0.0:8050 --timeout 120
"""

from app.main import server  # noqa: F401 — exposes WSGI server for gunicorn

if __name__ == "__main__":
    from app.main import app
    app.run(debug=True, host="0.0.0.0", port=8050)
