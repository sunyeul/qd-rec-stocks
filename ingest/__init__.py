"""Daily point-in-time data collection for the stocks app (PoC stage A).

Collectors write one partition per source and snapshot date under
``data/snapshots/<source>/snapshot_date=YYYY-MM-DD/``. Anything that builds
item cards must read through ``ingest.reader``, which never returns data
from after the snapshot cutoff.
"""
