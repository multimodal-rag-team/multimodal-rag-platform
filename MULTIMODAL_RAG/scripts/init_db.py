"""Initialize PostgreSQL schema (no psql required). Run from project root."""

import sys
from pathlib import Path

import psycopg2

# Allow `python scripts/init_db.py` from project root without PYTHONPATH.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import settings

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "indexer" / "schema.sql"


def main() -> None:
    if not settings.database_url:
        raise SystemExit("DATABASE_URL is not set in .env")

    raw = SCHEMA_PATH.read_text(encoding="utf-8")
    lines = [line for line in raw.splitlines() if not line.strip().startswith("--")]
    sql = "\n".join(lines)
    statements = [s.strip() for s in sql.split(";") if s.strip()]

    print("Connecting to database...")
    conn = psycopg2.connect(settings.database_url)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            for statement in statements:
                cur.execute(statement)
                label = statement.split()[0:3]
                print(f"  OK: {' '.join(label)}...")
        print("Schema created successfully (pgvector + text/table/image_chunks).")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
