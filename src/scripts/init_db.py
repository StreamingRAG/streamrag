from pathlib import Path
from sqlalchemy import text
from ._util import get_engine

def run_sql_file(engine, path: Path) -> None:
    sql = path.read_text(encoding="utf-8")
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))

def main():
    engine = get_engine()
    docs_dir = Path(__file__).resolve().parents[1] / "docs"
    run_sql_file(engine, docs_dir / "schema.sql")
    print("schema applied")

if __name__ == "__main__":
    main()
