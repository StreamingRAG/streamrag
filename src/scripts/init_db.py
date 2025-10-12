from pathlib import Path
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

def run_sql_file(engine, path: Path) -> None:
    sql = path.read_text(encoding="utf-8")
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))

def main():
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise SystemExit("DATABASE_URL missing in .env")
    engine = create_engine(db_url, future=True, pool_pre_ping=True)
    run_sql_file(engine, Path("src/docs/schema.sql"))
    print("schema applied")

if __name__ == "__main__":
    main()
