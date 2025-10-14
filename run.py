from pathlib import Path
import sys
import uvicorn

def ensure_src_on_path():
    project_root = Path(__file__).resolve().parent
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

# Pass the app as an import string at host and port with auto-reload enabled
def main():
    ensure_src_on_path()
    uvicorn.run("streamrag.api:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    main()
