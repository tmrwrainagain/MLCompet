"""Entry point: start the FastAPI server."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from module_G.config import API_HOST, API_PORT

if __name__ == "__main__":
    import uvicorn
    print(f"Starting API server at http://{API_HOST}:{API_PORT}")
    print(f"Swagger UI: http://127.0.0.1:{API_PORT}/docs")
    uvicorn.run("module_G.api:app", host=API_HOST, port=API_PORT, reload=False)
