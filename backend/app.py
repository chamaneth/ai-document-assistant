import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.main import app
from core.config import settings

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.SERVER_HOST, port=settings.SERVER_PORT)