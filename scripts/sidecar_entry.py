import logging

import uvicorn
from sr_engine.api.app import app

if __name__ == "__main__":
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")