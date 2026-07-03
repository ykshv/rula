from __future__ import annotations

import logging
import os

import uvicorn


def main() -> None:
    logging.basicConfig(
        level=os.getenv("RU_LOCAL_AVATAR_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    uvicorn.run(
        "ru_local_avatar_agent.api.app:app",
        host=os.getenv("RU_LOCAL_AVATAR_HOST", "127.0.0.1"),
        port=int(os.getenv("RU_LOCAL_AVATAR_PORT", "46181")),
        reload=False,
    )


if __name__ == "__main__":
    main()
