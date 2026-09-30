import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "api:app",
        host=os.getenv("FRAMESEG_HOST", "0.0.0.0"),
        port=int(os.getenv("FRAMESEG_PORT", "8000")),
        reload=os.getenv("FRAMESEG_RELOAD", "0") == "1",
    )


if __name__ == "__main__":
    main()
