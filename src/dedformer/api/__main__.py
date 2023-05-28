import os

import uvicorn

if __name__ == '__main__':
    uvicorn.run(
        "dedformer.api.app:create_app",
        factory=True,
        host='0.0.0.0',
        port=int(os.getenv('UV_PORT', '7088')),
        workers=1,
        # log_config=,
    )