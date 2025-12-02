import json
import os
from pathlib import Path

import anyio
import uvicorn
import yaml
from llama_cpp.server.app import create_app
from llama_cpp.server.settings import ConfigFileSettings, ServerSettings


class LlamaServer:
    def __init__(self, config_file: str | Path):
        self.config_file = Path(config_file)
        self.server = None
        self._task_group = None

    async def start(self):
        if not self.config_file.exists():
            raise ValueError(f"Config file {self.config_file} does not exist!")

        # Load config
        with open(self.config_file, "rb") as f:
            if self.config_file.suffix in (".yaml", ".yml"):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        config_settings = ConfigFileSettings.model_validate_json(json.dumps(data))
        server_settings = ServerSettings.model_validate(config_settings)
        model_settings = config_settings.models

        app = create_app(
            server_settings=server_settings,
            model_settings=model_settings,
        )

        config = uvicorn.Config(
            app,
            host=os.getenv("HOST", server_settings.host),
            port=int(os.getenv("PORT", server_settings.port)),
            ssl_keyfile=server_settings.ssl_keyfile,
            ssl_certfile=server_settings.ssl_certfile,
            log_level="info",
        )
        self.server = uvicorn.Server(config)

        # Start server in AnyIO task group
        self._task_group = await anyio.create_task_group().__aenter__()
        self._task_group.start_soon(self.server.serve)
        await anyio.sleep(0.1)  # give server time to start

    async def stop(self):
        if self.server:
            self.server.should_exit = True
        if self._task_group:
            await self._task_group.__aexit__(None, None, None)
            self._task_group = None
