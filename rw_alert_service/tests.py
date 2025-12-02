from __future__ import annotations

import json
from pathlib import Path

import anyio
from cheatsheet_parser_agent import CheatsheetParser
from llama_cpp_experiments.llama_server import LlamaServer

from inspect_ai.model import (
    ResponseSchema,
)
from inspect_ai.util import json_schema
from inspect_ai.util._json import json_schema_to_base_model

MetaSchema = CheatsheetParser.MetaSchema
SERVER_CONFIG_PATH = Path("llama_cpp_experiments") / "server_config_20251116.yaml"
LOG_DIR = Path.cwd().parent / "logs"


def test_json_schema():
    response_schema = ResponseSchema(
        name=MetaSchema.__name__,
        json_schema=json_schema(MetaSchema),
        strict=True,
    )
    dynamic_model = json_schema_to_base_model(
        response_schema.json_schema, model_name=response_schema.name
    )
    print(json.dumps(dynamic_model.model_json_schema(), indent=2))


async def test_llama_server():
    server = LlamaServer(SERVER_CONFIG_PATH)

    await server.start()
    print("Server is running...")

    # Perform test requests here (httpx / aiohttp)
    await anyio.sleep(5)

    await server.stop()
    print("Server stopped cleanly.")
