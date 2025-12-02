from __future__ import annotations

import json
from pathlib import Path

import anyio
from cheatsheet_parser_agent import CheatsheetParser
from inspect_screening_task import (
    OUTPUT_DIR,
    dump_latest_extraction_schema_pydantic,
    extract_cheatsheet_schema,
    find_latest_metaschema,
    read_latest_metaschema,
    screen_articles,
)
from llama_cpp_experiments.llama_server import LlamaServer

from inspect_ai import (
    eval,
    eval_async,
)
from inspect_ai.model import (
    ResponseSchema,
    get_model,
)
from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.model._providers.llama_cpp_python import LlamaCppPythonAPI
from inspect_ai.util import json_schema
from inspect_ai.util._json import json_schema_to_base_model

MetaSchema = CheatsheetParser.MetaSchema
SERVER_CONFIG_PATH = Path("llama_cpp_experiments") / "server_config_20251116.yaml"
LOG_DIR = Path.cwd().parent / "logs"
SCREENING_OUTPUTS_DIR = OUTPUT_DIR / "screening"


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


def _build_eval_payload(task, model):
    model_args = (
        {"provider": {"require_parameters": True}}
        if get_model(model).name.startswith("openrouter")
        else {}
    )
    return dict(
        tasks=task, model=model, model_args=model_args, log_dir=str(LOG_DIR.resolve())
    )


def _check_eval_logs(logs):
    for log in logs:
        assert log.status == "success"
        assert log.results.scores[0].metrics["accuracy"].value == 1


def run_eval(task, model):
    payload = _build_eval_payload(task, model)
    logs = eval(**payload)
    _check_eval_logs(logs)


async def run_eval_async(task, model):
    payload = _build_eval_payload(task, model)
    logs = await eval_async(**payload)
    _check_eval_logs(logs)


def test_extract_schema():
    try:
        run_eval(
            extract_cheatsheet_schema,
            "google/gemini-2.5-pro",
        )
    finally:
        pass


def test_read_latest_metaschema():
    metaschema_instance = read_latest_metaschema()
    extraction_schema = metaschema_instance.build_extraction_schema()
    print(json.dumps(extraction_schema.model_json_schema(), indent=2))


def test_llama_response_format():
    metaschema_instance = read_latest_metaschema()
    extraction_schema = metaschema_instance.build_extraction_schema()
    llm = LlamaCppPythonAPI("fake_model")
    params = llm.completion_params(
        config=GenerateConfig(
            response_schema=ResponseSchema(
                name="FakeSchema",
                json_schema=json_schema(extraction_schema),
                strict=True,
            )
        ),
        tools=False,
    )
    print(json.dumps(params, indent=2))


def test_dump_extraction_schema_pydantic():
    pydantic_class_dump_path = dump_latest_extraction_schema_pydantic()
    assert pydantic_class_dump_path.exists()
    print(f"Extraction schema class dumped to '{pydantic_class_dump_path}' for use.")


async def test_screen():
    try:
        server = LlamaServer(SERVER_CONFIG_PATH)
        await server.start()
        await run_eval_async(
            screen_articles(
                pmids=[
                    # 10618008,
                    21592376,
                ],  # some arbitrary PMIDs
                metaschema_json=find_latest_metaschema(),
                output_dir=SCREENING_OUTPUTS_DIR,
            ),
            "llama-cpp-python/llama-3.1-8b",
        )
    finally:
        await server.stop()
