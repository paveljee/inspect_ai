from typing import Any

from inspect_ai.util._json import json_schema_to_base_model

from .._generate_config import GenerateConfig
from .._openai import openai_completion_params
from .openai_compatible import OpenAICompatibleAPI

# from llama_cpp.llama_grammar import json_schema_to_gbnf

# def grammar_from_response_schema(response_schema: ResponseSchema) -> str:
#     """Returns a stringified grammar ready to
#     be passed as a param to chat completions."""
#     dynamic_model = json_schema_to_base_model(response_schema.json_schema)
#     # llama-cpp-server grammar schema format:
#     valid_schema_str = json.dumps(dynamic_model.model_json_schema())
#     grammar = json_schema_to_gbnf(valid_schema_str)
#     return grammar


class LlamaCppPythonAPI(OpenAICompatibleAPI):
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
    ) -> None:
        # grammar = None
        # if config.response_schema:
        #     grammar = grammar_from_response_schema(config.response_schema)
        #     config.response_schema = None
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key or "llama-cpp-python",
            config=config,
            service="llama_cpp_python",
            service_base_url="http://localhost:8000/v1",
        )

    def completion_params(self, config: GenerateConfig, tools: bool) -> dict[str, Any]:
        params = openai_completion_params(
            model=self.service_model_name(),
            config=config,
            tools=tools,
        )
        if config.response_schema is not None:
            dynamic_model = json_schema_to_base_model(
                config.response_schema.json_schema
            )
            params["response_format"] = dict(
                type="json_object",
                schema=dynamic_model.model_json_schema(),
            )
        return params
