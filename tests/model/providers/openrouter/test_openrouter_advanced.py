import pytest
from pydantic import BaseModel, Field
from test_helpers.utils import (
    force_runapi,
    skip_if_no_openrouter,
)
from test_openrouter_basic import (
    eval_binary_openrouter_require_params,
    score_stock,
    simple_custom_tool_calling,
)

from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig, ResponseSchema
from inspect_ai.model._model import get_model
from inspect_ai.util import json_schema


class StockAdvanced(BaseModel):
    """Always respond in Thai."""

    symbol: str
    price: float
    summary: str = Field(..., description="Human-readable summary of the response.")


def check_tool_call_with_structured_output(model_name: str) -> None:
    from textwrap import dedent

    TARGET = StockAdvanced(
        symbol="AAPL",
        price=150.0,
        summary="No Thai text here because response_format was *not* added to prompt on user's end.",
    ).model_dump_json()
    response_schema = ResponseSchema(
        name=StockAdvanced.__name__,
        json_schema=json_schema(StockAdvanced),
        description=StockAdvanced.__doc__,
        strict=True,
    )
    eval_binary_openrouter_require_params(
        simple_custom_tool_calling(
            dataset=[
                Sample(
                    input="What is the stock price of Apple?",
                    target=TARGET,
                ),
                Sample(
                    input=dedent("""What is the stock price of Apple? You must respond by using `get_stock_price` tool. Your response must be a valid JSON object conforming to this Pydantic class:

                    ```python
                    class StockAdvanced(BaseModel):
                        symbol: str
                        price: float
                        summary: str
                    ```
                    """),
                    target=TARGET,
                ),
                Sample(
                    input=f"""What is the stock price of Apple? You must respond by using `get_stock_price` tool. Your response must be a valid JSON object conforming to this JSON Schema:

```json
{response_schema.model_dump_json(indent=2)}
```""",
                    target=TARGET,
                ),
            ],
            scorer=score_stock(StockAdvanced),
        ),
        get_model(
            model_name,
            config=GenerateConfig(
                max_tokens=8192,  # otherwise BadRequestError: ...\\\'max_completion_tokens\\\' is too large: 36839. This model\\\'s  maximum context length is 32768 tokens and your request has 729 input tokens (36839 > 32768 - 729).
                response_schema=response_schema,
            ),
        ),
    )


@force_runapi  # equivalent to --runapi CLI flag
@skip_if_no_openrouter
@pytest.mark.parametrize(
    "model_name",
    [
        "openrouter/qwen/qwen3-4b:free",
        # "openrouter/openai/gpt-oss-20b:free",
        # "openrouter/qwen/qwen3-235b-a22b:free",
    ],
)
def test_openrouter_tool_calling_with_structured_output(model_name):
    check_tool_call_with_structured_output(model_name)
