import pytest
from pydantic import BaseModel, ValidationError
from test_helpers.utils import (
    force_runapi,  # decorator equivalent to --runapi CLI flag
    skip_if_no_openrouter,
)

from inspect_ai import (
    Task,
    eval,
    task,
)
from inspect_ai.dataset import (
    Sample,
)
from inspect_ai.model import GenerateConfig, ResponseSchema
from inspect_ai.model._model import get_model
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import (
    TaskState,
    generate,
    use_tools,
)
from inspect_ai.tool import (
    ToolFunction,
    tool,
)
from inspect_ai.util import json_schema

### Structured outputs ###


class Color(BaseModel):
    red: int
    green: int
    blue: int


@task
def rgb_color():
    return Task(
        dataset=[
            Sample(
                input="What is the RGB color for white?",
                target="255,255,255",
            )
        ],
        solver=generate(),
        scorer=score_color(),
        config=GenerateConfig(
            response_schema=ResponseSchema(
                name="color",
                json_schema=json_schema(Color),
                description=Color.__doc__,
                strict=True,
            )
        ),
    )


@scorer(metrics=[accuracy(), stderr()])
def score_color():
    async def score(state: TaskState, target: Target):
        try:
            color = Color.model_validate_json(state.output.completion)
            if f"{color.red},{color.green},{color.blue}" == target.text:
                value = CORRECT
            else:
                value = INCORRECT
            return Score(
                value=value,
                answer=state.output.completion,
            )
        except ValidationError as ex:
            return Score(
                value=INCORRECT,
                answer=state.output.completion,
                explanation=f"Error parsing response: {ex}",
            )

    return score


def eval_binary_openrouter_require_params(task, model):
    model_args = {"provider": {"require_parameters": True}}
    log = eval(task, model=model, model_args=model_args)[0]
    assert log.status == "success"
    assert log.results.scores[0].metrics["accuracy"].value == 1


def check_color_structured_output(model):
    eval_binary_openrouter_require_params(rgb_color(), model)


class Cell(BaseModel):
    paper_id: str
    column_name: str
    cell_value: str


class Table(BaseModel):
    cell_values: list[Cell]


@task
def nested_pydantic():
    return Task(
        dataset=[
            Sample(
                input="Please produce a Table object with three Cell objects "
                + "(you can use whatever values you want for paper_id, column_name, and cell_value)",
            )
        ],
        solver=generate(),
        scorer=score_table(),
        config=GenerateConfig(
            response_schema=ResponseSchema(
                name="table",
                json_schema=json_schema(Table),
                description=Table.__doc__,
                strict=True,
            )
        ),
    )


@scorer(metrics=[accuracy(), stderr()])
def score_table():
    async def score(state: TaskState, target: Target):
        try:
            table = Table.model_validate_json(state.output.completion)
            value = INCORRECT
            if len(table.cell_values) > 0:
                cell = table.cell_values[0]
                if cell.cell_value and cell.column_name and cell.paper_id:
                    value = CORRECT
            return Score(
                value=value,
                answer=state.output.completion,
            )
        except ValidationError as ex:
            return Score(
                value=INCORRECT,
                answer=state.output.completion,
                explanation=f"Error parsing response: {ex}",
            )

    return score


def check_nested_pydantic_output(model):
    eval_binary_openrouter_require_params(nested_pydantic(), model)


### Tool calling ###


@tool
def tool_get_stock_price():
    async def execute(symbol: str) -> float:
        """
        Gets the current stock price for a symbol.

        Args:
            symbol: The stock symbol (e.g. "AAPL")

        Returns:
            The current stock price in USD.
        """
        if symbol == "AAPL":
            return 150.0
        elif symbol == "GOOGL":
            return 2800.0
        else:
            return 0.0

    return execute


class Stock(BaseModel):
    symbol: str
    price: float


@scorer(metrics=[accuracy(), stderr()])
def score_stock(validation_class):
    async def score(state: TaskState, target: Target):
        try:
            stock = validation_class.model_validate_json(state.output.completion)
            target_stock = validation_class.model_validate_json(target.text)
            if (
                stock.symbol == target_stock.symbol
                and stock.price == target_stock.price
            ):
                value = CORRECT
            else:
                value = INCORRECT
            return Score(
                value=value,
                answer=state.output.completion,
            )
        except ValidationError as ex:
            return Score(
                value=INCORRECT,
                answer=state.output.completion,
                explanation=f"Error parsing response: {ex}",
            )

    return score


@task
def simple_custom_tool_calling(dataset, scorer):
    return Task(
        dataset=dataset,
        solver=[
            use_tools(
                [tool_get_stock_price()],
                tool_choice=ToolFunction(name="tool_get_stock_price"),
            ),
            generate(),
        ],
        scorer=scorer,
    )


def check_tool_calling(model):
    TARGET = '{"symbol": "AAPL", "price": 150.0}'
    eval_binary_openrouter_require_params(
        simple_custom_tool_calling(
            dataset=[
                Sample(
                    input="What is the stock price of Apple?",
                    target=TARGET,
                ),
                Sample(
                    input="What is the stock price of Apple? You must respond by using `get_stock_price` tool.",
                    target=TARGET,
                ),
            ],
            scorer=score_stock(Stock),
        ),
        model,
    )


### pytest ###


@pytest.fixture(
    params=[
        "openrouter/qwen/qwen3-4b:free",
        "openrouter/openai/gpt-oss-20b:free",
        "openrouter/qwen/qwen3-235b-a22b:free",
    ]
)
def model_name(request):
    return get_model(
        request.param,
        config=GenerateConfig(
            max_tokens=8192,  # otherwise BadRequestError: ...\\\'max_completion_tokens\\\' is too large: 36839. This model\\\'s  maximum context length is 32768 tokens and your request has 729 input tokens (36839 > 32768 - 729).
        ),
    )


@force_runapi  # equivalent to --runapi CLI flag
@skip_if_no_openrouter
def test_openrouter_structured_output(model_name):
    check_color_structured_output(model_name)
    check_nested_pydantic_output(model_name)


@force_runapi  # equivalent to --runapi CLI flag
@skip_if_no_openrouter
def test_openrouter_tool_calling(model_name):
    check_tool_calling(model_name)
