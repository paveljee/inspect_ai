import pytest
from pydantic import BaseModel, ValidationError
from test_helpers.utils import (
    skip_if_no_google,
    skip_if_no_mistral,
    skip_if_no_openai,
    skip_if_no_openrouter,
)

from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample
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
from inspect_ai.solver import TaskState, generate, tool
from inspect_ai.util import json_schema


@tool
def get_stock_price(symbol: str) -> float:
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


class Stock(BaseModel):
    symbol: str
    price: float


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
            response_schema=ResponseSchema(name="color", json_schema=json_schema(Color))
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


def check_structured_output(task, model):
    log = eval(task, model=model)[0]
    assert log.status == "success"
    assert log.results.scores[0].metrics["accuracy"].value == 1


def check_color_structured_output(model):
    check_structured_output(rgb_color(), model)


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
            response_schema=ResponseSchema(name="table", json_schema=json_schema(Table))
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
    check_structured_output(nested_pydantic(), model)


@task
def tool_calling_pydantic():
    return Task(
        dataset=[
            Sample(
                input="What is the stock price of Apple?",
                target='{"symbol": "AAPL", "price": 150.0}',
            )
        ],
        tools=[get_stock_price],
        solver=generate(),
        scorer=score_stock(),
        config=GenerateConfig(
            response_schema=ResponseSchema(
                name="stock", json_schema=json_schema(Stock)
            )
        ),
    )


@scorer(metrics=[accuracy(), stderr()])
def score_stock():
    async def score(state: TaskState, target: Target):
        try:
            stock = Stock.model_validate_json(state.output.completion)
            target_stock = Stock.model_validate_json(target.text)
            if stock.symbol == target_stock.symbol and stock.price == target_stock.price:
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


def check_tool_calling_pydantic_output(model):
    check_structured_output(tool_calling_pydantic(), model)


@skip_if_no_openai
def test_openai_structured_output():
    check_color_structured_output("openai/gpt-4o-mini")
    check_nested_pydantic_output("openai/gpt-4o-mini")


@skip_if_no_openai
def test_openai_responses_structured_output_color():
    model = get_model("openai/gpt-4o-mini", responses_api=True)
    check_color_structured_output(model)


@skip_if_no_openai
@pytest.mark.flaky
def test_openai_responses_structured_output_pydantic():
    # This test is flaky since is relies on the model returning objects of the expected
    # shape. This often happens, but it's common for the shape to differ quite a bit
    model = get_model("openai/gpt-4o-mini", responses_api=True)
    check_nested_pydantic_output(model)


@skip_if_no_google
def test_google_structured_output():
    check_color_structured_output("google/gemini-2.0-flash")
    check_nested_pydantic_output("google/gemini-2.0-flash")


@skip_if_no_mistral
def test_mistral_structured_output():
    check_color_structured_output("mistral/mistral-large-latest")
    check_nested_pydantic_output("mistral/mistral-large-latest")


@skip_if_no_openrouter
def test_openrouter_structured_output():
    check_color_structured_output("openrouter/openai/gpt-oss-20b:free")
    check_nested_pydantic_output("openrouter/openai/gpt-oss-20b:free")
    check_color_structured_output("openrouter/qwen/qwen3-235b-a22b:free")
    check_nested_pydantic_output("openrouter/qwen/qwen3-235b-a22b:free")


@skip_if_no_openrouter
def test_openrouter_structured_output_tool_calling():
    check_tool_calling_pydantic_output("openrouter/openai/gpt-oss-20b:free")
    check_tool_calling_pydantic_output("openrouter/qwen/qwen3-235b-a22b:free")
