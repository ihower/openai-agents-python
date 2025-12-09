from dataclasses import dataclass, field, fields
from typing import Any

from openai.types.responses import ResponseCustomToolCall, ResponseFunctionToolCall

from .run_context import RunContextWrapper, TContext


def _assert_must_pass_tool_call_id() -> str:
    raise ValueError("tool_call_id must be passed to ToolContext")


def _assert_must_pass_tool_name() -> str:
    raise ValueError("tool_name must be passed to ToolContext")


def _assert_must_pass_tool_arguments() -> str:
    raise ValueError("tool_arguments must be passed to ToolContext")


def _assert_must_pass_tool_input() -> str:
    raise ValueError("tool_input must be passed to CustomToolContext")


@dataclass
class ToolContextBase(RunContextWrapper[TContext]):
    """Base class for tool call contexts."""

    tool_name: str = field(default_factory=_assert_must_pass_tool_name)
    """The name of the tool being invoked."""

    tool_call_id: str = field(default_factory=_assert_must_pass_tool_call_id)
    """The ID of the tool call."""


@dataclass
class ToolContext(ToolContextBase[TContext]):
    """The context of a function tool call with structured JSON arguments."""

    tool_arguments: str = field(default_factory=_assert_must_pass_tool_arguments)
    """The raw JSON arguments string of the function tool call."""

    @classmethod
    def from_agent_context(
        cls,
        context: RunContextWrapper[TContext],
        tool_call_id: str,
        tool_call: ResponseFunctionToolCall,
    ) -> "ToolContext[TContext]":
        """Create a ToolContext from a RunContextWrapper."""
        base_values: dict[str, Any] = {
            f.name: getattr(context, f.name) for f in fields(RunContextWrapper) if f.init
        }
        return cls(
            tool_name=tool_call.name,
            tool_call_id=tool_call_id,
            tool_arguments=tool_call.arguments,
            **base_values,
        )


@dataclass
class CustomToolContext(ToolContextBase[TContext]):
    """The context of a custom tool call with free-form string input."""

    tool_input: str = field(default_factory=_assert_must_pass_tool_input)
    """The raw input string of the custom tool call."""

    @classmethod
    def from_agent_context(
        cls,
        context: RunContextWrapper[TContext],
        tool_call_id: str,
        tool_call: ResponseCustomToolCall,
    ) -> "CustomToolContext[TContext]":
        """Create a CustomToolContext from a RunContextWrapper."""
        base_values: dict[str, Any] = {
            f.name: getattr(context, f.name) for f in fields(RunContextWrapper) if f.init
        }
        return cls(
            tool_name=tool_call.name,
            tool_call_id=tool_call_id,
            tool_input=tool_call.input,
            **base_values,
        )
