from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseReasoningSummaryPartAddedEvent,
    ResponseReasoningSummaryPartDoneEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningSummaryTextDoneEvent,
    ResponseUsage,
)
from openai.types.responses.response_reasoning_item import ResponseReasoningItem, Summary
from openai.types.responses.response_reasoning_summary_part_added_event import Part as AddedEventPart
from openai.types.responses.response_reasoning_summary_part_done_event import Part as DoneEventPart
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

from agents.agent_output import AgentOutputSchemaBase
from agents.handoffs import Handoff
from agents.items import (
    ModelResponse,
    TResponseInputItem,
    TResponseOutputItem,
    TResponseStreamEvent,
)
from agents.model_settings import ModelSettings
from agents.models.interface import Model, ModelTracing
from agents.tool import Tool
from agents.tracing import SpanError, generation_span
from agents.usage import Usage


class FakeModel(Model):
    def __init__(
        self,
        tracing_enabled: bool = False,
        initial_output: list[TResponseOutputItem] | Exception | None = None,
    ):
        if initial_output is None:
            initial_output = []
        self.turn_outputs: list[list[TResponseOutputItem] | Exception] = (
            [initial_output] if initial_output else []
        )
        self.tracing_enabled = tracing_enabled
        self.last_turn_args: dict[str, Any] = {}
        self.hardcoded_usage: Usage | None = None

    def set_hardcoded_usage(self, usage: Usage):
        self.hardcoded_usage = usage

    def set_next_output(self, output: list[TResponseOutputItem] | Exception):
        self.turn_outputs.append(output)

    def add_multiple_turn_outputs(self, outputs: list[list[TResponseOutputItem] | Exception]):
        self.turn_outputs.extend(outputs)

    def get_next_output(self) -> list[TResponseOutputItem] | Exception:
        if not self.turn_outputs:
            return []
        return self.turn_outputs.pop(0)

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: Any | None,
    ) -> ModelResponse:
        self.last_turn_args = {
            "system_instructions": system_instructions,
            "input": input,
            "model_settings": model_settings,
            "tools": tools,
            "output_schema": output_schema,
            "previous_response_id": previous_response_id,
            "conversation_id": conversation_id,
        }

        with generation_span(disabled=not self.tracing_enabled) as span:
            output = self.get_next_output()

            if isinstance(output, Exception):
                span.set_error(
                    SpanError(
                        message="Error",
                        data={
                            "name": output.__class__.__name__,
                            "message": str(output),
                        },
                    )
                )
                raise output

            return ModelResponse(
                output=output,
                usage=self.hardcoded_usage or Usage(),
                response_id=None,
            )

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt: Any | None = None,
    ) -> AsyncIterator[TResponseStreamEvent]:
        self.last_turn_args = {
            "system_instructions": system_instructions,
            "input": input,
            "model_settings": model_settings,
            "tools": tools,
            "output_schema": output_schema,
            "previous_response_id": previous_response_id,
            "conversation_id": conversation_id,
        }
        with generation_span(disabled=not self.tracing_enabled) as span:
            output = self.get_next_output()
            if isinstance(output, Exception):
                span.set_error(
                    SpanError(
                        message="Error",
                        data={
                            "name": output.__class__.__name__,
                            "message": str(output),
                        },
                    )
                )
                raise output

            # Create the base response object
            response = get_response_obj(output, usage=self.hardcoded_usage)
            sequence_number = 0

            # Emit ResponseCreatedEvent first
            yield ResponseCreatedEvent(
                type="response.created",
                response=response,
                sequence_number=sequence_number,
            )
            sequence_number += 1

            # Emit ResponseInProgressEvent
            yield ResponseInProgressEvent(
                type="response.in_progress",
                response=response,
                sequence_number=sequence_number,
            )
            sequence_number += 1

            # Process each output item
            for output_index, output_item in enumerate(output):
                # Emit ResponseOutputItemAddedEvent for each item
                yield ResponseOutputItemAddedEvent(
                    type="response.output_item.added",
                    item=output_item,
                    output_index=output_index,
                    sequence_number=sequence_number,
                )
                sequence_number += 1

                # Special handling for ResponseReasoningItem
                if isinstance(output_item, ResponseReasoningItem):
                    if output_item.summary:
                        # Emit summary events for reasoning items
                        for summary_index, summary in enumerate(output_item.summary):
                            # Add summary part event
                            yield ResponseReasoningSummaryPartAddedEvent(
                                type="response.reasoning_summary_part.added",
                                item_id=output_item.id,
                                output_index=output_index,
                                summary_index=summary_index,
                                part=AddedEventPart(text=summary.text, type=summary.type),
                                sequence_number=sequence_number,
                            )
                            sequence_number += 1

                            # Text delta event (simulate streaming the summary text)
                            yield ResponseReasoningSummaryTextDeltaEvent(
                                type="response.reasoning_summary_text.delta",
                                item_id=output_item.id,
                                output_index=output_index,
                                summary_index=summary_index,
                                delta=summary.text,
                                obfuscation="fake_obfuscation",
                                sequence_number=sequence_number,
                            )
                            sequence_number += 1

                            # Text done event
                            yield ResponseReasoningSummaryTextDoneEvent(
                                type="response.reasoning_summary_text.done",
                                item_id=output_item.id,
                                output_index=output_index,
                                summary_index=summary_index,
                                text=summary.text,
                                sequence_number=sequence_number,
                            )
                            sequence_number += 1

                            # Summary part done event
                            yield ResponseReasoningSummaryPartDoneEvent(
                                type="response.reasoning_summary_part.done",
                                item_id=output_item.id,
                                output_index=output_index,
                                summary_index=summary_index,
                                part=DoneEventPart(text=summary.text, type=summary.type),
                                sequence_number=sequence_number,
                            )
                            sequence_number += 1

                # Emit ResponseOutputItemDoneEvent for each item
                yield ResponseOutputItemDoneEvent(
                    type="response.output_item.done",
                    item=output_item,
                    output_index=output_index,
                    sequence_number=sequence_number,
                )
                sequence_number += 1

            # Finally emit ResponseCompletedEvent
            yield ResponseCompletedEvent(
                type="response.completed",
                response=response,
                sequence_number=sequence_number,
            )


def get_response_obj(
    output: list[TResponseOutputItem],
    response_id: str | None = None,
    usage: Usage | None = None,
) -> Response:
    return Response(
        id=response_id or "123",
        created_at=123,
        model="test_model",
        object="response",
        output=output,
        tool_choice="none",
        tools=[],
        top_p=None,
        parallel_tool_calls=False,
        usage=ResponseUsage(
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            input_tokens_details=InputTokensDetails(cached_tokens=0),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        ),
    )
