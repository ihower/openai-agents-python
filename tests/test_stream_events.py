import asyncio
import time

import pytest
from openai.types.responses.response_reasoning_item import ResponseReasoningItem, Summary

from agents import Agent, HandoffCallItem, Runner, function_tool
from agents.extensions.handoff_filters import remove_all_tools
from agents.handoffs import handoff
from agents.items import ReasoningItem, ToolCallItem

from .fake_model import FakeModel
from .test_responses import get_function_tool_call, get_handoff_tool_call, get_text_message


def get_reasoning_item() -> ResponseReasoningItem:
    return ResponseReasoningItem(
        id="rid",
        type="reasoning",
        summary=[Summary(text="thinking", type="summary_text")]
    )


@function_tool
async def foo() -> str:
    await asyncio.sleep(3)
    return "success!"


@pytest.mark.asyncio
async def test_stream_events_main():
    model = FakeModel()
    agent = Agent(
        name="Joker",
        model=model,
        tools=[foo],
    )

    model.add_multiple_turn_outputs(
        [
            # First turn: a message and tool call
            [
                get_text_message("a_message"),
                get_function_tool_call("foo", ""),
            ],
            # Second turn: text message
            [get_text_message("done")],
        ]
    )

    result = Runner.run_streamed(
        agent,
        input="Hello",
    )
    tool_call_start_time = -1
    tool_call_end_time = -1
    async for event in result.stream_events():
        if event.type == "run_item_stream_event":
            if event.item.type == "tool_call_item":
                tool_call_start_time = time.time_ns()
            elif event.item.type == "tool_call_output_item":
                tool_call_end_time = time.time_ns()

    assert tool_call_start_time > 0, "tool_call_item was not observed"
    assert tool_call_end_time > 0, "tool_call_output_item was not observed"
    assert tool_call_start_time < tool_call_end_time, "Tool call ended before or equals it started?"


@pytest.mark.asyncio
async def test_stream_events_main_with_handoff():
    @function_tool
    async def foo(args: str) -> str:
        return f"foo_result_{args}"

    english_agent = Agent(
        name="EnglishAgent",
        instructions="You only speak English.",
        model=FakeModel(),
    )

    model = FakeModel()
    model.add_multiple_turn_outputs(
        [
            [
                get_text_message("Hello"),
                get_function_tool_call("foo", '{"args": "arg1"}'),
                get_handoff_tool_call(english_agent),
            ],
            [get_text_message("Done")],
        ]
    )

    triage_agent = Agent(
        name="TriageAgent",
        instructions="Handoff to the appropriate agent based on the language of the request.",
        handoffs=[
            handoff(english_agent, input_filter=remove_all_tools),
        ],
        tools=[foo],
        model=model,
    )

    result = Runner.run_streamed(
        triage_agent,
        input="Start",
    )

    handoff_requested_seen = False
    agent_switched_to_english = False

    async for event in result.stream_events():
        if event.type == "run_item_stream_event":
            if isinstance(event.item, HandoffCallItem):
                handoff_requested_seen = True
        elif event.type == "agent_updated_stream_event":
            if hasattr(event, "new_agent") and event.new_agent.name == "EnglishAgent":
                agent_switched_to_english = True

    assert handoff_requested_seen, "handoff_requested event not observed"
    assert agent_switched_to_english, "Agent did not switch to EnglishAgent"


@pytest.mark.asyncio
async def test_run_item_stream_event_order():
    """Test that reasoning_item_created events are emitted immediately after OUTPUT_ITEM.DONE (reasoning),
    not delayed until after tool calls. Also test that raw events are emitted immediately before processing logic,
    ensuring early streaming for both reasoning items and tool calls."""
    model = FakeModel()
    agent = Agent(
        name="ReasoningAgent",
        model=model,
        tools=[foo],
    )

    # Create a response with reasoning item followed by tool call
    model.add_multiple_turn_outputs(
        [
            # First turn: reasoning item, tool call
            [
                get_reasoning_item(),
                get_function_tool_call("foo", ""),
            ],
            # Second turn: final message
            [get_text_message("done")],
        ]
    )

    result = Runner.run_streamed(
        agent,
        input="Hello",
    )

    events = []
    async for event in result.stream_events():
        events.append(event)


    # Find indices of key events using both item-based and name-based approaches
    reasoning_item_indices = [
        i for i, event in enumerate(events)
        if (event.type == "run_item_stream_event" and isinstance(event.item, ReasoningItem))
    ]


    tool_call_item_indices = [
        i for i, event in enumerate(events)
        if (event.type == "run_item_stream_event" and isinstance(event.item, ToolCallItem))
    ]

    raw_event_indices = [
        i for i, event in enumerate(events)
        if event.type == "raw_response_event"
    ]

    # 在 events[1].data.response.output 是
    # [ResponseReasoningItem, ResponseFunctionToolCallItem]
    first_raw_event_indices = raw_event_indices[0]
    last_raw_event_indices = raw_event_indices[-1]

    reasoning_item_index = reasoning_item_indices[0]
    tool_call_item_index = tool_call_item_indices[0]

    # Raw events should come before reasoning item events
    assert first_raw_event_indices < reasoning_item_index, (
        f"Raw events (index {first_raw_event_indices}) should come before "
        f"reasoning_item_created (index {reasoning_item_index})"
    )

    # Raw events should also come before tool call events
    assert first_raw_event_indices < tool_call_item_index, (
        f"Raw events (index {first_raw_event_indices}) should come before "
        f"tool_call_item (index {tool_call_item_index})"
    )

    # Raw events should come before reasoning item events
    assert reasoning_item_index < last_raw_event_indices, (
        f"Reasoning events (index {reasoning_item_index}) should come before "
        f"last_raw_event (index {last_raw_event_indices})"
    )

    # Raw events should come before tool call events
    assert tool_call_item_index < last_raw_event_indices, (
        f"Tool call events (index {tool_call_item_index}) should come before "
        f"last_raw_event (index {last_raw_event_indices})"
    )