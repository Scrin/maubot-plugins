"""Main ChatGPT bot implementation."""

import json
import re
import time
import datetime
from typing import List, Dict, Tuple, Optional

from maubot import Plugin, MessageEvent
from maubot.handlers import command
from mautrix.types import TextMessageEventContent, MessageType, EventType, Format, EventID
from mautrix.util import markdown
from openai import OpenAI

from .config import Config
from .services.weather_service import get_weather
from .services.electricity_service import ElectricityService

class ChatGPTBot(Plugin):
    """A Matrix bot that interfaces with OpenAI's ChatGPT."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.assistant_replies: Dict[str, str] = {}
        self.max_messages = 100
        self.api_base = None
        self.electricity_service = None

    @classmethod
    def get_config_class(cls):
        return Config

    async def start(self) -> None:
        """Initialize the bot."""
        await super().start()
        self.config.load_and_update()
        self.electricity_service = ElectricityService(vat_multiplier=self.config["vat"])
        self.api_base = self.config.get("api-endpoint", None)
        self.api_base = self.api_base if self.api_base != '' else None

    async def chat_gpt_request(self, query: str, conversation_history: list, evt: MessageEvent, event_id: EventID) -> None:
        """Process a ChatGPT request.
        
        Args:
            query: The user's query
            conversation_history: List of previous messages
            evt: The Matrix event
            event_id: ID of the event to edit with the response
        """
        sender_name = evt["sender"]
        pattern = re.compile(r"^@([a-zA-Z0-9]+):")
        match = pattern.search(sender_name)
        filtered_name = match.group(1) if match else ""

        # Get current time in Helsinki timezone
        utc_now = datetime.datetime.now(datetime.timezone.utc)
        helsinki_offset = datetime.timedelta(hours=2)
        helsinki_now = utc_now.astimezone(datetime.timezone(helsinki_offset))
        current_date = helsinki_now.strftime("%A %B %d, %Y")
        current_time = helsinki_now.strftime("%H:%M %Z")

        messages = [
            {"role": "developer", "content": f"Your role is to be a chatbot called Matrix. Prefer metric units. Do not use latex, always use markdown Today is {current_date} and time is {current_time}."},
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "Get the current and forecasted weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city, default to Espoo, Finland",
                            }
                        },
                        "required": ["location"],
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch_electricity_prices",
                    "description": "Get the electricity prices in Finland in cents for a given date",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "date": {
                                "type": "string",
                                "description": "The date for which to get the electricity prices (can be 'today' or 'tomorrow' or a specific date formatted as YYYY-MM-DD)",
                            }
                        },
                        "required": ["date"],
                    }
                }
            }
        ]

        if conversation_history:
            messages.extend(conversation_history)

        messages.extend([{"role": "user", "name": filtered_name, "content": query}])

        # Check for model override
        pattern = re.compile("![\w-]+")
        override_model = None
        for message in messages:
            content = message["content"]
            match = pattern.search(content)
            if match:
                override_model = match.group(0)[1:]
                message["content"] = re.sub(pattern, "", content, count=1).strip()
                break

        if override_model and override_model not in self.config["allowed_models"]:
            await self._edit(evt.room_id, event_id, f"Invalid model: {override_model}")
            return

        start_time = time.time()
        max_retries = 5

        for i2 in range(4 + max_retries + 1):
            for retry in range(max_retries + 1):
                try:
                    client = OpenAI(api_key=self.config["api-key"], base_url=self.api_base)
                    if override_model in ["o1-mini", "o1", "o1-preview", "o3-mini"]:
                        messages[0]["role"] = "user"
                        chat_completion = client.chat.completions.create(
                            model=override_model,
                            messages=messages,
                            stream=True,
                        )
                    else:
                        chat_completion = client.chat.completions.create(
                            model=override_model if override_model else self.config["model"],
                            messages=messages,
                            tools=tools,
                            stream=True,
                        )
                    break
                except Exception as e:
                    if retry < max_retries:
                        print(f"Retry {retry + 1}/{max_retries}: {e}")
                        continue
                    await self._edit(evt.room_id, event_id, f"OpenAI API Error: {e}")
                    return

            collected_chunks = []
            collected_messages = []
            collected_functions = {}
            delay = 1

            for chunk in chat_completion:
                chunk_time = time.time() - start_time
                collected_chunks.append(chunk)
                chunk_message = chunk.choices[0].delta.content

                if chunk.choices[0].delta.tool_calls:
                    for tool_call in chunk.choices[0].delta.tool_calls:
                        if tool_call.id is not None:
                            tool_call_id = tool_call.id
                        if tool_call.function.name is not None:
                            collected_functions[tool_call_id] = {
                                "name": tool_call.function.name,
                                "arguments": ""
                            }
                        if tool_call.function.arguments is not None:
                            collected_functions[tool_call_id]["arguments"] += tool_call.function.arguments

                if chunk_message:
                    collected_messages.append(chunk_message)

                last_edit_time = getattr(self, "_last_edit_time", None)
                if last_edit_time is not None:
                    elapsed_time = time.time() - last_edit_time
                    if elapsed_time < delay and chunk.choices[0].finish_reason is None:
                        continue

                self._last_edit_time = time.time()
                try:
                    full_reply_content = ''.join(collected_messages)
                except:
                    pass

                if chunk.choices[0].finish_reason is None or chunk.choices[0].finish_reason == "tool_calls":
                    full_reply_content += "…"
                await self._edit(evt.room_id, event_id, f"{full_reply_content}")

            if collected_functions:
                full_reply_content = "Calling functions: "
                for tool_id, collected_function in collected_functions.items():
                    function_args = json.loads(collected_function["arguments"])
                    full_reply_content += f"{collected_function['name']}({function_args}) "
                await self._edit(evt.room_id, event_id, f"{full_reply_content}")

                available_functions = {
                    "weather": get_weather,
                    "fetch_electricity_prices": self.electricity_service.fetch_prices
                }

                for tool_id, collected_function in collected_functions.items():
                    try:
                        function_args = json.loads(collected_function["arguments"])
                        function_args["user"] = sender_name
                    except:
                        function_args = []

                    try:
                        function_to_call = available_functions[collected_function["name"]]
                        function_response = function_to_call(**function_args)
                    except Exception as e:
                        available_functions = str(list(available_functions.keys()))
                        function_response = f"Function error: {e}"

                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tool_id,
                            "type": "function",
                            "function": {
                                "name": collected_function["name"],
                                "arguments": json.dumps(function_args)
                            }
                        }]
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "name": collected_function["name"],
                        "content": function_response,
                    })

                continue

            full_reply_content = ''.join(collected_messages)
            return

    async def get_conversation_history(self, evt: MessageEvent, event_id: str) -> list:
        """Get the conversation history for a given event.
        
        Args:
            evt: The Matrix event
            event_id: The ID of the event to get history for
            
        Returns:
            List of previous messages in the conversation
        """
        history = []
        bot_name = self.config["bot-name"]
        pattern = re.compile(r"^@([a-zA-Z0-9]+):")
        userIdPattern = re.compile(fr'<a href="https://matrix\.to/#/{re.escape(bot_name)}">.*?</a>:? ?')

        while event_id:
            event = await self.client.get_event(evt.room_id, event_id)
            if event["type"] == EventType.ROOM_MESSAGE:
                sender_name = event["sender"]
                match = pattern.search(sender_name)
                filtered_name = match.group(1) if match else ""
                role = "assistant" if sender_name == bot_name else "user"
                if sender_name == bot_name:
                    content = self.assistant_replies.get(event_id, event['content']['body'])
                else:
                    content = userIdPattern.sub('', event['content']['formatted_body'] if event['content']['formatted_body'] else event['content']['body'])
                history.insert(0, {"role": role, "name": filtered_name, "content": content})

            if event.content.get("_relates_to") and event.content["_relates_to"]["in_reply_to"].get("event_id"):
                event_id = event["content"]["_relates_to"]["in_reply_to"]["event_id"]
            else:
                break

        return history

    @command.new("chatgpt", aliases=["c"], help="Chat with ChatGPT from Matrix.")
    @command.argument("query", pass_raw=True)
    async def chat_gpt_handler(self, evt: MessageEvent, query: str) -> None:
        """Handle the chatgpt command.
        
        Args:
            evt: The Matrix event
            query: The user's query
        """
        query = query.strip()
        if not query:
            await evt.reply("Please provide a message to chat with ChatGPT.")
            return

        if evt.content.get("_relates_to") and evt.content["_relates_to"]["in_reply_to"].get("event_id"):
            in_reply_to_event_id = evt.content["_relates_to"]["in_reply_to"]["event_id"]
            conversation_history = await self.get_conversation_history(evt, in_reply_to_event_id)
        else:
            conversation_history = []

        event_id = await evt.reply("…", allow_html=True)
        await self.chat_gpt_request(query, conversation_history, evt, event_id)

    @command.passive(".*")
    async def on_message(self, evt: MessageEvent, match: Tuple[str]) -> None:
        """Handle passive messages (mentions and replies).
        
        Args:
            evt: The Matrix event
            match: The regex match object
        """
        bot_name = self.config["bot-name"]
        in_reply_to_event_id = None

        if evt.content.get("msgtype") == MessageType.TEXT:
            formatted_body = evt.content["formatted_body"] if evt.content["formatted_body"] else evt.content["body"]
            pattern = re.compile(fr'<a href="https://matrix\.to/#/{re.escape(bot_name)}">.*?</a>:? ?')

            if evt.content.get("_relates_to") and evt.content["_relates_to"]["in_reply_to"].get("event_id"):
                in_reply_to_event_id = evt.content["_relates_to"]["in_reply_to"]["event_id"]

            if pattern.search(formatted_body) or in_reply_to_event_id in self.assistant_replies:
                query = pattern.sub('', formatted_body)
                conversation_history = []

                if in_reply_to_event_id:
                    conversation_history = await self.get_conversation_history(evt, in_reply_to_event_id)

                event_id = await evt.reply("…", allow_html=True)
                await self.chat_gpt_request(query, conversation_history, evt, event_id)

    async def _edit(self, room_id: str, event_id: EventID, text: str) -> None:
        """Edit a message with new content.
        
        Args:
            room_id: The Matrix room ID
            event_id: The event ID to edit
            text: The new text content
        """
        content = TextMessageEventContent(
            msgtype=MessageType.NOTICE,
            body=text,
            format=Format.HTML,
            formatted_body=markdown.render(text)
        )
        content.set_edit(event_id)
        await self.client.send_message(room_id, content)
        self.assistant_replies[event_id] = text

        if len(self.assistant_replies) > self.max_messages:
            oldest_event_id = next(iter(self.assistant_replies))
            del self.assistant_replies[oldest_event_id] 
