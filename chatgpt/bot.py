"""Main ChatGPT bot implementation."""

import json
import re
import time
import datetime
from typing import List, Dict, Tuple, Optional, Any

from maubot import Plugin, MessageEvent
from maubot.handlers import command
from mautrix.types import TextMessageEventContent, MessageType, EventType, Format, EventID
from mautrix.util import markdown

from .config import Config
from .services.weather_service import get_weather
from .services.electricity_service import ElectricityService
from .services.ai_service import AIService

class ChatGPTBot(Plugin):
    """A Matrix bot that interfaces with OpenAI's ChatGPT and Mistral AI."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.assistant_replies: Dict[str, str] = {}
        self.max_messages = 100
        self.electricity_service = None
        self.ai_service = None

    @classmethod
    def get_config_class(cls):
        return Config

    async def start(self) -> None:
        """Initialize the bot."""
        await super().start()
        self.config.load_and_update()
        self.electricity_service = ElectricityService(vat_multiplier=self.config["vat"])
        
        # Initialize AI service
        openai_api_base = self.config.get("api-endpoint", None)
        openai_api_base = openai_api_base if openai_api_base != '' else None
        
        mistral_api_base = self.config.get("mistral-api-endpoint", None)
        mistral_api_base = mistral_api_base if mistral_api_base != '' else None
        
        self.ai_service = AIService(
            openai_api_key=self.config["api-key"],
            openai_api_base=openai_api_base,
            mistral_api_key=self.config["mistral-api-key"],
            mistral_api_base=mistral_api_base,
            allowed_models=self.config["allowed_models"],
            mistral_allowed_models=self.config["mistral-allowed-models"]
        )

    async def chat_gpt_request(self, query: str, conversation_history: list, evt: MessageEvent, event_id: EventID) -> None:
        """Process a chat request.
        
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
        model = self.config["model"]  # Default model
        for message in messages:
            content = message["content"]
            match = pattern.search(content)
            if match:
                model = match.group(0)[1:]
                message["content"] = re.sub(pattern, "", content, count=1).strip()
                break

        try:
            is_mistral = self.ai_service.is_mistral_model(model)
            chat_completion = self.ai_service.create_chat_completion(
                messages=messages,
                model=model,
                tools=None if is_mistral else tools,
                stream=True
            )
        except ValueError as e:
            await self._edit(evt.room_id, event_id, str(e))
            return
        except Exception as e:
            await self._edit(evt.room_id, event_id, f"API Error: {e}")
            return

        collected_messages = []
        collected_functions = {}
        delay = 1

        for chunk in chat_completion:
            processed = self.ai_service.process_chunk(chunk, is_mistral)
            chunk_message = processed["content"]
            tool_calls = processed["tool_calls"]

            if tool_calls:
                for tool_call in tool_calls:
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
                if elapsed_time < delay and not processed["finish_reason"]:
                    continue

            self._last_edit_time = time.time()
            try:
                full_reply_content = ''.join(collected_messages)
            except:
                continue

            if not processed["finish_reason"] or processed["finish_reason"] == "tool_calls":
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

            # Make another request with the function results
            await self.chat_gpt_request("", messages[:-1], evt, event_id)
        else:
            full_reply_content = ''.join(collected_messages)
            await self._edit(evt.room_id, event_id, full_reply_content)

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
