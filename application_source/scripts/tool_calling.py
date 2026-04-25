from transformers import pipeline, AutoTokenizer
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.output_parsers import PydanticOutputParser
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional
import os


hf_llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-70B-Instruct",
    task="conversational",
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
    temperature=0.0,
    max_new_tokens=128
)
llm = ChatHuggingFace(llm=hf_llm)

## ---------

@tool
def get_weather(city: str) -> str:
    """Get current weather for a given city."""
    
    weather_data = {
        "Mumbai": "32°C, Humid",
        "Delhi": "28°C, Sunny",
        "Bangalore": "24°C, Cloudy"
    }
    
    return weather_data.get(city, "Weather data not available")


@tool
def get_current_time() -> str:
    """Get the current system time."""
    
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools = [get_weather, get_current_time]

## ---------

system_prompt = f"""

You are a helpful assistant.

Use tools whenever needed to answer user questions accurately.

Do not guess information if a tool can help.

"""

human_prompt = """

What's the weather in Mumbai and what time is it right now?

"""

## ---------

messages = [
    SystemMessage(content=system_prompt),
    HumanMessage(content=human_prompt)
]
llm_with_tools = llm.bind_tools(tools)

response = llm_with_tools.invoke(messages)

print(response)

## Output ##
# content='' additional_kwargs={'tool_calls': [{'function': {'arguments': '{"city": "Mumbai"}', 'name': 'get_weather', 'description': None}, 'id': 'chatcmpl-tool-9f512502dfa7542c', 'type': 'function'}]} response_metadata={'token_usage': {'completion_tokens': 24, 'prompt_tokens': 269, 'total_tokens': 293}, 'model_name': 'meta-llama/Llama-3.1-70B-Instruct', 'system_fingerprint': None, 'finish_reason': 'tool_calls', 'logprobs': None} id='lc_run--019dc095-a572-70b3-aaed-d69c87482e70-0' tool_calls=[{'name': 'get_weather', 'args': {'city': 'Mumbai'}, 'id': 'chatcmpl-tool-9f512502dfa7542c', 'type': 'tool_call'}] invalid_tool_calls=[] usage_metadata={'input_tokens': 269, 'output_tokens': 24, 'total_tokens': 293}
##