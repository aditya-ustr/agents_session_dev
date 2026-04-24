from transformers import pipeline, AutoTokenizer
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.output_parsers import PydanticOutputParser
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional
import os

from services.utilities import SingletonMeta


class ModelServing(metaclass=SingletonMeta):

    def __init__(self):

        if os.getenv("OPENAI_KEY"):
            self.use_model = "openai"
        elif os.getenv("HF_TOKEN"):
            self.use_model = "hf"
        else:
            raise Exception("Either set 'OPENAI_KEY' OR 'HF_TOKEN' to use hosted models")
        # self.use_model = "hf"
        print(f"\n\nUsing {self.use_model} model")

    def run_inference(self, messages, temperature=0.0, max_new_tokens=128, response_format=None):

        if self.use_model == "hf":

            hf_llm = HuggingFaceEndpoint(
                repo_id="meta-llama/Llama-3.1-8B-Instruct",
                task="conversational",
                huggingfacehub_api_token=os.getenv("HF_TOKEN"),
                temperature=temperature,
                max_new_tokens=max_new_tokens
            )
            llm = ChatHuggingFace(llm=hf_llm)

            if response_format:
                parser = PydanticOutputParser(pydantic_object=response_format)


            for msg in messages[::-1]:
                if isinstance(msg, SystemMessage):
                    msg.content +=  f"""
                        Return ONLY valid JSON.

                        {parser.get_format_instructions()}
                        """
                    break


            messages = [SystemMessage(content=system_prompt)] + messages
            response = llm.invoke(messages)

            

        else:
            llm = ChatOpenAI(
                model="gpt-4.1-mini",
                api_key=os.getenv("OPENAI_KEY"),
                temperature=temperature,
                max_tokens=max_new_tokens
            )

            if response_format:
                llm = llm.with_structured_output(response_format, include_raw=True)

            response = llm.invoke(messages)
            response = response["raw"]

        return response
    
    def run_inference_tools(self, messages, tools, temperature=0.0, max_new_tokens=128):

        if self.use_model == "hf":

            hf_llm = HuggingFaceEndpoint(
                repo_id="meta-llama/Llama-3.1-70B-Instruct",
                task="conversational",
                huggingfacehub_api_token=os.getenv("HF_TOKEN"),
                temperature=temperature,
                max_new_tokens=max_new_tokens
            )
            llm = ChatHuggingFace(llm=hf_llm)

        else:
            llm = ChatOpenAI(
                model="gpt-4.1-mini",
                api_key=os.getenv("OPENAI_KEY"),
                temperature=temperature,
                max_tokens=max_new_tokens
            )

        llm_with_tools = llm.bind_tools(tools)
        response = llm_with_tools.invoke(messages)

        return response


if __name__=="__main__":

    serving = ModelServing()

    class CandidateInfo(BaseModel):
        name: Optional[str] = Field(default=None)
        email: Optional[str] = Field(default=None)
        city: Optional[str] = Field(default=None)
        skills: List[str] = Field(default_factory=list)
        experience_years: Optional[int] = Field(default=None)

    system_prompt = f"""

    You are an information extraction assistant.

    Extract details from the user message and return ONLY valid JSON.

    """

    human_prompt = """

    Hi, my name is Rahul Sharma.
    I live in Bangalore.
    You can contact me at rahul.sharma@example.com.
    I know Python, Java, SQL, and Machine Learning.
    I have been working in software development for 3 years.

    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]

    res = serving.run_inference(messages, response_format=CandidateInfo)
    print(res)

    ## ------------------------ ##

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

    res = serving.run_inference_tools(messages, tools)
    print(res)

    ## Expected Outputs ##
    # content='{"name":"Rahul Sharma","email":"rahul.sharma@example.com","city":"Bangalore","skills":["Python","Java","SQL","Machine Learning"],"experience_years":3}' additional_kwargs={'parsed': CandidateInfo(name='Rahul Sharma', email='rahul.sharma@example.com', city='Bangalore', skills=['Python', 'Java', 'SQL', 'Machine Learning'], experience_years=3), 'refusal': None} response_metadata={'token_usage': {'completion_tokens': 39, 'prompt_tokens': 215, 'total_tokens': 254, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-4.1-mini-2025-04-14', 'system_fingerprint': 'fp_babb1d6bc3', 'id': 'chatcmpl-DYFyJPeTpW2i769thC0CWUUupbusD', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--019dc0d8-aacc-72d1-bb35-2fec4eaa38f4-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 215, 'output_tokens': 39, 'total_tokens': 254, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}
    # content='' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 43, 'prompt_tokens': 107, 'total_tokens': 150, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-4.1-mini-2025-04-14', 'system_fingerprint': 'fp_283a574ac4', 'id': 'chatcmpl-DYFyKFQJ8YQSzHzWl8rGd9qHxdD2Q', 'service_tier': 'default', 'finish_reason': 'tool_calls', 'logprobs': None} id='lc_run--019dc0d8-b18e-7f21-8dd0-de9012ccf038-0' tool_calls=[{'name': 'get_weather', 'args': {'city': 'Mumbai'}, 'id': 'call_AAraQENGM1Zfk215FKb0mCD1', 'type': 'tool_call'}, {'name': 'get_current_time', 'args': {}, 'id': 'call_OpkyNcajR4v1tYUOkCfRG6v1', 'type': 'tool_call'}] invalid_tool_calls=[] usage_metadata={'input_tokens': 107, 'output_tokens': 43, 'total_tokens': 150, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}