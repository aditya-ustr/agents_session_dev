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
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="conversational",
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
    temperature=0.0,
    max_new_tokens=128
)
llm = ChatHuggingFace(llm=hf_llm)

class CandidateInfo(BaseModel):
    name: Optional[str] = Field(default=None)
    email: Optional[str] = Field(default=None)
    city: Optional[str] = Field(default=None)
    skills: List[str] = Field(default_factory=list)
    experience_years: Optional[int] = Field(default=None)


# Create parser for structured output
parser = PydanticOutputParser(pydantic_object=CandidateInfo)

system_prompt = f"""

You are an information extraction assistant.

Extract details from the user message and return ONLY valid JSON.

{parser.get_format_instructions()}

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

response = llm.invoke(messages)
result = parser.invoke(response)

print(response)
print(result)

## Output ##
# content='{"name": "Rahul Sharma", "city": "Bangalore", "email": "rahul.sharma@example.com", "skills": ["Python", "Java", "SQL", "Machine Learning"], "experience_years": 3}' additional_kwargs={} response_metadata={'token_usage': {'completion_tokens': 51, 'prompt_tokens': 376, 'total_tokens': 427}, 'model_name': 'meta-llama/Llama-3.1-8B-Instruct', 'system_fingerprint': 'fp_f3ef9115178b5033ce85', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--019dc08c-4f34-7803-a81e-9ea6ab100e79-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 376, 'output_tokens': 51, 'total_tokens': 427}
# name='Rahul Sharma' email='rahul.sharma@example.com' city='Bangalore' skills=['Python', 'Java', 'SQL', 'Machine Learning'] experience_years=3
##