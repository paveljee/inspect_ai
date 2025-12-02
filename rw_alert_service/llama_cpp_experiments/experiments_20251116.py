#!/usr/bin/env python
# coding: utf-8

# ruff: noqa: E402

# In[5]:


import json
import sys

from llama_cpp import Llama, LlamaGrammar

log_file = open("llama_2025-11-16.log", "w")
sys.stdout = log_file
sys.stderr = log_file

# Initialize model
llm = Llama(
    model_path="/Volumes/home/anonymous/models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
    n_batch=1,  # Process one token at a time
    n_threads=1,  # Single-threaded
    seed=42,
    n_gpu_layers=-1,  # Use Metal
    n_ctx=2048,
    verbose=True,
)

# Define your JSON schema
pydantic_code = """
class UserModel(BaseModel):
    name: str
    age: int
    skills: List[str]
    email: Optional[EmailStr] = None
    user_bio: str = Field(..., description="Summarizes User info.")
""".strip()
exec(pydantic_code)
valid_schema = json.dumps(UserModel.model_json_schema(), indent=2)  # noqa: F821


# In[2]:


# Run inference with grammar-based constrained decoding
prompt = """Extract the following information in JSON format:
"John is a 28 year old software engineer who knows Python, Rust, and C++. His email is john@example.com"
Your response must be a valid JSON following strictly this Pydantic model: {}
""".format(pydantic_code)

response = llm.create_chat_completion(
    messages=[{"role": "user", "content": prompt}],
    grammar=LlamaGrammar.from_json_schema(valid_schema),
    temperature=0.7,
    max_tokens=512,
)

# Parse the response
result = json.loads(response["choices"][0]["message"]["content"])
print(json.dumps(result, indent=2))


# In[3]:


canonical_result = {
    "name": "John",
    "age": 28,
    "skills": ["Python", "Rust", "C++"],
    "user_bio": "John is a 28 year old software engineer who knows Python, Rust, and C++. His email is john@example.com",
    "email": "john@example.com",
}

display(result == canonical_result)  # noqa: F821


# Below is same code rewritten by GPT for server version.
#
# But, it messed up, so I had to fix manually.
#
# Config is in `./server_config_2025-11-16.yaml`

# In[1]:


import json
import sys

import requests
from llama_cpp.llama_grammar import json_schema_to_gbnf

log_file = open("llama_2025-11-16.log", "w")
sys.stdout = log_file
sys.stderr = log_file

# -------------------------------------------------------
# Pydantic model
# -------------------------------------------------------
pydantic_code = """
class UserModel(BaseModel):
    name: str
    age: int
    skills: List[str]
    email: Optional[EmailStr] = None
    user_bio: str = Field(..., description="Summarizes User info.")
""".strip()
exec(pydantic_code)

# llama-cpp-server grammar schema format:
valid_schema_str = json.dumps(UserModel.model_json_schema(), indent=2)  # noqa: F821
grammar = json_schema_to_gbnf(valid_schema_str)

# -------------------------------------------------------
# Prompt
# -------------------------------------------------------
user_input = "John is a 28 year old software engineer who knows Python, Rust, and C++. His email is john@example.com"

prompt = """Extract the following information in JSON format:
"{}"
Your response must be a valid JSON following strictly this Pydantic model: {}
""".format(
    user_input,
    pydantic_code,
)

# -------------------------------------------------------
# Chat completion (server)
# -------------------------------------------------------
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "llama-3.1-8b",  # <-- model_alias
        "messages": [{"role": "user", "content": prompt}],
        "grammar": grammar,
        "temperature": 0.7,
        "max_tokens": 512,
    },
)

resp = response.json()
content = resp["choices"][0]["message"]["content"]

# -------------------------------------------------------
# Parse model output
# -------------------------------------------------------
parsed = json.loads(content)
print(json.dumps(parsed, indent=2))

canonical_result = {
    "name": "John",
    "age": 28,
    "skills": ["Python", "Rust", "C++"],
    "user_bio": "John is a 28 year old software engineer who knows Python, Rust, and C++. His email is john@example.com",
    "email": "john@example.com",
}

canonical_result_2 = {
    "name": "John",  # somehow different from above!
    "age": 28,
    "skills": ["Python", "Rust", "C++"],
    "user_bio": "software engineer",
    "email": "john@example.com",
}

display(parsed == canonical_result)  # noqa: F821
display(parsed == canonical_result_2)  # noqa: F821
parsed  # noqa: B018
