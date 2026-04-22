import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Constants
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MAX_ITERATIONS = 10        # max LLM call rounds before forcing end
MAX_COST_PER_RUN = 0.10    # USD hard ceiling
COST_PER_1M_INPUT = 0.27
COST_PER_1M_OUTPUT = 1.10
PROMPT_VERSION = "v1.0.0"

# config.py
def get_deepseek_client():
    return OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"
    )


