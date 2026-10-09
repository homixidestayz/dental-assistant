import os
from dotenv import load_dotenv
load_dotenv()
from dental.llm import get_llm
from dental.case_gen import generate_case
from dental.llm import PROVIDERS
import dental.llm as L

for model in ("openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"):
    os.environ["GROQ_MODEL"] = model
    try:
        b = generate_case(difficulty="mid")
        p = b.patient
        print(f"OK   {model}")
        print(f"     patient: {p.age}{p.sex}: {p.complaint}")
        print(f"     dx: {p.intended_diagnosis}")
        print(f"     rubric phases: {len(b.rubric)}")
    except Exception as e:
        print(f"FAIL {model}: {type(e).__name__}: {str(e)[:120]}")
