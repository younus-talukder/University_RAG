import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retriever import Retriever
from src.generator import generate_answer

questions = [
    "CSE 101 কোর্সের কোর্স কোড এবং শিরোনাম কী?",
    "Computer Fundamentals and Programming কোর্সের প্রধান উদ্দেশ্যগুলো কী?",
    "CSE 101-এর CLO 1 কী?",
    "CSE 101 course-er course code ebong title ki?",
    "Computer Fundamentals and Programming course-er main objectives-gulo ki?",
    "CSE 101-er CLO 1 ki?",
]

retriever = Retriever()

for q in questions:
    context = retriever.retrieve(q)
    answer = generate_answer(q, context)
    print("QUESTION:", repr(q))
    print("ANSWER:", repr(answer))
    print("---")
