# Definition of Parallel Chain:
# A parallel chain allows multiple chains to run simultaneously.
# Their outputs are independent, improving efficiency and modularity.

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

prompt1 = PromptTemplate(
    template="Generate short and simple notes for the following topic:\n{topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate 5 question-answer pairs for the following text:\n{text}",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template=(
        "Combine the following content but DO NOT summarize it.\n"
        "Keep the original structure intact as:\n\n"
        "=== NOTES ===\n"
        "{notes}\n\n"
        "=== QUESTIONS & ANSWERS ===\n"
        "{qa}\n\n"
        "Give the output in the same two-section format."
    ),
    input_variables=["notes", "qa"]
)

parser = StrOutputParser()

runnable_chain = RunnableParallel({
    "notes": prompt1 | model | parser,
    "qa": prompt2 | model | parser
})
final_chain = prompt3 | model | parser

chain = runnable_chain | final_chain
topic = """
Support vector machines (SVMs) are a set of supervised learning methods used for 
classification, regression and outlier detection.

Advantages:
- Effective in high dimensional spaces.
- Works well when number of dimensions > number of samples.
- Uses support vectors → memory efficient.
- Versatile: different kernel functions.

Disadvantages:
- Risk of overfitting when features >> samples.
- No direct probability estimates.
- Sparse/dense format requirements in scikit-learn.
"""

response = chain.invoke(topic)
print(response)
