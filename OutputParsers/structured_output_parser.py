from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema, OutputFixingParser
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# --- Step 1: Create Response Schemas ---
schema = [
    ResponseSchema(name="fact1", description="first fact about the black hole"),
    ResponseSchema(name="fact2", description="second fact about the black hole"),
    ResponseSchema(name="fact3", description="third fact about the black hole"),
]

parser = StructuredOutputParser.from_response_schemas(schema)

# Optional safety parser (auto-fixes bad JSON)
safe_parser = OutputFixingParser.from_llm(model, parser)

# --- Step 2: Create Prompt Template ---
template = ChatPromptTemplate.from_template(
    template=(
        "Give me 3 facts about the topic {topic}. "
        "Return ONLY valid JSON that follows this format:\n"
        "{response_format}"
    ),
    input_variables=["topic"],
    partial_variables={"response_format": parser.get_format_instructions()},
)

# --- Step 3: Build the chain ---
chain = template | model | parser  # or use safe_parser instead of parser

# --- Step 4: Invoke chain ---
result = chain.invoke({
    "topic": "black holes"
})

print("Final Facts about Black holes:\n", result)
