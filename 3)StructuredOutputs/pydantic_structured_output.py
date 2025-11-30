from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Literal,Optional
from pydantic import  BaseModel,Field

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.7)

class Rivew(BaseModel):
    key_theme=list[str]=Field(description="Write down 3 Keys Themes discussed in the review in a list ")
    summary:str=Field(description="Write a concise summary of the review in 2-3 sentences")
    sentiment:Literal["Positive","Negative","Neutral"]=Field(description="Classify the overall sentiment of the review as Positive, Negative or Neutral")
    name: Optional[str]=Field(description="If the review contains the name of the product or service being reviewed, extract and provide it here. If not present, return null.")

st_model=model.with_structured_output(Rivew,strict=True)
prompt="""
Google’s Pixel phones have never been the most powerful handsets, with their Tensor chipsets falling behind rivals in benchmarks. But surprisingly, the Google Pixel 10 series might be even more compromised than the Pixel 9 series, at least when it comes to the GPU (Graphics Processing Unit).
In a post on X, @lafaiel (via Phone Arena) shared a screenshot of a Geekbench listing for the Google Pixel 10 Pro, in which the Pixel 10 Pro achieved a GPU score of just 3,707. Higher is better here, and for comparison, the Pixel 9 Pro’s score is 9,023, while rivals like the Samsung Galaxy S25 Plus and iPhone 16 Pro achieve scores of 26,333 and 33,374 respectively.
So based on this result the Pixel 10 Pro is way behind, though the fact that it scored even less than its predecessor is especially worrying.
Performance upgrades in other areas.Now, GPU performance is only one part of the power picture, and the Pixel 10 Pro should outperform the Pixel 9 Pro on Geekbench overall, with the same source recording a single core score of 2,329 and a multi-core result of 6,358 for the similarly spec'd Google Pixel 10 Pro XL, compared to 1,948 and 4,530 for single and multi-core repectively on the Pixel 9 Pro XL.
But such a low GPU score means the Google Pixel 10 Pro might still struggle with demanding games.
Google claims the Pixel 10 series will feel faster in everday use, and perform better for AI tasks, and that’s probably true, so if you don’t care about games then this shouldn’t affect you much. But if you’re a big mobile gamer then you might want to think twice about buying the Pixel 10 Pro – or any Pixel 10 model, as they all use the same Tensor G5 chipset, so will probably all have similar GPU performance.
That said, while this looks like a legitimate Geekbench screenshot, it’s still just one result, so it’s possible this will turn out to be an outlier. It’s also feasible that Google might be able to improve the Pixel 10 Pro’s GPU performance through a software update.   
"""
response=st_model.invoke(prompt)
print(response)
"""
Prompt-Template- Defines What we want and keep the Intructions clear and specific.
Str-Paser- Ensures the clean and reliable extraction of structured data from the model's output Neet predictable string.
Chaining(Multi step purpose)-Lets us link multiple steps together to produce more and more of the advacement in the final output.
When we combine these three together we can make the ai advancements more powerfull and effective in solving complex tasks.
""" 