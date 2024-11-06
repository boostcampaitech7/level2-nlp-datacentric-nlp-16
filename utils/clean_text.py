import re

import transformers
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def calculate_ratio(text):
    total_chars = len(text)
    if total_chars == 0:
        return (0, 0)

    special_chars = len(re.findall(r"[^가-힣0-9a-zA-Z\s.]", text))
    korean_chars = len(re.findall(r"[가-힣]", text))

    special_ratio = special_chars / total_chars
    korean_ratio = korean_chars / total_chars
    return special_ratio, korean_ratio


def denoise_text(texts, model_id, template):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256, device=0, batch_size=1)
    llm = HuggingFacePipeline(pipeline=pipe)
    llm = ChatHuggingFace(llm=llm)

    prompt = PromptTemplate.from_template(template)
    chain = {"title": RunnablePassthrough()} | prompt | llm.bind(stop=[r"\n"]) | StrOutputParser()

    output_txts = []
    removed = 0
    for text in tqdm(texts):
        answer = chain.invoke(text)

        ## answer line extraction in prompt
        answer = answer.split("\n").pop()
        ## answer phrase extraction in answer line
        ### for case that the answer is empty
        try:
            answer = answer.split('"')[1::2].pop()
        except:
            removed += 1
            answer = text
        ### for case that the answer is less than or equal to 5 in length
        if len(answer) <= 5:
            removed += 1
            answer = text

        output_txts.append(answer)
    print(f"{removed}-rows returned less than or equal to 5 in length, so they remain unchanged")
    return output_txts
