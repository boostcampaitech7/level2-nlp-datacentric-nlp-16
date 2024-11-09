import re

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def calculate_ratio(text):
    """
    Calculate korean character ratio in text to find the noisy text

    Args:
        text (str): input text

    Returns:
        float: korean character ratio
    """
    total_chars = len(text)
    if total_chars == 0:
        return (0, 0)

    korean_chars = len(re.findall(r"[가-힣]", text))
    korean_ratio = korean_chars / total_chars

    return korean_ratio


def denoise_text(texts, model_id, template):
    """
    Denoise noisy text using LM with prompt engineering

    Args:
        texts (List[str]): input texts
        model_id (str): huggingface model id
        template (str): prompt template

    Returns:
        List[str]: denoised texts
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256, device=0, batch_size=1)
    llm = HuggingFacePipeline(pipeline=pipe)
    llm = ChatHuggingFace(llm=llm)

    prompt = PromptTemplate.from_template(template)
    chain = {"text": RunnablePassthrough()} | prompt | llm.bind(stop=[r"\n"]) | StrOutputParser()

    output_txts = []
    removed = 0
    for text in tqdm(texts):
        answer = chain.invoke(text)

        ## answer extraction in prompt
        answer = answer.split("\n").pop()

        ### for case that the answer is less than or equal to 5 in length
        if len(answer) <= 5:
            removed += 1
            answer = text

        output_txts.append(answer)
    print(f"{removed}-rows with lengths less than or equal to 5 are returned, and they remain unchanged")
    return output_txts
