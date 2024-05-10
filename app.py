import boto3
import json
import os

from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

from transformers import AutoTokenizer,AutoModelForCausalLM,AutoConfig
import transformers
import torch

import streamlit as st

from dotenv import load_dotenv


load_dotenv()


hf_token = os.environ.get('HF_TOKEN')

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = os.environ.get('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = 'default'


lambda_client = boto3.client(
    'lambda',
    region_name='us-east-1',
    aws_access_key_id=os.environ.get('aws_access_key_id'),
    aws_secret_access_key=os.environ.get('aws_secret_access_key'),
)


@st.cache_resource
def load_model():
    model_checkpoint = 'meta-llama/Meta-Llama-3-8B-Instruct'

    model_config = AutoConfig.from_pretrained(model_checkpoint,
                                            trust_remote_code=True,
                                            max_new_tokens=1024)

    model = AutoModelForCausalLM.from_pretrained(model_checkpoint,
                                                trust_remote_code=True,
                                                config=model_config,
                                                device_map='auto')

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    text_gen_pipeline = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    torch_dtype=torch.float16,
                    max_length=3000,
                    device_map="auto",)

    llm = HuggingFacePipeline(pipeline=text_gen_pipeline)

    return llm


def get_context(question: str):
    # Invoke the Lambda function
    response = lambda_client.invoke(
        FunctionName='prtmraginference',
        InvocationType='RequestResponse',
        Payload=json.dumps({"question": question})
    )

    print(response)
    
    # Read the Lambda function's response stream and parse it
    response_payload = response['Payload'].read()
    response_payload_dict = json.loads(response_payload)
    
    # Navigate to the retrievalResults
    results = response_payload_dict['body']['answer']['retrievalResults']

    print(results)
    
    # Initialize an empty string to store the extracted paragraph
    extracted_paragraph = ""
    
    # Loop through each result and concatenate text to a paragraph
    for result in results:
        text = result['content']['text']
        extracted_paragraph += text + " "

    # Return the concatenated paragraph
    return {"response": extracted_paragraph.strip()}


def get_answer_from_kb(query: str, context:str, llm):
    kb_prompt_template = f"""<|begin_of_text|>
<|start_header_id|>
    system
<|end_header_id|>
    You are a helpful, respectful and honest assistant designated answer questions related to the user's document.If the user tries to ask out of topic questions do not engange in the conversation.If the given context is not sufficient to answer the question,Do not answer the question.
<|eot_id|>
<|start_header_id|>
    user
<|end_header_id|>
    Answer the user question based on the context provided below
    Context :{context}
    Question: {query}
<|eot_id|>
<|start_header_id|>
    assistant
<|end_header_id|>"""

    prompt_template_kb = PromptTemplate(
        input_variables=["context", "query"], template=kb_prompt_template
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt_template_kb)
    
    result = llm_chain.run({"context":context, "query":query})

    return result


def main():
    st.title("Matrixly Bedrock RAG")

    llm=load_model()

    # Add a text input for the user to enter their query
    query = st.text_input("Enter your query here:")

    # Add a button to submit the query
    if st.button("Get Answer"):
        # Call the function to get the answer from the knowledge base
        context=get_context(query)
        context=context['response']
        answer = get_answer_from_kb(query, context, llm)

        index = answer.find("<|end_header_id|>")
        content_after_point = answer[index + len("<|end_header_id|>"):].strip()
        
        # Display the answer
        st.write("Answer from Knowledge Base:")
        st.write(content_after_point)


if __name__ == "__main__":
    main()