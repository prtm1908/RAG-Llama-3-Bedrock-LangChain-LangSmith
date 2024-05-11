import boto3
import json
import os

from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Bedrock

import streamlit as st

from dotenv import load_dotenv


load_dotenv()


os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = os.environ.get('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = 'default'

session = boto3.session.Session()

lambda_client = session.client(
    'lambda',
    region_name='us-east-1',
    aws_access_key_id=os.environ.get('aws_access_key_id'),
    aws_secret_access_key=os.environ.get('aws_secret_access_key'),
)

llm = Bedrock(
    model_id="meta.llama3-8b-instruct-v1:0",
    region_name='us-east-1'
)


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

    # Add a text input for the user to enter their query
    query = st.text_input("Enter your query here:")

    # Add a button to submit the query
    if st.button("Get Answer"):
        # Call the function to get the answer from the knowledge base
        context=get_context(query)
        context=context['response']
        answer = get_answer_from_kb(query, context, llm)
        
        # Display the answer
        st.write("Answer from Knowledge Base:")
        st.write(answer)


if __name__ == "__main__":
    main()