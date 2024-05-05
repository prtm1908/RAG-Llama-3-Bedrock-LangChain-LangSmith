import os
import boto3

boto3_session=boto3.session.Session()
bedrock_agent_runtime_client=boto3.client('bedrock-agent-runtime')

kb_id=os.environ.get("KNOWLEDGE_BASE_ID")

def retrieve(input_text, kbId):
    print("Retrieving information for:", input_text, "from KB:", kbId)
    
    response=bedrock_agent_runtime_client.retrieve(
        knowledgeBaseId=kbId,
        retrievalQuery={
            'text': input_text
        },
        retrievalConfiguration={
            'vectorSearchConfiguration': {
                'numberOfResults': 5,
            }
        }
    )
    
    return response


def lambda_handler(event, context):
    if 'question' not in event:
        return {
            'statusCode': 400,
            'body': 'No question provided.'
        }
    
    query=event['question']
    response=retrieve(query, kb_id)
    print(response)
    
    return {
        'statusCode': 200,
        'body': {
            "question": query.strip(),
            "answer": response
        }
    }