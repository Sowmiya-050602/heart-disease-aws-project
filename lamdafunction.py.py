import json
import os
import boto3
import uuid
from datetime import datetime, timezone

sm = boto3.client('sagemaker-runtime')
db = boto3.resource('dynamodb')

table = db.Table('HeartDiseasePredictions')

ENDPOINT = os.environ['ENDPOINT_NAME']

FEATURE_COLS = [
    'age','sex','cp','trestbps','chol','fbs',
    'restecg','thalach','exang','oldpeak',
    'slope','ca','thal'
]

def lambda_handler(event, context):

    body = json.loads(event.get('body', '{}'))

    row = [str(body.get(f, 0)) for f in FEATURE_COLS]
    payload = ",".join(row)

    resp = sm.invoke_endpoint(
        EndpointName=ENDPOINT,
        ContentType='text/csv',
        Body=payload
    )

    score = float(resp['Body'].read().decode().strip())

    risk = "High" if score >= 0.5 else "Low"

    pid = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    table.put_item(Item={
        'prediction_id': pid,
        'timestamp': now,
        'prediction': str(round(score, 4)),
        'risk_label': risk,
        **{k: str(v) for k, v in body.items()},
        'ttl': int(datetime.now().timestamp()) + 7776000
    })

    return {
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps({
            'score': score,
            'risk': risk,
            'id': pid
        })
    }