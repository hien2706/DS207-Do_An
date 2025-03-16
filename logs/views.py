from django.shortcuts import render
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Log, PredictionLog
from .predictor import LoanApprovalPredictor
import logging

# Import DRF decorators and drf-yasg tools.
from rest_framework.decorators import api_view
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

# Import your custom API key decorator.
from .decorators import require_api_key

@csrf_exempt
@require_api_key
def log_api(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            message = data.get('message', '')
            if not message:
                return JsonResponse({'error': 'No message provided.'}, status=400)
            log = Log.objects.create(message=message)
            return JsonResponse({
                'id': log.id,
                'message': log.message,
                'created_at': log.created_at.isoformat()
            })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    elif request.method == 'GET':
        logs = list(Log.objects.all().values('id', 'message', 'created_at'))
        return JsonResponse(logs, safe=False)
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)

# Define Swagger schemas for the prediction endpoint.
loan_request_schema = openapi.Schema(
    type=openapi.TYPE_OBJECT,
    properties={
        'model_choice': openapi.Schema(
            type=openapi.TYPE_STRING,
            description='Which model to use: KNN, Decision_Tree, Logistic_Regression, or Random_Forest',
            enum=['KNN', 'Decision_Tree', 'Logistic_Regression', 'Random_Forest'],
        ),
        'data': openapi.Schema(
            type=openapi.TYPE_ARRAY,
            items=openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    'person_income': openapi.Schema(
                        type=openapi.TYPE_NUMBER, 
                        description='Annual income of the applicant (typical range: 12000-72000)',
                        example=55000
                    ),
                    'person_home_ownership': openapi.Schema(
                        type=openapi.TYPE_STRING,
                        description='Home ownership status',
                        enum=['RENT', 'OWN', 'MORTGAGE', 'OTHER'],
                        example='RENT'
                    ),
                    'loan_amnt': openapi.Schema(
                        type=openapi.TYPE_NUMBER,
                        description='Loan amount requested (typical range: 1000-35000)',
                        example=10000
                    ),
                    'loan_intent': openapi.Schema(
                        type=openapi.TYPE_STRING,
                        description='Purpose of the loan',
                        enum=['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'],
                        example='HOMEIMPROVEMENT'
                    ),
                    'loan_int_rate': openapi.Schema(
                        type=openapi.TYPE_NUMBER,
                        description='Annual interest rate of the loan (typical range: 10.0-20.0)',
                        example=13.5
                    ),
                    'loan_percent_income': openapi.Schema(
                        type=openapi.TYPE_NUMBER,
                        description='Loan amount as a percentage of income (typical range: 0.01-0.66)',
                        example=0.20
                    ),
                    'previous_loan_defaults_on_file': openapi.Schema(
                        type=openapi.TYPE_STRING,
                        description='Whether the applicant has defaulted on previous loans',
                        enum=['Yes', 'No'],
                        example='No'
                    ),
                },
                required=['person_income', 'person_home_ownership', 'loan_amnt',
                          'loan_intent', 'loan_int_rate', 'loan_percent_income',
                          'previous_loan_defaults_on_file']
            ),
            description='A list of input rows for prediction.'
        )
    },
    required=['data'],
)

loan_response_schema = openapi.Schema(
    type=openapi.TYPE_OBJECT,
    properties={
        'predictions': openapi.Schema(
            type=openapi.TYPE_ARRAY,
            items=openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    'value': openapi.Schema(
                        type=openapi.TYPE_INTEGER,
                        description='Prediction result code: 0=Rejected, 1=Approved',
                        enum=[0, 1]
                    ),
                    'description': openapi.Schema(
                        type=openapi.TYPE_STRING,
                        description='Human-readable prediction result',
                        enum=['approved', 'rejected']
                    ),
                }
            ),
            description='A list of prediction results with 0=Rejected or 1=Approved.'
        )
    }
)

@csrf_exempt
@require_api_key
@swagger_auto_schema(
    method='post',
    request_body=loan_request_schema,
    responses={200: loan_response_schema}
)
@api_view(['POST'])
def predict_loan_status(request):
    """
    Predicts the loan status (approved or rejected) based on input data.
    """
    if request.method == 'POST':
        try:
            payload = json.loads(request.body)
            # Check if payload is a list or a dict
            if isinstance(payload, list):
                model_choice = "KNN"  # default model if only a list is provided
                input_data = payload
            elif isinstance(payload, dict):
                model_choice = payload.get("model_choice", "KNN")
                input_data = payload.get("data")
                if input_data is None:
                    return JsonResponse({"error": "No 'data' field provided in the request."}, status=400)
            else:
                return JsonResponse({"error": "Invalid payload format."}, status=400)
            
            # Instantiate the predictor with the chosen model.
            predictor = LoanApprovalPredictor(scaler_path="logs/model_files/scaler.pkl", model_choice=model_choice)
            predictions = predictor.predict(input_data)
            
            # Map predictions: 1 to "approved", 0 to "rejected"
            tuned_predictions = [
                {"value": pred, "description": "approved" if pred == 1 else "rejected"}
                for pred in predictions
            ]
            
            # Log the request into PostgreSQL.
            log_entry = PredictionLog.objects.create(
                model_choice=model_choice,
                input_data=input_data,
                prediction=tuned_predictions
            )
            logging.info("Prediction log saved with id: %s", log_entry.id)
            
            return JsonResponse({"predictions": tuned_predictions})
        except Exception as e:
            logging.error("Prediction API error: %s", str(e))
            return JsonResponse({"error": str(e)}, status=400)
    else:
        return JsonResponse({"error": "Only POST requests are allowed."}, status=405)
