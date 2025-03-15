from django.shortcuts import render

# Create your views here.
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Log


@csrf_exempt
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