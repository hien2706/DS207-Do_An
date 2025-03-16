from django.conf import settings
from django.http import JsonResponse
import functools

def require_api_key(view_func):
    @functools.wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        provided_key = request.META.get("HTTP_X_API_KEY")
        expected_key = getattr(settings, "API_KEY", None)
        if expected_key is None:
            # Optionally, if no API key is set in settings, allow access.
            return view_func(request, *args, **kwargs)
        if provided_key != expected_key:
            return JsonResponse({"error": "Invalid API key."}, status=403)
        return view_func(request, *args, **kwargs)
    return _wrapped_view
