from django.db import models

# Create your models here.
class Log(models.Model):
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.message[:50]
    

class PredictionLog(models.Model):
    model_choice = models.CharField(max_length=50)
    input_data = models.JSONField()
    prediction = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.created_at} - {self.model_choice}"