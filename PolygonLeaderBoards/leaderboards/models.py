from django.db import models


class LeaderBoards(models.Model):
    specification = models.CharField(max_length=50)
    user_id = models.IntegerField()
    user_name = models.CharField(max_length=50)
    model_name = models.CharField(max_length=60)
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()
