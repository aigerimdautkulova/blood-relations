# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.db import models
from .settings import BASE_DIR

# Create your models here.
class DadInfo(models.Model):
    descriptor_dad = models.TextField()
    photo_dad = models.CharField(max_length=100)

    def __str__(self):          
        return self.descriptor_dad
    class Meta:
        verbose_name_plural = "DadInfo"

class MomInfo(models.Model):
    descriptor_mom = models.TextField()
    photo_mom = models.CharField(max_length=100)

    def __str__(self):          
        return self.descriptor_mom
    class Meta:
        verbose_name_plural = "MomInfo"

class ChildInfo(models.Model):
    descriptor_child = models.TextField()
    photo_child = models.CharField(max_length=100)
    dad = models.ForeignKey(DadInfo, on_delete=models.PROTECT)
    mom = models.ForeignKey(MomInfo, on_delete=models.PROTECT)

    def __str__(self):          
        return self.descriptor_child
    class Meta:
        verbose_name_plural = "ChildInfo"

