from django import forms

class IngForm(forms.Form):
    ing= forms.CharField(label='ing', max_length=100)