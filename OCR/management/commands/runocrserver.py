from django.core.management.base import BaseCommand
from django.core.management import call_command

class Command(BaseCommand):
    help = 'Runs the OCR server more elegantly'

    def handle(self, *args, **options):
        # Call the runserver command with your desired options
        call_command('runserver', '0.0.0.0:8000')