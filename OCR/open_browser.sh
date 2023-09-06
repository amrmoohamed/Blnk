#!/bin/bash

# Run the Django server in the background
python manage.py runserver &

# Wait for the server to start
sleep 5

# Open a new browser tab with the Django app
python -m webbrowser -n "http://127.0.0.1:8000/ExtractDate/"