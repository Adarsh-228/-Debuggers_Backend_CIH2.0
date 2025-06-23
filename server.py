from flask import Flask, request, render_template
from flask_socketio import SocketIO, emit
import base64
import cv2
import numpy as np
# Assuming exercise.py is in the same directory
from exercise import ExerciseTracker
import os  # For path joining

app = Flask(__name__, template_folder=os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'templates'))
app.config['SECRET_KEY'] = 'your_secret_key_here!'  # Change this!
socketio = SocketIO(app, cors_allowed_origins="*")  # Allow all origins for dev

# Global tracker instance (consider per-client if handling multiple users uniquely)
# For a single user app, one instance is fine.
tracker = None
current_exercise_type = "bicep_curl"  # Default


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('connect')
def handle_connect():
    global tracker, current_exercise_type
    print('Client connected', request.sid)
    if tracker is None:
        print(
            f"Initializing new tracker with exercise: {current_exercise_type}")
        tracker = ExerciseTracker(current_exercise_type)
    else:
        print(
            f"Re-assigning exercise to existing tracker: {current_exercise_type}")
        tracker.exercise_type = current_exercise_type
        tracker.reset_counter()
    emit('connection_ack', {
         'message': 'Connected! Select exercise & start.', 'current_exercise': current_exercise_type})


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected', request.sid)
    # Optional: Consider if tracker should be reset or deallocated if no clients for a while
    # global tracker
    # tracker = None


@socketio.on('message')  # Generic message handler for JSON objects
def handle_json_message(jsonData):
    global tracker, current_exercise_type
    print(f"Received JSON: {jsonData}")

    event_type = jsonData.get('event_type')

    if event_type == 'select_exercise':
        exercise_type = jsonData.get('exercise_type')
        if exercise_type:
            current_exercise_type = exercise_type
            if tracker:
                tracker.exercise_type = current_exercise_type
                tracker.reset_counter()
                print(
                    f"Exercise changed to: {current_exercise_type} for client {request.sid}")
                emit('exercise_changed', {
                     'message': f'Exercise set to {current_exercise_type}', 'current_exercise': current_exercise_type})
            else:
                # This case should ideally not happen if tracker is initialized on connect
                tracker = ExerciseTracker(current_exercise_type)
                print(
                    f"Tracker was None, initialized with {current_exercise_type}")
                emit('exercise_changed', {
                     'message': f'Tracker initialized. Exercise: {current_exercise_type}', 'current_exercise': current_exercise_type})
        else:
            emit('error', {'message': 'Invalid exercise selection data'})

    elif event_type == 'process_frame':
        if tracker is None:
            print("Tracker is None, initializing with default for process_frame")
            # Fallback initialization
            tracker = ExerciseTracker(current_exercise_type)

        frame_b64 = jsonData.get('image_data')
        if not frame_b64:
            emit('server_error', {'message': 'No image_data provided'})
            return

        try:
            if ',' in frame_b64:
                header, encoded_data = frame_b64.split(',', 1)
            else:
                encoded_data = frame_b64

            frame_bytes = base64.b64decode(encoded_data)

            landmarks_data, exercise_status, annotated_frame_b64 = tracker.process_frame_for_server(
                frame_bytes)

            if landmarks_data is None and exercise_status and 'error' in exercise_status:
                emit('frame_processed', {
                    'landmarks': [],
                    'exercise_status': exercise_status,
                    'annotated_frame': None
                })
            else:
                emit('frame_processed', {
                    'landmarks': landmarks_data,
                    'exercise_status': exercise_status,
                    # 'annotated_frame': annotated_frame_b64 # Client-side drawing is preferred for web
                })

        except base64.binascii.Error as b64_error:
            print(f"Base64 decoding error: {b64_error}")
            emit('server_error', {
                 'message': f'Base64 decoding error: {str(b64_error)}'})
        except Exception as e:
            print(f"Error processing frame: {e}")
            emit('server_error', {
                 'message': f'Error processing frame: {str(e)}'})
    else:
        print(f"Unknown event_type: {event_type}")
        emit('error', {'message': f'Unknown event type: {event_type}'})


if __name__ == '__main__':
    print("Starting Flask-SocketIO server on http://0.0.0.0:5000")
    # use_reloader=False is important for Flask-SocketIO when not using a production Gunicorn setup
    # especially on Windows or if you see issues with duplicate initializations.
    socketio.run(app, host='0.0.0.0', port=5000, debug=True,
                 use_reloader=False, allow_unsafe_werkzeug=True)
