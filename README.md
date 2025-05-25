# Hand Tracking Mouse Control

This project aims to create a virtual mouse control system using hand tracking. The current implementation provides basic hand tracking functionality using OpenCV and MediaPipe.

## Setup

1. Make sure you have Python 3.8+ installed on your system
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the hand tracking script:
```bash
python hand_tracker.py
```

- The script will open your webcam and start tracking your hand
- Press 'q' to quit the application

## Features

- Real-time hand tracking
- Detection of 21 hand landmarks
- Visual feedback with hand landmark connections

## Next Steps

This is just the beginning! Future improvements will include:
- Mouse cursor control using hand position
- Click detection using finger gestures
- Scroll control
- Right-click functionality "# PhantomPointer" 
