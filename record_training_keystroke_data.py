import csv
import datetime
from pynput import keyboard

# Function to write data to CSV file
def write_to_csv(data):
    with open('keypress_log.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

# Function to handle key press events
def on_press(key):
    try:
        if key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
            write_to_csv(['Alt+Tab', datetime.datetime.now()])
        elif key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
            write_to_csv(['Ctrl+Tab', datetime.datetime.now()])
    except AttributeError:
        pass

# Function to handle key release events
def on_release(key):
    if key == keyboard.Key.esc:  # Stop listener on pressing 'esc' key
        return False

# Start listener for key events
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
