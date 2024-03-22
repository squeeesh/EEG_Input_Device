import socket
import csv
import struct
import time
from threading import Thread
from pynput import keyboard

server_ip = '127.0.0.1' 
server_port = 13854  

csv_file = "eeg_and_keystroke_data.csv"
csv_columns = ['Timestamp', 'Delta', 'Theta', 'Low-Alpha', 'High-Alpha', 'Low-Beta', 'High-Beta', 'Low-Gamma', 'High-Gamma', 'Keystroke']

def parse_payload(payload):
    i = 0
    data = {'Delta': 0, 'Theta': 0, 'Low-Alpha': 0, 'High-Alpha': 0,
            'Low-Beta': 0, 'High-Beta': 0, 'Low-Gamma': 0, 'High-Gamma': 0}

    while i < len(payload):
        code = payload[i]
        print(code)
        i += 1
        if code == 0x02:  # Poor Signal
            #data['Poor Signal'] = payload[i]
            print(payload(i))
            i += 1
        elif code == 0x04:  # Attention
            i += 1  # Skip length byte
            #data['Attention'] = payload[i]
            i += 1
        elif code == 0x05:  # Meditation
            i += 1  # Skip length byte
            #data['Meditation'] = payload[i]
            i += 1
        elif code == 0x81:  # EEG powers
            i += 1  # Skip length byte (24)
            for band in ['Delta', 'Theta', 'Low-Alpha', 'High-Alpha', 'Low-Beta', 'High-Beta', 'Low-Gamma', 'High-Gamma']:
                # Each value is 3 bytes big-endian
                value = struct.unpack('>I', b'\x00' + payload[i:i+3])[0]
                data[band] = value
                i += 3
        else:
            print(f"Unhandled code: {code}")
            break

    return data

keystroke_data = {'Keystroke': 0}

def keyboard_listening():
    def on_press(key):
        try:
            print("got to first try loop")
            if key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
                keystroke_data['Timestamp'] = time.time()
                keystroke_data['Keystroke'] = 1
            else:
                pass
        except AttributeError:
           pass
    def on_release(key):
        if key == keyboard.Key.esc:  # Stop listener on pressing 'esc' key
            return False

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

keyboard_thread = Thread(target=keyboard_listening)
keyboard_thread.start()

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.connect((server_ip, server_port))
    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()

        try:
            while True:
                packet = b''
                while len(packet) < 2 or packet[-2:] != b'\xaa\xaa':
                    packet += sock.recv(1)
                payload_length = sock.recv(1)[0] 
                print(packet)
                payload = packet
                checksum = sock.recv(1)[0]
                data = parse_payload(payload)
                data['Timestamp'] = time.time()
                data['Keystroke'] = keystroke_data['Keystroke']
                if data['Delta'] != 0:
                    writer.writerow(data)
                keystroke_data['Keystroke'] = 0
        except KeyboardInterrupt:
            print("Keyboard Interrupt")
