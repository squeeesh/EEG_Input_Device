The ultimate goal is to take in EEG data and output a keystroke combination (e.g. ctrl+tab) that will enable the user to move between windows or tabs.

To use: with a Mindset EEG headset connected, run $FILE to generate the paired EEG and keyboard shortcut data. While you are using it, it will record your EEG waveform values and the keyboard shortcuts you use. When you have several hours of training data, run the Train_LSTM jupyter notebook. Then, incorporate the trained model into a script using the pyautogui module to switch windows and tabs (to do).
