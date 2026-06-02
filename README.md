# shiny-detector

An AI powered sound detector. Similar to the earlier version, it automates clicking when it detects a sound.

The controller creates a sliding window of 4 seconds with steps of 1 second. In each iteration, a sample is recorded and fed into the CNN network. The network utilizes a pre-trained model that was designed to listen to a specific frequency using a mel spectrogram.

When a specific sound is detected, in this case the shiny sound effect, the automated clicking terminates, allowing users to take control.