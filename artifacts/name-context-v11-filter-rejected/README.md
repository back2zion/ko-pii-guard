# Rejected v11 development filter

This is not a loadable checkpoint. None of 480 filter epochs simultaneously
put all train/validation gold below 0.5 non-person probability and rejected
every required false proposal at 0.99. Fresh v11 evaluation was never run.
The frozen source bytes are preserved in `training_source.txt`.
The follow-up expands the hidden layer from 64 to 128 and replays hard examples.
