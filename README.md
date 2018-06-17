# a3c-player
An implementation of the A3C algorithm used with OpenAI's retro emulator. 
Currently does not work on windows due to a crash when using both pytorch and retro.
Works on nearly all atari roms, still experimental for sonic roms.

Command format:
python a3c-player.py (args)

args:
--env (environment name/rom name) (default: pong-v4)
--render (true/false) (default: false) (enable to only use one thread and to watch model play)
--sonic (true/false) (default: false) (enable if using sonic rom)

Delete folders with name of environment if you want to restart training.
Models are saved periodically and added to their respective folders.
You can run tensorboard against the runs folder in each environment's folder to view training progress realtime.


To-Do list:
1. Train greenhill act1 model on greenhill act2, see how long convergence takes. 
2. Submit act2 model to leaderboard, compare score to act1 model
3. Apply filters from sonic_utils in retro baselines repo, they seem helpful
4. Train blank model on greenhill act 2, compare time to act1 model
5. Train any greenhill model on different stage, see how long convergence takes.
