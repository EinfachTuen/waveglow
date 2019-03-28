"""Converts .au to .wav file using the sox tool.
IN: Paths to directory consisting of .au files.
OUT: Complete conversion of files in respective directories passed as inputs.
Run instructions:
python convert-to-wav.py path_dir_1 path_dir_2 ... path_dir_N
Where path_dir_i consists of .au files to be converted
NOTE: 
1. .au files will be DELETED. Make sure you have a backup of it to be safe.
2. Use ONLY absolute paths.
3. Run as per instruction or will lead to disastrous results
"""

import sys
import os
from pathlib import Path
# Store all command line args in genre_dirs
genre_dirs = sys.argv[1:]
target_dir = sys.argv[2:]

for genre_dir in genre_dirs:


	# loop through each file in current dir
	for file in os.listdir(genre_dir):
		# SOX
		path = Path(__file__).parent.absolute()
	
		print(format(path))
		print('path')
		file = format(path)+'/'+genre_dir+'/'+file
		output = format(path)+'/'+target_dir+'/'+file
		print(file)
		os.system("sox " + str(file) + " -r 16000 " + str(file[:-4]) + ".wav")

print("Conversion complete. Check respective directories.")