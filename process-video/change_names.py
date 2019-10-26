import os
from pathlib import Path

count = 0 #left pad this

downloadDir = os.path.join(os.getcwd(), 'videos')
dirs = Path(downloadDir).glob('*/')
for folder in dirs:
	for _,_,files in os.walk(folder):
		for file in files:
			name = file.split('.')
			name[0] = str(count)
			new_name = '.'.join(name)
			old_name_path = os.path.join(folder, file)
			new_name_path = os.path.join(folder, new_name)
			os.rename(old_name_path, new_name_path)
	os.rename(folder, os.path.join(os.path.dirname(folder), str(count)))
	count += 1