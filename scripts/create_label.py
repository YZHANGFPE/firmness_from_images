import os
import yaml

image_root = '../saved_images/'
res = {}
for folder in os.listdir(image_root):
    res[folder] = 0

with open('label.yaml', 'w') as outfile:
        outfile.write( yaml.dump(res, default_flow_style=False))
