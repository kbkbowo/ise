import argparse
import glob
import json
import os

def main():
    
    parser = argparse.ArgumentParser(description='Generate dataset json')
    parser.add_argument('-r', '--root', type=str, required=True)
    
    args = parser.parse_args()
    
    files = glob.glob(args.root + '/*/*/*/*.mp4')
    
    files = sorted(files)
    
    dictionary = {}
    
    dictionary['files'] = files

    with open(os.path.join(args.root, 'dataset.json'), 'w') as f:
        json.dump(dictionary, f)
        
    print('number of videos:', len(files))

if __name__ == '__main__':
    main()