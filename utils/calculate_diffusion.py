import argparse
import json

def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-f', '--file', type=str)
    
    args = parser.parse_args()
    
    with open(args.file) as f:
        results = json.load(f)
        
        
    cal = 0
    for result in results:
        if 'slide' in result['pred_identity'][0]:
            cal += 1
            
    print(cal)
        
    cnt = 0
    counter = 0
    
    for result in results:
        if len(result['pred_identity']) > 1:
            cnt += 1
            if result['pred_identity'][1] == result['pred_identity'][0]:
                counter += 1
            
    
        
    print(cnt, counter)
    
    
if __name__ == '__main__':
    main()