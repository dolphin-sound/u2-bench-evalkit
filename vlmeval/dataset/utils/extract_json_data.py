from json_repair import repair_json
from rich import print_json
import json
import re

def extract_json_data(i, raw_prediction):
    if raw_prediction == 'Failed to obtain answer via API.':
        return 'failed'
    else:
        print("Original prediction", raw_prediction)
        try:
            prediction_tmp_dict = json.loads(raw_prediction)    
            print(f'---------------{i}_route1------------------')
            print_json(raw_prediction)
            return prediction_tmp_dict
        except json.JSONDecodeError:
            try: 
                raw_prediction = repair_json(raw_prediction)
                prediction_tmp_dict = json.loads(raw_prediction)
                print(f'---------------{i}_route2------------------')
                print_json(raw_prediction)
                return prediction_tmp_dict
            except json.JSONDecodeError:
                pattern = r'```json\s*(\{.*?\})\s*```'  
                raw_prediction = re.findall(pattern, raw_prediction, flags=re.DOTALL)
                if isinstance(raw_prediction, list):
                    cleaned_prediction = raw_prediction[0] if len(raw_prediction) > 0 else raw_prediction
                else:
                    cleaned_prediction = raw_prediction
                try:
                    cleaned_prediction = repair_json(cleaned_prediction)
                    prediction_tmp_dict = json.loads(cleaned_prediction)
                    print(f'---------------{i}_route3------------------')
                    print_json(cleaned_prediction)
                    return prediction_tmp_dict
                except json.JSONDecodeError:
                    return 'failed'