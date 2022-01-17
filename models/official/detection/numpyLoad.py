import numpy as np;
import json;

pred= np.load("/workspaces/fashionpediaBenchmark/models/official/detection/projects/fashionpedia/output/outputRes.npy",
allow_pickle=True);


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return obj.decode("utf-8") 
        else:
            return super(MyEncoder, self).default(obj)

def attributeIds(outputNeurons, threshhold):
    return np.where(outputNeurons >= threshhold)

attributeThreshhold = 0.5

imgCount = pred.size;
for i in range(0, imgCount): 
    detectionCount = pred[i]['classes'].size
    allDetections = []
    for x in range(0, detectionCount):    
        data = {}
        data["image_id"] = pred[i]['image_file']
        data["category_id"] = pred[i]['classes'][x]
        data["attribute_ids"] = attributeIds(pred[i]['attributes'][x] , attributeThreshhold)
        data["segmentation"] = pred[i]['masks'][x]
        data["score"] = pred[i]['scores'][x]
        allDetections.append(data)
 

json_data = json.dumps(allDetections,cls=MyEncoder);
print(i)


