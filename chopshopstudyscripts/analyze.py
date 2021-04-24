import pymongo
from pymongo import MongoClient, ReturnDocument
from bson.objectid import ObjectId
from config import Config
import numpy as np

def get_test_drive_distribution(experiment_type):
    client = MongoClient(Config.MONGO_HOST, Config.MONGO_PORT)
    db = client[Config.DB_NAME]
    experiments = db.experiments
    filtered_experiments = experiments.find({'experiment_type':experiment_type})
    results = []
    flattened_results = []
    result_means = []
    for e in filtered_experiments:
        if len(e['test_drive_results']) > 0:
            results.append(e['test_drive_results'])
            flattened_results += e['test_drive_results']
            result_means.append(np.mean(e['test_drive_results']))
    
    return np.mean(result_means), np.std(result_means), np.mean(flattened_results), results


if __name__=='__main__':
    r,s,f,_ = get_test_drive_distribution('h1control')
    print(r,s,f)