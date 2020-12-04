from utils import car2config
from utils import config2car
from utils import feature_labels
from utils import feature_ranges
from utils import chopshopfeatures2indexlist


# {
#     "WHEEL_RAD" : False,
#     "WHEEL_WIDTH" : False,
#     "FRICTION_LIM" : False,
#     "ENG_POWER" : True,
#     "BRAKE_SCALAR" : True,
#     "STEERING_SCALAR" : True,
#     "REAR_STEERING_SCALAR" : False,
#     "MAX_SPEED" : False,
#     "COLOR" : False,
#     "BUMPER" : False,
#     "SPOILER" : False,
#     "REAR_BODY" : False,
#     "FRONT_BODY" : False
# }


from config import Config
import pymongo
from pymongo import MongoClient
from bson.objectid import ObjectId

client = MongoClient('localhost', 27017)
db = client[Config.DB_NAME]
sessions = db.sessions

session = sessions.find_one({"_id": ObjectId("5f8df42ad5c68de5e8185f59")},sort=[("_id",pymongo.ASCENDING)])
features = session['selected_features']
features['BUMPER'] = True
feature_indices = chopshopfeatures2indexlist(features)
assert feature_indices == [0,5,6,19,21]

features['BUMPER'] = False
features['SPOILER'] = True
feature_indices = chopshopfeatures2indexlist(features)
assert feature_indices == [0,16,17,19,21]

features['SPOILER'] = False
features['FRONT_BODY'] = True
feature_indices = chopshopfeatures2indexlist(features)
assert feature_indices == [0,8,9,19,21]

features['FRONT_BODY'] = False
features['REAR_BODY'] = True
feature_indices = chopshopfeatures2indexlist(features)
assert feature_indices == [0,11,12,13,14,19,21]


selected_features = {
    "WHEEL_RAD" : False,
    "WHEEL_WIDTH" : False,
    "FRICTION_LIM" : False,
    "ENG_POWER" : True,
    "BRAKE_SCALAR" : False,
    "STEERING_SCALAR" : True,
    "REAR_STEERING_SCALAR" : False,
    "MAX_SPEED" : True,
    "COLOR" : False,
    "BUMPER" : False,
    "SPOILER" : False,
    "REAR_BODY" : False,
    "FRONT_BODY" : False
}
    ###############################################################
mask_indices = chopshopfeatures2indexlist(selected_features, invert=True)
print(mask_indices)
assert mask_indices == [2,3,4,5,6,8,9,11,12,13,14,16,17,20,21,23]
import pdb; pdb.set_trace()