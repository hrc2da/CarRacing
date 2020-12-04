from flask import Flask
from flask_pymongo import PyMongo

app = Flask(__name__)
import pdb; pdb.set_trace()
app.config["MONGO_URI"] = "mongodb://localhost:27017/myDatabase"
mongo = PyMongo(app)
import pdb; pdb.set_trace()
mongo.db.create_collection("sessions")
mongo.db.sessions.insert_one({
    'id': 0,
    'videos': ['test.mp4'],
    'config': {'hp':400}
})
import pdb; pdb.set_trace()