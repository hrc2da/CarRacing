import pymongo
from pymongo import MongoClient, ReturnDocument
from bson.objectid import ObjectId
from pilotsessions import sessions, session2indexdict
from config import Config

out_dir = 'pilot_responses'
questions = ["driver_approach","driver_strengths","driver_struggles","driver_design","design_description","design_rationale"]

client = MongoClient(Config.MONGO_HOST, Config.MONGO_PORT)
db = client[Config.DB_NAME]
sessions_collection = db.sessions

session_ids = [ObjectId(s) for s in sessions]
pilot_sessions = list(sessions_collection.find({"_id":{"$in":session_ids}}))


def write_responses_to_file(session_list, question):
    filename = f'{out_dir}/{question}.txt'
    responses = [session['question_answers'][question] for session in session_list]
    ordered_responses = [None]*12
    for session in session_list:
        ordered_responses[session2indexdict[str(session['_id'])]] = session['question_answers'][question]
    
    with open(filename, 'w+') as outfile:
        for r in ordered_responses:
            outfile.write(f'* {r}\n\n')


for q in questions:
    write_responses_to_file(pilot_sessions, q)


