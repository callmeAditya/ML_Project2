#### calling external resources like mongoDB, redis, etc.

import os
import pickle
import sys

from src.exception import MyException

def save_object( path, obj):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as file:
            pickle.dump(obj, file)
    except Exception as e:
        raise MyException(e, sys)