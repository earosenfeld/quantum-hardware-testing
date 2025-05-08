import random
import time

def simple_logger(msg):
    print(f"[LOG] {time.strftime('%Y-%m-%d %H:%M:%S')} - {msg}")

def inject_random_error(probability):
    return random.random() < probability
