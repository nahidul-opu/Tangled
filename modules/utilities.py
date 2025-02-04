import numpy as np

def parse_hash_delimited_string(history:str) -> list:
   return [int(num) for num in history.split("#")]

def get_bug_count(history:str) -> int:
   return int(np.sum(parse_hash_delimited_string(history)))

def has_bug(history:str) -> bool:
   return get_bug_count(history) > 0

def has_one_bug(history:str) -> bool:
   return get_bug_count(history) == 1

def is_same_history(history1:str, history2:str) -> bool:
   return history1 == history2

def is_same_history(histories:list[str]) -> bool:
   for i in range(1, len(histories)):
      if histories[i] != histories[i-1]:
         return False
   return True

def are_bugs_from_tangled(change_list:str, history: str) -> bool:
    change_list = parse_hash_delimited_string(change_list)
    history = parse_hash_delimited_string(history)
    for change_count, has_bug in zip(change_list,history):
        if has_bug == 1 and change_count != 1:
            return True
    return False
