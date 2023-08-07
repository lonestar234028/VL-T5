import os
import sys
import json
from collections import defaultdict
def check_pattern(f, pattern_prefix_str):
    # print(f)
    return str(f).startswith(pattern_prefix_str)

class P:
    def __init__(self) -> None:
        self.f = 0
        self.s = 0
        self.t = 0

q2votes = defaultdict(P)
ann_map = {}

input_file_path = "/vc_data/users/taoli1/VL-T5/analysis/submits/"
input_file_pattern = "submit"
ann_json_path = "/vc_data/users/taoli1/VL-T5-Ori/datasets/nlvr/test_back.json"
with open(ann_json_path,'r') as ann_lines:
    ann_tmp = json.load(ann_lines)
    for js in ann_tmp:
        id = js["identifier"]
        ann_map[id] = js["label"] == 1
is_dir = os.path.isdir(input_file_path)
if is_dir:
    files = os.listdir(input_file_path)
    files = [f for f in files if check_pattern(f, input_file_pattern)]
    print("len(files):", len(files))
    for f in files:
        with open(os.path.join(input_file_path, f),'r') as s:
            for v1 in s:
                v = v1.strip().split(',')
                
                assert len(v) == 2, "each line format is expected as: 'question_id,answer'" + ", but got:" + v1
                if 'False' == v[1]:
                    q2votes[v[0]].f =  q2votes[v[0]].f + 1
                elif 'True' == v[1]:
                    q2votes[v[0]].s =  q2votes[v[0]].s + 1
                else:
                    q2votes[v[0]].t =  q2votes[v[0]].t + 1
                    print("warning: unexpected answer:", v1)
    q2votes = sorted(q2votes.items(), key=lambda p: p[0])
    c = 0
    all_c1 = len(ann_map)
    all_c0 = len(q2votes)
    all_miss = 0
    positives = 0
    ann_positives = 0
    for q in q2votes:
        res = q[1].s > q[1].f
        # print (q[0], res, q[1].s, q[1].f, ann_map[q[0]])
        ann_positives += ann_map[q[0]]
        # print (q[0], res)
        if(not q[0] in ann_map):
            all_miss = all_miss + 1
        else:
            if res == ann_map[q[0]]:
                c = c + 1
        positives += res
    print("acc:", float(c)/all_c1, ", ref:", [all_c0, all_c1, all_miss])
    print("all miss:", all_miss)
    print("positives:", positives, positives / all_c1)
    print("ann_positives:", ann_positives, ann_positives / all_c1)
# calculate the count of 1 and 0 in ann_map
# calculate the count of 1 and 0 in q2votes
