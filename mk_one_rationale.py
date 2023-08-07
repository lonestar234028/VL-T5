img_root = '/vc_data/users/taoli1/mm/nlvr/'
json_root = '/vc_data/users/taoli1/VL-T5/datasets/nlvr/test_back.json'
# json_root = 'test.json'
rationale_root = '/vc_data/users/taoli1/nlvrresearch/modelscope/reason_step_huge/reasons_v2/'
ann = {}
import json,os,sys



with open(json_root, 'r') as f:
    ann = json.load(f)
print("ann:", len(ann))
promptsall = []
id2prompt = {}

for i, file in enumerate(os.listdir(rationale_root)):
    if file.endswith('json'):
        id2prompt[i] = file
        with open(rationale_root + file) as f:
            prompts_tmp = json.load(f)
            promptsall.append(prompts_tmp)
print("prompts:", len(promptsall))
print("id2prompt:", id2prompt)
import copy
# There is an empty glass.##test1/test1-0-1-img0.png##test1/test1-0-1-img1.png
for i, prompts in enumerate(promptsall):
    # deep copy ann
    print("i:", i)
    ann_to_write = copy.deepcopy(ann)
    for j, an in enumerate(ann_to_write):
        img_key = an['sent'] + '##test1/' + an['img0'] + '.png##test1/'+ an['img1'] + '.png'
        if img_key not in prompts:
            continue
        pmts = prompts[img_key].split('##')
        if len(pmts) != 3:
            continue
        text_inner = pmts[0] + ',left image:' + pmts[1] + ', right image:' + pmts[2] \
            + '. Therefore, does it make sense:' + an['sent']
        if (pmts[1] == 'yes' or pmts[2] == 'yes' or pmts[1] == 'no' or pmts[2] == 'no'):
            continue
        if len(pmts[2]) == 0: # caption generated 
            text_inner =  'left image:' + pmts[0] + ', right image:' + pmts[1] \
            + '. Therefore, does it make sense:' + an['sent']
        if j == 79:
            print("pmts:", pmts)
            print("text_inner:", text_inner)
        an['sent'] = text_inner
    print("ann_to_write:", ann_to_write[79])
    with open('test_' + str(i)+'.json', 'w') as f:
        json.dump(ann_to_write, f)



