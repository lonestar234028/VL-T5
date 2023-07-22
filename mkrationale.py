img_root = '/vc_data/users/taoli1/mm/nlvr/'
json_root = '/vc_data/users/taoli1/VL-T5/datasets/nlvr/test.json'
# json_root = 'test.json'
rationale_root = '/vc_data/users/taoli1/nlvrresearch/modelscope/reason_step_huge/reasons_v2/'
ann = {}
import json,os,sys



with open(json_root, 'r') as f:
    ann = json.load(f)
print("ann:", len(ann))
promptsall = []
for file in os.listdir(rationale_root):
    if file.endswith('json'):
        with open(rationale_root + file) as f:
            prompts_tmp = json.load(f)
            promptsall.append(prompts_tmp)
print("prompts:", len(promptsall))

# There is an empty glass.##test1/test1-0-1-img0.png##test1/test1-0-1-img1.png
for an in ann:
    img_key = an['sent'] + '##test1/' + an['img0'] + '.png##test1/'+ an['img1'] + '.png'
    text = []
    # print(img_key)
    for prompts in promptsall:
        if img_key not in prompts:
            continue
        pmts = prompts[img_key].split('##')
        # print(pmts)
        if len(pmts) != 3:
            continue
        text_inner = pmts[0] + ',left image:' + pmts[1] + ', right image:' + pmts[2] \
            + '. Therefore, does it make sense:' + an['sent']
        if (pmts[1] == 'yes' or pmts[2] == 'yes' or pmts[1] == 'no' or pmts[2] == 'no'):
            continue
        if len(pmts[2]) == 0: # caption generated 
            text_inner =  'left image:' + pmts[0] + ', right image:' + pmts[1] \
            + '. Therefore, does it make sense:' + an['sent']
        text.append(text_inner)
    an['sent'] = text
print("ann:", ann[:1])
with open('test.json', 'w') as f:
    json.dump(ann, f)
#     input = {'image': [img1, img2], 'text': text}
#     # input = {'image': img, 'text': text}
#     print("input:", input)
#     print("len(text):", len(text))
#     result = ofa_pipe(input)
#     reason = result[OutputKeys.TEXT][0]
#     res[img_key] = reason
# ff = './answers_fid_20230714/' + args.filename +  '_part_'+ str(start) +'_' +  str(end) + '.json'
# print("writing:",ff)
# print("res:", res)
# with open(ff, 'w') as f:
#     json.dump( res, f)
# -


