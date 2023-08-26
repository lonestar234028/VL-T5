import json
ori_json_path = './datasets/vqa/OpenEnded_mscoco_val2014_questions.json'
ori_json_path = './datasets/vqa/OpenEnded_mscoco_train2014_questions.json'
ori_json = []
with open(ori_json_path, 'r') as f:
    ori_json = json.load(f)
print("ori_json: ", len(ori_json))
example_str_prefix = 'COCO_val2014_'
example_str_prefix = 'COCO_train2014_'
example_str_surfix = '.jpg'
exaple_str = '000000004011'
example_len = len(exaple_str)
for ann in ori_json:
    x = (str(ann['image_id']))
    need_to_add = example_len - len(x)
    #  make a string contains all 0, and len is need_to_add
    add_str = '0' * need_to_add
    new_img_id = example_str_prefix + add_str + x
    ann['img_id'] = new_img_id
    # print(new_img_id)

with open('./datasets/vqa/OpenEnded_mscoco_train2014_questions_new.json', 'w') as f:
    json.dump(ori_json, f)