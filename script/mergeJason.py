import json

def merge_json_files(file_paths):
    merged_dict = {'train': {}, 'val': {}}

    for file_path in file_paths:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)

            # 合并 Train 部分
            merged_dict['train'].update(data.get('train', []))

            # 合并 Val 部分
            merged_dict['val'].update(data.get('val', []))

    return merged_dict

# 使用示例
file_paths = ['dataset/CrossLoc/train_outPlace.json', 'dataset/CrossLoc/train.json'] #, 'dataset/CrossLoc/Synthesis.json']
merged_data = merge_json_files(file_paths)

# 将合并后的字典保存到新的 JSON 文件
output_json_path = 'dataset/CrossLoc/merged_dataset.json'
with open(output_json_path, 'w') as output_json_file:
    json.dump(merged_data, output_json_file)

print("合并后的字典已保存到:", output_json_path)
