import re

# 模型名称字符串
model_name = "whisper-high-wa_ctr_8_5_10_25_50"

# 使用正则表达式匹配字符串中的所有数字
numbers = re.findall(r'\d+', model_name)

# 将匹配到的数字字符串转换为整数
numbers_list = [int(num) for num in numbers]

# 分离出第一个数字
first_number = numbers_list[0]

# 创建除第一个数字外的数字列表
remaining_numbers = numbers_list[1:]

print("第一个数字:", first_number)
print("剩余数字列表:", remaining_numbers)