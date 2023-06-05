from itertools import islice

# 定义生成器函数 infinite_sequence()，可以生成无限长的数字序列
def infinite_sequence(start=1):
    num = start
    while True:
        yield num
        num += 1

# 查找字符串中子串的位置
def find_substring_position(string, substring):
    pos = string.find(substring)
    if pos != -1:
        return pos + 1
    else:
        return None

# 主函数
def main():
    # 初始时从1开始生成数字序列
    last_num = 1
    # 获取用户输入的要查找的子串
    substring = input()
    digits_str = ''

    while True:
        # 调用 infinite_sequence() 生成器函数，然后使用 islice() 截取前 20 个值
        seq = list(islice(infinite_sequence(last_num), 20))

        # 将列表 seq 转换为字符串并查找子串 
        digits_str += ''.join(map(str, seq))

        # 查找指定的子串，并输出其在字符串中的位置
        pos = find_substring_position(digits_str, substring)

        # 如果指定子串不存在，则更新 digits_str 变量并再次查找
        if pos is None:
            last_num += 20
            continue

        # 输出指定子串在字符串中的位置
        print(pos)
        break

if __name__ == '__main__':
    main()