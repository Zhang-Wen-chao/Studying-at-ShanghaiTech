def longest_valid_parentheses(s):
    stack = [-1] # initialize stack with -1
    max_length = 0
    for i in range(len(s)):
        if s[i] == '(':
            stack.append(i)
        else:
            stack.pop()
            if len(stack) == 0:
                stack.append(i)
            else:
                length = i - stack[-1]
                max_length = max(max_length, length)
    return max_length

# example usage
input_str = ")()())"
output = longest_valid_parentheses(input_str)
print("Input:", input_str)
print("Output:", output)