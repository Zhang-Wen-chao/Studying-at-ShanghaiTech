def rob(nums):
    n = len(nums)
    if n == 0:
        return 0
    if n == 1:
        return nums[0]
    
    dp = [0]*n
    
    # Case 1: Include first tree, exclude the last one.
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])
    for i in range(2, n - 1):
        dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
    
    max_apples = dp[n - 2]  # Keep track of maximum apples in this case.
    
    # Case 2: Exclude first tree, include the last one.
    dp[0] = 0
    dp[1] = nums[1]
    for i in range(2, n):
        dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])

    max_apples = max(max_apples, dp[n - 1])  # Compare with maximum apples in case 2.
    
    return max_apples

if __name__ == "__main__":
    nums = list(map(int, input().split()))
    print(rob(nums))