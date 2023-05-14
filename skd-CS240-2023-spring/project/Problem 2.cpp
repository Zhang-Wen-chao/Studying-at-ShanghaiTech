#include <iostream>
#include <vector>
using namespace std;

int rob(vector<int>& nums) {
    int n = nums.size();
    if (n == 0) return 0;
    if (n == 1) return nums[0];
    
    vector<int> dp(n, 0);
    
    // Case 1: Include first house, exclude the last one.
    dp[0] = nums[0];
    dp[1] = max(nums[0], nums[1]);
    for (int i = 2; i < n - 1; i++)
        dp[i] = max(dp[i - 1], dp[i - 2] + nums[i]);
    
    int max_apples = dp[n - 2];  // Keep track of maximum apples in this case.
    
    // Case 2: Exclude first house, include the last one.
    dp[0] = 0;
    dp[1] = nums[1];
    for (int i = 2; i < n; i++)
        dp[i] = max(dp[i - 1], dp[i - 2] + nums[i]);

    max_apples = max(max_apples, dp[n - 1]);  // Compare with maximum apples in case 2.
    
    return max_apples;
}

int main() {
    int n;
    cin >> n;
    vector<int> nums(n);
    for (int i = 0; i < n; i++)
        cin >> nums[i];
    
    cout << rob(nums) << endl;
    
    return 0;
}