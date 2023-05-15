#include <vector>
#include <algorithm>
#include <iostream>
using namespace std;

class Solution {
public:
    int longestIncreasingPath(vector<vector<int>>& matrix) {
        if (matrix.empty()) return 0;
        int n = matrix.size(), m = matrix[0].size();
        vector<vector<int>> dp(n, vector<int>(m, 0));
        int res = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                res = max(res, dfs(matrix, dp, i, j, n, m));
            }
        }
        return res;
    }
    
    int dfs(vector<vector<int>>& matrix, vector<vector<int>>& dp, int i, int j, int n, int m) {
        if (dp[i][j]) return dp[i][j];
        vector<vector<int>> dirs{{0,1}, {0,-1}, {1,0}, {-1,0}};
        int max_len = 1;
        for (auto& dir : dirs) {
            int x = i + dir[0], y = j + dir[1];
            if (x < 0 || x >= n || y < 0 || y >= m || matrix[x][y] <= matrix[i][j]) continue;
            int len = 1 + dfs(matrix, dp, x, y, n, m);
            max_len = max(max_len, len);
        }
        dp[i][j] = max_len;
        return dp[i][j];
    }
};


int main() {
    int n;
    cin >> n; // Read the size of the matrix
    vector<vector<int>> matrix(n, vector<int>(n));
    
    // Read the matrix
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            cin >> matrix[i][j];
        }
    }
    
    // Solve the problem
    Solution solution;
    int longestPathLength = solution.longestIncreasingPath(matrix);
    
    // Output the result
    cout << longestPathLength << endl;
    
    return 0;
}