#include <vector>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <cmath>
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
vector<vector<int>> parseInput(string input) {
    vector<int> numbers;
    stringstream ss(input);
    string num;

    // Remove all spaces and square brackets from the input
    input.erase(remove(input.begin(), input.end(), ' '), input.end());
    input.erase(remove(input.begin(), input.end(), '['), input.end());
    input.erase(remove(input.begin(), input.end(), ']'), input.end());

    ss = stringstream(input); // Reset stringstream with the cleaned input

    // Read each number in the input
    while (getline(ss, num, ',')) {
        try {
            numbers.push_back(stoi(num));
        } catch (const invalid_argument& e) {
            cout << "Invalid matrix: Element is not a valid integer." << endl;
            return {};
        }
    }

    // Check if the number of elements is a perfect square
    int numCount = numbers.size();
    int sqrtNumCount = sqrt(numCount);
    if (sqrtNumCount * sqrtNumCount != numCount) {
        cout << "Invalid matrix: Number of elements is not a perfect square." << endl;
        return {};
    }

    // Rearrange the numbers into an n x n matrix
    int n = sqrtNumCount;
    vector<vector<int>> matrix(n, vector<int>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix[i][j] = numbers[i * n + j];
        }
    }

    return matrix;
}

int main() {
    string input;
    getline(cin, input);
    vector<vector<int>> matrix = parseInput(input);
    
    // Solve the problem
    Solution solution;
    int longestPathLength = solution.longestIncreasingPath(matrix);

    // // Output the result
    cout << longestPathLength << endl;

    return 0;
}
