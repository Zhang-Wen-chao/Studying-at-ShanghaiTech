class Solution:
    def longestIncreasingPath(self, matrix):
        if not matrix: return 0
        n, m = len(matrix), len(matrix[0])
        dp = [[0]*m for _ in range(n)]
        res = 0
        for i in range(n):
            for j in range(m):
                res = max(res, self.dfs(matrix, dp, i, j, n, m))
        return res
    
    def dfs(self, matrix, dp, i, j, n, m):
        if dp[i][j]: return dp[i][j]
        dirs = [(0,1), (0,-1), (1,0), (-1,0)]
        max_len = 1
        for dx, dy in dirs:
            x, y = i + dx, j + dy
            if x<0 or x>=n or y<0 or y>=m or matrix[x][y] <= matrix[i][j]: continue
            len = 1 + self.dfs(matrix, dp, x, y, n, m)
            max_len = max(max_len, len)
        dp[i][j] = max_len
        return dp[i][j]
    
if __name__ == "__main__":
    n = int(input())
    matrix = []
    for _ in range(n):
        row = list(map(int, input().split()))
        matrix.append(row)
    
    solution = Solution()
    longestPathLength = solution.longestIncreasingPath(matrix)
    
    print(longestPathLength)