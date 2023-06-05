N, K = map(int, input().split())

if N < K*2:
    print("-1")
    exit(0)

x = 1
while (N - x + 1) - K*2 >= K*2:
    for i in range(K):
        print(x + K + i, end=" ")
    for i in range(K):
        print(x + i, end=" ")
    x += K*2

add = (N - x + 1) - K*3
if add >= 1:
    for i in range(K):
        print(x + K + i, end=" ")
    for i in range(add):
        print(x + i, end=" ")

    v = []
    for i in range(x + add, x + K):
        v.append(i)
    for i in range(x + K + K, N + 1):
        v.append(i)

    assert len(v) == K + K

    for i in range(K):
        print(v[i + K], end=" ")
    for i in range(K):
        print(v[i], end=" ")
else:
    for i in range(x, N - K + 1):
        print(i + K, end=" ")
    for i in range(K):
        print(x + i, end=" ")

print()
