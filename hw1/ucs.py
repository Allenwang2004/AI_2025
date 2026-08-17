# the distance of path should match
import csv
import heapq
edgeFile = 'edges.csv'

def readFile():
    graph = {}
    distances = {}
    with open(edgeFile, newline='') as csvfile:
        rows = csv.DictReader(csvfile)
        for row in rows:
            if int(row['start']) not in graph:
                graph[int(row['start'])]=[]
            graph[int(row['start'])].append(int(row['end']))
            distances[(int(row['start']), int(row['end']))] = float(row['distance'])
    return graph, distances

def ucs(start, end):
    graph, distances = readFile()
    pq = [(0, start, [start])]  # (累積距離, 當前節點, 當前路徑)
    visited = {}

    while pq:
        cost, node, path = heapq.heappop(pq)  # 取出當前成本最低的節點

        if node in visited and visited[node] <= cost:
            continue  # 如果這條路徑不是最短的，跳過

        visited[node] = cost  # 紀錄當前節點的最短路徑成本

        if node == end:
            return path, cost, len(visited)  # 找到最短路徑，回傳

        for neighbor in graph.get(node, []):
            new_cost = cost + distances.get((node, neighbor), float('inf'))
            new_path = path + [neighbor]
            heapq.heappush(pq, (new_cost, neighbor, new_path))  # 加入優先隊列

    return None, float('inf'), visited  # 無法到達目標節點


if __name__ == '__main__':
    path, dist, num_visited = ucs( 1718165260,  8513026827)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
