# the distance of path should matchimport csv
import csv
import heapq
edgeFile = 'edges.csv'
heuristicFile = 'heuristic_values.csv'

def readFile(end):
    graph = {}
    distances = {}
    heuristic = {}
    with open(edgeFile, newline='') as csvfile:
        rows = csv.DictReader(csvfile)
        for row in rows:
            if int(row['start']) not in graph:
                graph[int(row['start'])]=[]
            graph[int(row['start'])].append(int(row['end']))
            distances[(int(row['start']), int(row['end']))] = float(row['distance'])
            
    with open(heuristicFile, newline='') as csvfile:
        rows = csv.DictReader(csvfile)
        for row in rows:
            heuristic[int(row['node'])] = float(row[str(end)])
    #print(heuristic)
    return graph, distances, heuristic

def astar(start, end):
    graph, distances, heuristic = readFile(end)
    pq = [(heuristic[start], 0, start, [start])]  # (f(n), g(n), 當前節點, 當前路徑)
    visited = {}

    while pq:
        f, cost, node, path = heapq.heappop(pq)  # 取出 f(n) 最小的節點

        if node in visited and visited[node] <= cost:
            continue  # 若已有更短路徑訪問此節點，則跳過

        visited[node] = cost  # 記錄當前節點的最短成本

        if node == end:
            return path, cost, visited  # 找到目標，回傳最佳路徑

        for neighbor in graph.get(node, []):
            new_cost = cost + distances.get((node, neighbor), float('inf'))  # g(n)
            f_new = new_cost + heuristic.get(neighbor, float('inf'))  # f(n) = g(n) + h(n)
            new_path = path + [neighbor]
            heapq.heappush(pq, (f_new, new_cost, neighbor, new_path))  # 放入優先隊列

    return None, float('inf'), visited  # 無法到達目標

if __name__ == '__main__':
    path, dist, num_visited = astar(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')