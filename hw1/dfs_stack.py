import csv
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

def dfs(start, end):
    graph, distances = readFile()
    stack = [(start, [start], 0)]  # (當前節點, 當前路徑, 累積距離)
    visited = set()  # 記錄已訪問節點

    while stack:
        node, path, total_dist = stack.pop()  # 從堆疊取出當前節點
        
        if node == end:
            return path, total_dist, len(visited)  # 找到目標節點

        if node not in visited:
            visited.add(node)

            for neighbor in reversed(graph.get(node, [])):  # 反轉以模擬遞歸順序
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    new_dist = total_dist + distances.get((node, neighbor), 0)
                    stack.append((neighbor, new_path, new_dist))

    return None, float('inf'), visited  # 若無法抵達目標，回傳 None

if __name__ == '__main__':
    path, dist, num_visited = dfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')