# number of nodes in the path should be the same
import csv
from collections import deque
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

def bfs(start, end):
    graph, distances = readFile()
    print("Graph:")
    print(graph)
    print("Distances:")
    print(distances)
    queue = deque([(start, [start], 0)])
    visited = set()
    
    while queue:
        node, path, total_dist = queue.popleft() # 取出 queue 中的第一個元素
        
        if node == end:
            return path, total_dist, len(visited)
        
        if node not in visited:
            visited.add(node)
            
            for neighbor in graph.get(node, []):  # 確保 node 有鄰接點
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    new_dist = total_dist + distances.get((node, neighbor), 0)
                    queue.append((neighbor, new_path, new_dist))
    
    return None, float('inf'), visited  # 如果找不到路徑

if __name__ == '__main__':
    path, dist, num_visited = bfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
