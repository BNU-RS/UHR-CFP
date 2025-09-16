
class Feature:
    def __init__(self, id) -> None:
        self.id = id
        self.linked = set()
        self.dir = None
        self.dist= None
        self.len= None
        self.width = None
        self.daul = False
        
    def set_daul(self):
        self.daul = True
        
    def is_daul(self):
        return self.daul

class Line(Feature):
    def __init__(self, id, direction=None) -> None:
        super(Line, self ).__init__(id)
        
        self.dir = direction
        self.dual = False
    
    def set_dir(self, direction):
        self.dir = direction
        
    def set_dist(self, dist):
        self.dist = dist
        
    def set_length(self, len):
        self.len = len
    
    def set_width(self, width):
        self.width = width
        
    def __len__(self):
        return self.len
        
class Node(Feature):
    def __init__(self, id) -> None:
        super(Node, self).__init__(id)
        self.linked_pair = dict()
        self.coord = None
        self.dual = False
        
        
    def set_link_pair(self, nid, link_type=0):
        # link_type: 0 -> direct; >1 -> defined by lineid
        if nid in self.linked_pair:
            if link_type != self.linked_pair[nid]:
                raise
        else:
            self.linked_pair[nid] = link_type
            
    def add_coord(self, coordinate:list):
        self.coord = coordinate
        
    def add_width(self, width):
        self.width = width
    

        
    
class Graph:
    def __init__(self) -> None:
        self.nodes = {}
        self.lines = {}
        
    def add_node(self, node_id):
        if node_id not in self.nodes:
            self.nodes[node_id] = Node(node_id)
    
    def add_line(self, line_id):
        if line_id not in self.lines:
            self.lines[line_id] = Line(line_id)
    
    def add_edge(self, node_id, line_id):
        self.add_node(node_id)
        self.add_line(line_id)
        
        self.nodes[node_id].linked.add(line_id)
        self.lines[line_id].linked.add(node_id)

    def add_linkpair(self, n1, n2, linktype=0):
        
        self.add_node(n1)
        self.add_node(n2)
        
        self.nodes[n1].set_link_pair(n2, linktype)
        self.nodes[n2].set_link_pair(n1, linktype)

    

# class LineSegment:
#     def __init__(self, angle):
#         self.angle = angle
#         self.connected_segments = []  # 与之相连的线段列表
#         self.visited = False

# graph = {
#     1: LineSegment(angle=30),
#     2: LineSegment(angle=45),
#     3: LineSegment(angle=35),
#     # ... 可以根据实际情况继续添加线段
# }

# # 设置线段之间的连接关系
# graph[1].connected_segments = [2, 3]
# graph[2].connected_segments = [1, 3]
# graph[3].connected_segments = [1, 2]


# def is_collinear(angle1, angle2, threshold_angle):
#     # 根据实际需求，确定共线的角度范围
#     return abs(angle1 - angle2) < threshold_angle

# def find_collinear_segments(node, threshold_angle, current_collinear_group):
#     node.visited = True
#     current_collinear_group.append(node)

#     for neighbor_id in node.connected_segments:
#         neighbor = graph[neighbor_id]
#         if not neighbor.visited and is_collinear(node.angle, neighbor.angle, threshold_angle):
#             find_collinear_segments(neighbor, threshold_angle, current_collinear_group)

# threshold_angle = 5  # 共线的角度阈值
# collinear_groups = []

# for node_id, node in graph.items():
#     if not node.visited:
#         current_collinear_group = []
#         find_collinear_segments(node, threshold_angle, current_collinear_group)
#         collinear_groups.append(current_collinear_group)

# # 输出结果
# for group in collinear_groups:
#     print([node_id for node_id in group])