import numpy as np
import json
import time
from sympy import Matrix
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


ERROR = 1e-200
CACHE = None

class Utils:
    @staticmethod
    def tan_inverse(x, y):
        angle = np.arctan(Utils.divide(y, x)) + (np.pi if x < 0 else 0)
        return angle

    @staticmethod
    def divide(a, b):
        # if b != 0: return a / b
        # if a == 0: return 0
        # return (1 if a > 0 else -1) / ERROR
        
        # return a / (max(b, ERROR) if b >= 0 else min(b, -ERROR))
        
        return a / (b if b != 0 else ERROR)
        
        # global CACHE
        # if b != 0:
        #     CACHE = a / b
        #     return a / b
        # val = (1 if CACHE >= 0 else -1) / ERROR
        # CACHE = val
        # return val
    
    @staticmethod
    def get_mask(size, fixed_nodes = []):
        [u, v] = fixed_nodes
        if u > v: u, v = v, u
        mask = np.ones(size, dtype=np.int32)
        mask[2*u: 2*(u+1)] = 0
        mask[2*v: 2*(v+1)] = 0
        return mask

    @staticmethod
    def null_vector(matrix, mask):
        n = matrix.shape[1]
        matrix = matrix[:, mask == 1]
        _, S, V = np.linalg.svd(matrix)
        
        vector = np.zeros(n)
        vector[mask == 1] = V[np.argmin(S)]
        return vector

class GraphRigidity:
    def __init__(self, node_positions, edges, fixed_edge, move_node, step_size):
        self.node_positions = np.array(node_positions, dtype=np.float32)
        self.edges = np.array(edges, dtype=np.int32)
        self.num_nodes = len(self.node_positions)
        self.edge_lengths = [{} for _ in range(self.num_nodes)]
        for [u, v] in self.edges:
            self.edge_lengths[u][v] = np.linalg.norm(self.node_positions[v] - self.node_positions[u])
            self.edge_lengths[v][u] = np.linalg.norm(self.node_positions[u] - self.node_positions[v])

        self.fixed_edge = fixed_edge
        self.move_node = move_node
        self.step_size = step_size

    def rigidity_matrix(self):
        rigidity = np.zeros((len(self.edges), 2*self.num_nodes), dtype=np.float32)
        for i, [u, v] in enumerate(self.edges):
            rigidity[i][2*u: 2*(u+1)] = self.node_positions[u] - self.node_positions[v]
            rigidity[i][2*v: 2*(v+1)] = self.node_positions[v] - self.node_positions[u]
        return rigidity

    def null_vector(self):
        rigidity = self.rigidity_matrix()
        mask = Utils.get_mask(2*self.num_nodes, self.edges[self.fixed_edge])
        rigidity = rigidity[:,mask == 1]

        null_space = np.array(Matrix(rigidity).nullspace(), dtype=np.float32)
        if len(null_space) == 0:
            raise Exception("Graph configuration is rigid!")
        if len(null_space) > 1:
            raise Exception("Graph has more than 1 non-trivial degree of freedom!")

        vector = np.zeros(2*self.num_nodes)
        vector[mask == 1] = null_space[0,:,0]
        return vector

    def velocity(self):
        node_velocities = Utils.null_vector(
            self.rigidity_matrix(),
            Utils.get_mask(2*self.num_nodes, self.edges[self.fixed_edge])
        ).reshape((self.num_nodes, 2))
        # node_velocities = self.null_vector().reshape((self.num_nodes, 2))

        v, u = self.move_node, self.edges[self.fixed_edge][0]
        if [v, u] not in self.edges and [u, v] not in self.edges:
            u = self.edges[self.fixed_edge][1]
        node_velocities *= Utils.divide(self.step_size, self.angular_velocity(node_velocities, u, v))
        return node_velocities

    def angular_velocity(self, node_velocities, u, v):
        relative_velocity = node_velocities[v] - node_velocities[u]
        relative_position = self.node_positions[v] - self.node_positions[u]
        return Utils.divide(relative_velocity[0], relative_position[1]) - Utils.divide(relative_velocity[1], relative_position[0])

    def bfs_edge(self):
        adj_list = [[] for _ in range(self.num_nodes)]
        for [u, v] in self.edges:
            adj_list[u].append(v)
            adj_list[v].append(u)

        visited = [False for _ in range(self.num_nodes)]
        queue = list(self.edges[self.fixed_edge])
        for u in queue:
            visited[u] = True

        while len(queue) != 0:
            u = queue[0]
            queue.pop(0)
            for v in adj_list[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
                    yield(v, [w for w in adj_list[v] if visited[w]])

    def move(self):
        # time.sleep(0.1)
        new_node_positions = np.zeros((self.num_nodes, 2))
        for u in self.edges[self.fixed_edge]:
            new_node_positions[u] = self.node_positions[u]

        node_velocities = self.velocity()
        # new_node_positions = self.node_positions + node_velocities

        for v, visited_nodes in self.bfs_edge():
            positions = []
            for w in visited_nodes:
                relative_position = self.node_positions[v] - self.node_positions[w]
                angle = Utils.tan_inverse(relative_position[0], relative_position[1]) - self.angular_velocity(node_velocities, w, v)
                positions.append([
                    new_node_positions[w][0] + self.edge_lengths[w][v] * np.cos(angle),
                    new_node_positions[w][1] + self.edge_lengths[w][v] * np.sin(angle)
                ])
            new_node_positions[v] = np.mean(np.array(positions, dtype=np.float32), axis=0)

        self.node_positions = new_node_positions
        return self.node_positions, self.edges


class Animation:
    def __init__(self, x_limit, y_limit, unit_length_points, edge_color, fps):
        self.x_limit = tuple(x_limit)
        self.y_limit = tuple(y_limit)
        self.fig, self.axis = plt.subplots()

        self.unit_length_points = unit_length_points
        self.edge_color = edge_color
        self.fps = fps

        self.fig.canvas.mpl_connect('button_press_event', self.toggle_pause)
        self.paused = False

    def plot(self, node_positions, edges):
        self.axis.clear()
        self.axis.set_xlim(self.x_limit)
        self.axis.set_ylim(self.y_limit)
        self.axis.set_aspect('equal')
        for [u, v] in edges:
            points = int(self.unit_length_points * np.linalg.norm(node_positions[v] - node_positions[u]))
            self.axis.plot(
                np.linspace(node_positions[u][0], node_positions[v][0], points),
                np.linspace(node_positions[u][1], node_positions[v][1], points),
                color=self.edge_color
            )

    def show(self, next):
        self.animate = FuncAnimation(self.fig, lambda x: self.plot(*next()), frames = 500, interval = 1000 / self.fps, blit = True)
        plt.show()

    def toggle_pause(self, *args, **kwargs):
        if self.paused:
            self.animate.resume()
        else:
            self.animate.pause()
        self.paused = not self.paused

if __name__ == '__main__':
    with open('config2.json') as f:
        config = json.load(f)

    graph_rigidity = GraphRigidity(**config['graph'])
    animation = Animation(**config['animation'])
    animation.show(graph_rigidity.move)

