import numpy as np 
from tqdm import tqdm
from scipy.optimize import minimize
import multiprocessing as mp

 
def get_rigid_transform(ins, cin, outs): 
    # convert to numpy arrays 
    p1, q1 = np.array(ins) 
    c1 = np.array(cin) 
    p2, q2 = np.array(outs) 
 
    v1, v2 = q1 - p1, q2 - p2 
    theta = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0]) 
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) 
    scale = np.linalg.norm(v2) / np.linalg.norm(v1) 
    c2 = np.dot(R, (c1 - p1)) * scale + p2 
 
    A2 = np.array([[p2[0], q2[0], c2[0]], 
            [p2[1], q2[1], c2[1]], 
            [1, 1, 1]]) 
    A1 = np.array([[p1[0], q1[0], c1[0]], 
            [p1[1], q1[1], c1[1]], 
            [1, 1, 1]]) 
    A = np.dot(A2, np.linalg.inv(A1)) 
    info = { 
        'rotation': 180*theta/np.pi, 
        'scale': scale, 
        'translation': c2 - c1 
    } 
    return A, info 

def SolveRansac(args):
    ins, cin, outs, weights, idx, threshold = args
    try: 
        A, _ = get_rigid_transform(ins[idx], cin, outs[idx]) 
    except:
        A = np.eye(3)
    pred = np.dot(A, np.concatenate((ins, np.ones((len(ins), 1))), axis=1).T).T[:, :2]
    dist = np.linalg.norm(pred - outs, axis=1)
    inliers = np.where(dist < threshold)[0]
    score = np.sum(weights[inliers])
    return score, inliers, A


 
def ransac(ins, cin, outs, n=100, threshold=0.1, lstsq=False, focus_moving_point=False): 
    ins, cin, outs = [np.array(x) for x in (ins, cin, outs)] 
    weights = np.clip(np.linalg.norm(outs - ins, axis=1), 0, 2) if focus_moving_point else np.ones(len(ins))

    results = list(map(SolveRansac, [(ins, cin, outs, weights, np.random.choice(len(ins), 2, replace=False), threshold) for _ in range(n)]))

    best_idx = np.argmax([x[0] for x in results])
    best_score, best_inliers, best_A = results[best_idx]

    ### recompute best_A using all inliers
    ### use least squares to solve for A 
    if lstsq:
        A = np.zeros((2*len(best_inliers), 6))
        b = np.zeros((2*len(best_inliers), 1))
        for j, idx in enumerate(best_inliers):
            i = ins[idx]
            o = outs[idx]
            A[2*j, :] = [i[0], i[1], 1, 0, 0, 0]
            A[2*j+1, :] = [0, 0, 0, i[0], i[1], 1]
            b[2*j] = o[0]
            b[2*j+1] = o[1]
        best_A = np.linalg.lstsq(A, b, rcond=None)[0].reshape((2, 3))

    return best_A, best_inliers

def get_info_from_transform(cin, A):
    c2 = np.dot(A, np.array([cin[0], cin[1], 1]))[:2]
    p1 = np.array([cin[0]+1, cin[1]])
    p2 = np.dot(A, np.array([p1[0], p1[1], 1]))[:2]
    scale = np.linalg.norm(p2 - c2)
    theta = np.arctan2(p2[1] - c2[1], p2[0] - c2[0])
    return {'rotation': theta*180/np.pi, 'scale': scale, 'translation': c2 - cin}

def get_transformation_matrix(x, y, z, roll, pitch, yaw):
    solution_matrix = np.eye(4)
    solution_matrix[:3, :3] = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ]) @ np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ]) @ np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    solution_matrix[:3, 3] = np.array([x, y, z])
    return solution_matrix

class Solver():
    def __init__(self, points1_ext, points_2_uv, cmat, x0=None):
        self.points1_ext = points1_ext
        self.points_2_uv = points_2_uv
        self.cmat = cmat
        if x0 is None:
            self.x0 = np.random.randn(6)
        else:
            self.x0 = x0

    def loss(self, solution):
        x, y, z, roll, pitch, yaw = solution
        solution_matrix = get_transformation_matrix(x, y, z, roll, pitch, yaw)
        transformed = solution_matrix @ self.points1_ext.T
        transformed_uv = self.cmat @ transformed
        transformed_uv = transformed_uv[:2] / transformed_uv[2:3]
        return np.mean(np.square(transformed_uv - self.points_2_uv)) + max(1, np.linalg.norm([x, y, z])) - 1
    
    def calc_solution(self):
        solution = minimize(self.loss, self.x0)
        return solution
    
def solve(solver):
    return solver.calc_solution()

def solve_3d_rigid_tfm(points_1, points_2_uv, cmat, max_iter=50, early_stop_threshold=1e-3):
    N = len(points_1)
    points1_ext = np.concatenate([points_1, np.ones([N, 1])], axis=1)
    points_2_uv = points_2_uv.T

    # first run once, if we got a good solution already, stop here
    # saves time for the initial few frames, where the object hasn't started moving yet
    solution = Solver(points1_ext, points_2_uv, cmat, x0=np.zeros(6)).calc_solution()
    if solution.fun < early_stop_threshold:
        best_solution = solution
    # otherwise, run {max_iter} times and pick the best one
    else:
        with mp.Pool(8) as pool:
            solutions = pool.map(solve, [Solver(points1_ext, points_2_uv, cmat) for _ in range(max_iter)])
        best_solution = solutions[np.argmin([s.fun for s in solutions])]

    return best_solution, get_transformation_matrix(*best_solution.x)


if __name__ == '__main__':
    ins = np.array([(1, 0), (1, 2), (0.5, 1), (0.5, 0.5), (1, 1)])
    cin = np.array((0, 1))
    outs =np.array([(2, 2), (4, 4), (2.5, 3.5), (2, 3), (3, 3)])
    best_A = ransac(ins, cin, outs, n=1000, threshold=0.1)
    print(best_A)
    print(np.dot(best_A, np.array([0.5, 0.5, 1])))
    print(get_info_from_transform(cin, best_A))
