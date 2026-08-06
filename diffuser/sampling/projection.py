import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from scipy.optimize import minimize, Bounds

# uses SLSQP optimization


def _build_slsqp_constraints(C, d, A, b, obstacle_specs, transition_dim, horizon):
    """Build the SLSQP constraint tuple for one solve. Shared by the serial path
    (built once, reused across the batch) and each parallel worker (rebuilt locally
    since the lambdas can't cross a process boundary).

    obstacle_specs: list of (P_seq, q_seq, v_seq), one triple-of-sequences per
    obstacle, each sequence of length (horizon-1) — the (P,q,v) to use at each
    planned timestep t=1..horizon-1 (static obstacles just repeat the same one)."""
    constraints = ()
    for (P_seq, q_seq, v_seq) in obstacle_specs:
        for t in range(1, horizon):
            P, q, v = P_seq[t - 1], q_seq[t - 1], v_seq[t - 1]
            start_idx = t * transition_dim
            end_idx = (t + 1) * transition_dim
            constraints += ({'type': 'ineq',
                              'fun': lambda x, start_idx=start_idx, end_idx=end_idx, P=P, q=q, v=v: -x[start_idx: end_idx] @ P @ x[start_idx: end_idx] - q @ x[start_idx: end_idx] + v,
                              'jac': lambda x, start_idx=start_idx, end_idx=end_idx, P=P, q=q: np.concatenate([np.zeros(start_idx), -2 * P @ x[start_idx: end_idx] - q, np.zeros(len(x) - end_idx)])},)
    if C.size > 0:
        constraints += ({'type': 'ineq', 'fun': lambda x: -C @ x + d, 'jac': lambda x: -C},)
    if A.size > 0:
        constraints += ({'type': 'eq', 'fun': lambda x: A @ x - b, 'jac': lambda x: A},)
    return constraints


def _solve_slsqp_candidate(x0, r_i, Q, C, d, A, b, obstacle_specs, transition_dim, horizon):
    """Pickle-safe, module-level worker: solves one candidate's SLSQP problem.
    Used by the process pool; produces exactly the same optimization problem as
    the serial loop in Projector.project()."""
    constraints = _build_slsqp_constraints(C, d, A, b, obstacle_specs, transition_dim, horizon)
    cost_fun = lambda x: 0.5 * x @ Q @ x + r_i @ x
    jac_cost_fun = lambda x: Q @ x + r_i
    res = minimize(fun=cost_fun,
                    x0=x0,
                    constraints=constraints,
                    method='SLSQP',
                    jac=jac_cost_fun,
                    bounds=Bounds(-5 * np.ones_like(x0), 5 * np.ones_like(x0)),
                    tol=1e-6,
                    options={'maxiter': 1000, 'disp': False})
    return res.x


class Projector:
    # Projects sampled trajectories onto a feasible set (safety, dynamics, obstacles).
    # Used inside SDPC during denoising to enforce constraints.

    def __init__(self, horizon, transition_dim, action_dim=0, goal_dim=0, constraint_list=[], normalizer=None, variant='states', 
                 dt=0.1, cost_dims=None, skip_initial_state=True, diffusion_timestep_threshold=0.5, gradient=False, gradient_weights=None,
                 device='cuda', solver='proxsuite', parallelize=False):
        
        self.horizon = horizon # Store horizon length H (number of time steps in a planned snippet).
        self.transition_dim = transition_dim # Dimension of each step’s vector (usually state dim, or [state;action] dim).
        self.dt = torch.tensor(dt, device=device) # Sampling time used by dynamic constraints (as a torch scalar on correct device).
        self.skip_initial_state = skip_initial_state # If True, we don’t optimize the very first state (we pin s_0 to input trajectory).
        # self.only_last = only_last
        self.diffusion_timestep_threshold = diffusion_timestep_threshold # Threshold for when to activate projection inside the diffusion time steps (used in Denoising process in the diffusion.py)
        self.gradient = gradient # If True, compute gradients of constraint violations instead of solving the QP/NLP. (used in eval.py and diffusion.py)
        self.gradient_weights = gradient_weights
        self.device = device # Torch device (e.g., 'cuda' or 'cpu').
        self.solver = solver
        self.parallelize = parallelize
        self._pool = None  # lazily created ProcessPoolExecutor, see _get_pool()
        self.last_nit = []  # per-candidate SLSQP iteration counts from the most
                             # recent project() call with collect_nit=True (serial
                             # path only -- see project()'s docstring)

        # Determine whether to include actions in the projection
        if normalizer is None:
            # No normalization
            self.normalizer = None
        elif variant == 'states':
            # Only states are projected/normalized.
            self.normalizer = ProjectionNormalizer(observation_normalizer=normalizer.normalizers['observations'], goal_dim=goal_dim)
        elif variant == 'states_actions':
        # elif transition_dim != normalizer.normalizers['observations'].maxs.size:
            # States and actions are projected/normalized together.
            self.normalizer = ProjectionNormalizer(observation_normalizer=normalizer.normalizers['observations'], 
                                                   action_normalizer=normalizer.normalizers['actions'], goal_dim=goal_dim)
        else:
            KeyError('Invalid variant. Choose either "states" or "states_actions".')            

        # -------------------------
        # Quadratic cost matrix Q for min  1/2 z^T Q z + r^T z
        # -------------------------
        if cost_dims is not None:
            costs = torch.ones(transition_dim, device=self.device) # is the vector used to build cost matrix
            for idx in cost_dims: #idx means index here in cost_dims has some index and it points to which dimensions of your state vector get a custom weight
                costs[idx] = 1
            self.Q = torch.diag(torch.tile(costs, (self.horizon, )))
        else:
            self.Q = torch.eye(transition_dim * horizon, device=self.device) # This creates a cost matrix Q of size (transition_dim * horizon) × (transition_dim * horizon).

        # Initialize linear constraint matrices (will be concatenated later).
        self.A = torch.empty((0, self.transition_dim * self.horizon), device=self.device)   # Equality constraints: A z = b, it’s a tensor with zero rows and transition_dim * horizon columns
        self.b = torch.empty(0, device=self.device) #1D tensor of length 0
        self.C = torch.empty((0, self.transition_dim * self.horizon), device=self.device)   # Inequality constraints: C z ≤ d
        self.d = torch.empty(0, device=self.device)

        # Instantiate constraint builders. They hold specs and build A,b,C,d and/or nonlinear pieces.
        self.safety_constraints = SafetyConstraints(horizon=horizon, transition_dim=transition_dim, normalizer=self.normalizer, 
                                                 skip_initial_state=self.skip_initial_state, action_dim=action_dim, device=self.device)
        self.dynamic_constraints = DynamicConstraints(horizon=horizon, transition_dim=transition_dim, normalizer=self.normalizer,
                                                      skip_initial_state=self.skip_initial_state, dt=self.dt, device=self.device)
        self.obstacle_constraints = ObstacleConstraints(horizon=horizon, transition_dim=transition_dim, normalizer=self.normalizer,
                                                        skip_initial_state=self.skip_initial_state, dt=self.dt, device=self.device)

        # Distribute incoming constraint specs to the right builder.
        for constraint_spec in constraint_list:
            if constraint_spec[0] == 'deriv':
                # Dynamics coupling (e.g., x_{t+1} = x_t + dt * v_t).
                self.dynamic_constraints.constraint_list.append(constraint_spec)
            elif constraint_spec[0] == 'lb' or constraint_spec[0] == 'ub' or constraint_spec[0] == 'eq' or constraint_spec[0] == 'ineq':
                # Linear bounds / equalities / inequalities on stacked variables.
                self.safety_constraints.constraint_list.append(constraint_spec)
            elif constraint_spec[0] == 'sphere_inside' or constraint_spec[0] == 'sphere_outside':
                # Quadratic obstacle constraints of the form s^T P s + q^T s ≤ v (or ≥ v).
                self.obstacle_constraints.constraint_list.append(constraint_spec)

        self.safety_constraints.build_matrices()
        self.dynamic_constraints.build_matrices()
        self.obstacle_constraints.build_matrices()
        self.append_linear_constraint(self.safety_constraints)
        self.append_linear_constraint(self.dynamic_constraints)
        self.add_numpy_constraints()     

    def project(self, trajectory, constraints=None, collect_nit=False):
        """
            model-based projection into the feasible set
            trajectory: np.ndarray of shape (batch_size, horizon, transition_dim)

            collect_nit: if True, populates self.last_nit with each candidate's
            SLSQP iteration count (res.nit) from this call, for offline solver-cost
            analysis. Only populated on the serial path -- if self.parallelize and
            batch_size > 1 both hold, the parallel worker path runs instead and
            self.last_nit is left empty; pass parallelize=False at construction
            time (or batch_size=1) to guarantee the serial path when this matters.
            Solve an optimization problem of the form 
                \hat z =   argmin_z 1/2 z^T Q z + r^T z
                        subject to  Az  = b
                                    Cz <= d
            where z = (o_0, o_1, ..., o_{H-1}) is the trajectory in vector form over horizon H. 
            The matrices A, b, C, and d are defined by the dynamic and safety constraints.

            Inputs:
                trajectory: torch.Tensor with shape (B, H, T)  (B=batch, H=horizon, T=transition_dim)
            
            Returns:
                projected_trajectory (torch.Tensor with shape like input),
                projection_costs (np.ndarray of length B), a diagnostic scalar per batch item.
                                    
        """
        
        dims = trajectory.shape

        # Reshape the trajectory to a batch of vectors (from B x H x T to B x (HT) If you had (B=8, H=5, T=4), this becomes shape (8, 20)
        batch_size = trajectory.shape[0]
        trajectory_reshaped = trajectory.reshape(trajectory.shape[0], -1)

        # Linear term r encodes “pull toward original trajectory” via r = -z0^T Q
        # which penalizes how far the optimized trajectory z deviates from the original trajectory_reshaped.
        r = - trajectory_reshaped @ self.Q
        # Convert everything to NumPy arrays (and double precision) because the solver (scipy.optimize.minimize) expects NumPy input.
        r_np = r.cpu().numpy()
        Q = self.Q_np.astype('double')
        trajectory_np = trajectory_reshaped.cpu().numpy()

        # Constraints define linear equalities (A z = b) and inequalities (C z ≤ d).
        A = self.A_np.astype('double')
        b = self.b_np.astype('double')
        C = self.C_np.astype('double')
        d = self.d_np.astype('double')

        if self.skip_initial_state:
            s_0 = trajectory_reshaped[0, :self.transition_dim]
            if self.solver == 'proxsuite' or self.solver == 'gurobi':
                s_0 = s_0.cpu().numpy()
            counter = 0
            for constraint in self.dynamic_constraints.constraint_list:
                if constraint[0] == 'deriv':
                    x_idx = int(constraint[1][0])
                    b[counter * self.horizon] = s_0[x_idx]
                    counter += 1
        
        # ensures they’re 64-bit floats.
        r_np_double = r_np.astype('double')
        trajectory_np_double = trajectory_np.astype('double')

        # Creates inequality constraints for obstacles. Each obstacle has parameters (P, q, v) describing a quadratic region: st⊤*​P*st​+q⊤*st​−v≤0
        # For each time step and obstacle, it creates:
        # fun: computes the value of the inequality (must be ≥0 for SLSQP)
        # jac: its analytical gradient (Jacobian) That’s how SLSQP knows how to enforce obstacle avoidance.
        obstacle_specs = list(zip(self.obstacle_constraints.P_list,
                                   self.obstacle_constraints.q_list,
                                   self.obstacle_constraints.v_list))

        #Prepare arrays to store the solution and cost for each trajectory in the batch.
        projection_costs = np.ones(batch_size, dtype=np.float32)
        sol_np = np.zeros((batch_size, self.horizon * self.transition_dim), dtype=np.float32)

        # Each candidate i solves an independent SLSQP problem (same A,b,C,d,obstacles,
        # different x0/r_np_double[i]) — embarrassingly parallel across the batch. Run
        # them concurrently in worker processes when there's more than one candidate;
        # fall back to the proven serial path otherwise or if parallel execution fails.
        ran_parallel = False
        if self.parallelize and batch_size > 1:
            try:
                pool = self._get_pool(min(batch_size, os.cpu_count() or 1))
                futures = [
                    pool.submit(_solve_slsqp_candidate, trajectory_np_double[i], r_np_double[i],
                                Q, C, d, A, b, obstacle_specs, self.transition_dim, self.horizon)
                    for i in range(batch_size)
                ]
                for i, fut in enumerate(futures):
                    sol_np[i] = fut.result()
                ran_parallel = True
            except Exception as e:
                print(f"[Projection] parallel SLSQP solve failed ({e}); falling back to serial.")

        if not ran_parallel:
            if collect_nit:
                self.last_nit = []
            # Built once and reused across the batch — identical constraints for every i.
            constraints = _build_slsqp_constraints(C, d, A, b, obstacle_specs, self.transition_dim, self.horizon)
            for i in range(batch_size):
                cost_fun = lambda x: 0.5 * x @ Q @ x + r_np_double[i] @ x # + (A_double @ x - b_double) @ (A_double @ x - b_double)
                jac_cost_fun = lambda x: Q @ x + r_np_double[i]
                #Run the optimizer
                res = minimize(fun=cost_fun,
                                x0=trajectory_np_double[i],
                                constraints=constraints,
                                method='SLSQP',
                                jac=jac_cost_fun,
                                bounds=Bounds(-5 * np.ones_like(trajectory_np_double[i]), 5 * np.ones_like(trajectory_np_double[i])),
                                tol=1e-6,
                                options={'maxiter': 1000, 'disp': False})
                sol_np[i] = res.x # Save the optimized trajectory vector.
                if collect_nit:
                    self.last_nit.append(int(res.nit))

        #Compute a “projection cost” (a measure of how much the trajectory changed) for every candidate.
        for i in range(batch_size):
            projection_costs[i] = 0.5 * sol_np[i] @ Q @ sol_np[i] + r_np[i] @ sol_np[i] + 0.5 * trajectory_np[i] @ Q @ trajectory_np[i]

        sol = torch.tensor(sol_np, device=self.device).reshape(dims) #Reshape the solution back to dims

        # print(f'Projection time {self.solver}:', time.time() - start_time)
        return sol, projection_costs    # only implemented for proxsuite and scipy

    def _get_pool(self, n_workers):
        # 'spawn' (not 'fork'): the parent process holds an active CUDA context
        # (torch on self.device), and forking a CUDA-initialized process is unsafe.
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context('spawn'))
        return self._pool

    def __del__(self):
        if getattr(self, '_pool', None) is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
    
    def compute_gradient(self, trajectory, constraints=None):
        """
            Define a violation cost over the flattened trajectory τ (vectorized as 𝑧)
            trajectory: np.ndarray of shape (batch_size, horizon, transition_dim) or (horizon, transition_dim)
            Calculate the (weighted) gradients for the following cost functions:
            - equalities:   c_1 = ||A * tau - b||^2                             --> grad_1 = 2 * A^T (A * tau - b)
            - inequalities: c_2 = max(0, C * tau - d)^2                         --> grad_2 = 2 * C^T max(0, C * tau - d)
            - obstacles:    c_3 = sum_{t=1}^{H-1} (s_t^T P s_t + q^T s_t - v)^2 --> grad_3 = 2 * (2 * P s_t + q) * (s_t^T P s_t + q^T s_t - v)  
                            At each timestep t, for each obstacle, the constraint is formulated as: vt=(stT*P*st)+qT*st-v,
                            if that vt is negative, vt<=0 the system is safe
                            if that vt is positive,vt>=0 (inside obstacle), 
                            In that case, we compute the gradient (direction of steepest increase of vt) is (2Pst+q)
                            the descent direction that pushes the state out of the obstacle region is the negative of that: -(2Pst+q),
            
            Shapes: trajectory is (B,H,T). We flatten to (B, H*T), compute grads, then reshape back.
            Return an analytic descent direction that reduces violations              
        """
        
        trajectory_reshaped = trajectory.reshape(trajectory.shape[0], -1)
        trajectory_np = trajectory_reshaped.cpu().numpy()

        # Constraints
        A, b, C, d = self.A, self.b, self.C, self.d

        if self.skip_initial_state:
            s_0 = trajectory_reshaped[0, :self.transition_dim]
            counter = 0
            for constraint in self.dynamic_constraints.constraint_list:
                if constraint[0] == 'deriv':
                    x_idx = int(constraint[1][0])
                    b[counter * self.horizon] = s_0[x_idx]
                    counter += 1

        # Equality and polytopic constraints
        grad1 = torch.zeros_like(trajectory_reshaped)
        grad2 = torch.zeros_like(trajectory_reshaped)
        for i in range(trajectory.shape[0]):
            grad1[i] = - A.T @ (A @ trajectory_reshaped[i] - b)
            grad2[i] = - C.T @ torch.max(torch.zeros_like(C @ trajectory_reshaped[i] - d), C @ trajectory_reshaped[i] - d)
        grad1 = grad1.reshape(trajectory.shape)
        grad2 = grad2.reshape(trajectory.shape)

        # Obstacle constraints — vectorized across batch and timestep. Each obstacle now
        # carries one (P,q,v) per planned timestep (P_seq/q_seq/v_seq, length horizon-1)
        # instead of a single matrix shared across the whole horizon — static obstacles
        # just repeat the same triple at every t, dynamic ones vary per t (predicted
        # future position), so this still handles both with one code path.
        batch_size = trajectory_np.shape[0]
        S_all = trajectory_np.reshape(batch_size, self.horizon, self.transition_dim)
        S = S_all[:, 1:, :]  # (B, horizon-1, transition_dim), t = 1..horizon-1
        grad3 = np.zeros_like(S_all)
        for constraint_idx in range(len(self.obstacle_constraints.P_list)):
            P_seq = np.stack(self.obstacle_constraints.P_list[constraint_idx], axis=0)  # (T, dim, dim)
            q_seq = np.stack(self.obstacle_constraints.q_list[constraint_idx], axis=0)  # (T, dim)
            v_seq = np.asarray(self.obstacle_constraints.v_list[constraint_idx])        # (T,)
            s_dot_q = np.einsum('bti,ti->bt', S, q_seq)                                 # (B, T)
            s_P_s = np.einsum('bti,tij,btj->bt', S, P_seq, S)                           # (B, T), s^T P s
            violated = (s_P_s + s_dot_q) > v_seq[None, :]                               # (B, T)
            grad_dir = 2.0 * np.einsum('bti,tij->btj', S, P_seq) + q_seq[None, :, :]    # (B, T, dim); P symmetric so P==P.T
            grad3[:, 1:, :] -= grad_dir * violated[..., None]
        grad3 = torch.tensor(grad3.reshape(trajectory_np.shape), device=self.device).reshape(trajectory.shape)
        
        if self.gradient_weights is not None:
            grad1 = grad1 * self.gradient_weights[0]
            grad2 = grad2 * self.gradient_weights[1]
            grad3 = grad3 * self.gradient_weights[2]
        
        return grad1 + grad2 + grad3 

    def project_inloop(self, x_recon_norm):
        """
        In-loop SLSQP projection during denoising.
        Converts normalized delta-actions → world positions, projects against obstacles,
        then converts back. Requires self.pos0 (numpy (3,)) and self.action_normalizer
        to be set before each denoising call.

        x_recon_norm: (K, H, action_dim) normalized delta-actions (torch tensor)
        Returns:      (K, H, action_dim) normalized delta-actions (torch tensor)
        """
        K, H, D = x_recon_norm.shape
        x_np = x_recon_norm.detach().cpu().numpy()          # (K, H, D)
        x_real = self.action_normalizer.unnormalize(x_np)   # (K, H, D)

        # integrate deltas → absolute world positions  (K, H+1, D)
        pos_traj = np.zeros((K, H + 1, D), dtype=np.float32)
        pos_traj[:, 0] = self.pos0[:D].astype(np.float32)
        pos_traj[:, 1:] = pos_traj[:, :1] + np.cumsum(x_real, axis=1)

        if self.transition_dim == 2 * D:
            # Velocity dims appended (see build_obstacle_constraint_list's
            # rate_limit path): v[t] is the velocity driving x[t] -> x[t+1], tied
            # together by the 'deriv' dynamics constraint x[t+1] = x[t] + dt*v[t]
            # already built into DynamicConstraints. Derive it from the raw deltas
            # so the state we hand to the solver starts feasible w.r.t. that equality.
            dt_val = float(self.dt.item())
            vel_traj = np.zeros((K, H + 1, D), dtype=np.float32)
            vel_traj[:, :H] = x_real / dt_val
            vel_traj[:, H] = vel_traj[:, H - 1]
            state_traj = np.concatenate([pos_traj, vel_traj], axis=-1)  # (K, H+1, 2D)
        else:
            state_traj = pos_traj

        state_t = torch.tensor(state_traj, dtype=torch.float32, device=self.device)
        state_proj_t, _ = self.project(state_t)              # (K, H+1, transition_dim)
        state_proj = state_proj_t.detach().cpu().numpy()
        pos_proj = state_proj[..., :D]

        # positions → real deltas → renormalize
        x_real_proj = (pos_proj[:, 1:] - pos_proj[:, :-1]).astype(np.float32)
        x_norm_proj = self.action_normalizer.normalize(x_real_proj)
        np.clip(x_norm_proj, -1.0, 1.0, out=x_norm_proj)

        return torch.tensor(x_norm_proj, dtype=torch.float32, device=x_recon_norm.device)

    def append_linear_constraint(self, constraint):
        # build constraints in separate modules, then concatenate their matrices, solver sees all constraints at once.
        # dim=0 = append rows. Column count matches transition_dim*horizon.
        # Stack inequalities: C z ≤ d
        self.C = torch.cat([self.C, constraint.C], dim=0)
        self.d = torch.cat([self.d, constraint.d], dim=0)
        # Stack equalities: A z = b
        self.A = torch.cat([self.A, constraint.A], dim=0)
        self.b = torch.cat([self.b, constraint.b], dim=0)

        if constraint.__class__.__name__ == 'SafetyConstraints':
            self.A_safe, self.b_safe, self.C_safe, self.d_safe = constraint.A, constraint.b, constraint.C, constraint.d
        elif constraint.__class__.__name__ == 'DynamicConstraints':
            self.A_dyn, self.b_dyn, self.C_dyn, self.d_dyn = constraint.A, constraint.b, constraint.C, constraint.d

    def add_numpy_constraints(self):
        #SciPy and many third-party solvers expect NumPy arrays on CPU. 
        #This method snapshots torch tensors into NumPy, 
        #so the solver path can reuse them without repeated conversion overhead.
        self.A_np = self.A.cpu().numpy()
        self.b_np = self.b.cpu().numpy()     
        self.C_np = self.C.cpu().numpy()
        self.d_np = self.d.cpu().numpy()
        self.Q_np = self.Q.cpu().numpy()


class Constraints:

    #Stores the planning horizon H, 
    #he per-timestep vector length (transition_dim), 
    #an optional normalizer (so constraints can be expressed/converted consistently),and the device. 
    #Initializes empty constraint matrices/vectors on the chosen device.
    def __init__(self, horizon, transition_dim, normalizer=None, device='cuda'):
        self.horizon = horizon
        self.transition_dim = transition_dim
        self.normalizer = normalizer
        self.device = device

        self.A = torch.empty((0, self.transition_dim * self.horizon), device=device)
        self.b = torch.empty(0, device=device)
        self.C = torch.empty((0, self.transition_dim * self.horizon), device=device)
        self.d = torch.empty(0, device=device)

    def build_matrices(self):
        pass

class SafetyConstraints(Constraints):

    def __init__(self, skip_initial_state=True, action_dim=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_initial_state = skip_initial_state
        self.action_dim = action_dim
        self.constraint_list = []
        
    def build_matrices(self, constraint_list=None):
        """
            Input:
                constraint_list: list of constraints
                    e.g.
                        [('lb', [-1.0, -inf, 0]), ('ub', [1.0, 2.0, inf])] -->  x_0 in [-1, 1], x_1 in [-inf, 2], x_2 in [0, inf],
                        0 * x_0 + 1 * x_1 + 1 * x_2 = 1.5   ('eq', ([0, 1, 1], 1.5))
                        1 * x_0 + 0 * x_1 + 0 * x_2 <= 0.5  ('ineq': ([1, 0, 0], 0.5))
                        where x_i is the i-th dimension of the state or state-action vectors
            The matrices have the following shapes:
                C: (horizon * n_bounds, transition_dim * horizon)
                d: (horizon * n_bounds)
                lb: horizon * transition_dim
            C consists of n_bounds blocks of shape (horizon, transition_dim * horizon), where each block corresponds to a 
            constraint (ub or lb) on a specific dimension. The block has a 1 or -1 at the corresponding dimension and time step.
        """

        if constraint_list is None:
            constraint_list = self.constraint_list
        else:
            self.constraint_list.extend(constraint_list)

        for constraint in constraint_list:
            type = constraint[0]
            bound = constraint[1]
            if type == 'lb' or type == 'ub':
                for dim in range(len(bound)):
                    if bound[dim] == -np.inf or bound[dim] == np.inf:
                        continue
                    
                    mat_append = torch.zeros(self.horizon, self.transition_dim * self.horizon, device=self.device)
                    vec_append = torch.zeros(self.horizon, device=self.device)

                    sign = 1 if type == 'ub' else -1
                    for t in range(self.horizon):
                        mat_append[t, t * self.transition_dim + dim] = sign
                        vec_append[t] = sign * bound[dim]
                        
                    if self.normalizer is not None:
                        x_min = self.normalizer.mins[dim]
                        x_max = self.normalizer.maxs[dim]
                        mat_append = mat_append * (x_max - x_min) / 2
                        vec_append = vec_append - sign * (x_min + x_max) / 2

                    if self.skip_initial_state and dim >= self.action_dim:
                        mat_append = mat_append[1:]
                        vec_append = vec_append[1:]

                    self.C = torch.cat((self.C, mat_append), dim=0)
                    self.d = torch.cat((self.d, vec_append), dim=0)
                continue         

            # type == 'eq' or 'ineq'
            mat_append = torch.zeros(self.horizon, self.transition_dim * self.horizon, device=self.device)
            vec_append = torch.zeros(self.horizon, device=self.device)

            for i in range(self.horizon):
                if self.normalizer is not None:
                    x_min = self.normalizer.mins
                    x_max = self.normalizer.maxs

                    # Unnormalize the constraints. TODO: Extend to actions by checking if dim <= action_dim
                    # We have Cs <= d, where s is the unnormalized state vector. This is converted to C's_n <= d', where s_n is the normalized state vector,
                    # by using the fact that s = (s_n + 1) * (s_max - s_min) / 2 + s_min.
                    a = bound[0] * (x_max - x_min) / 2
                    b = bound[1] - bound[0] @ (x_max + x_min) / 2
                else:
                    a = bound[0]
                    b = bound[1]
                
                mat_append[i, i * self.transition_dim: (i + 1) * self.transition_dim] = torch.tensor(a, device=self.device)
                vec_append[i] = torch.tensor(b, device=self.device)

            if self.skip_initial_state:
                mat_append = mat_append[1:]
                vec_append = vec_append[1:]

            if type == 'eq':
                self.A = torch.cat((self.A, mat_append), dim=0)
                self.b = torch.cat((self.b, vec_append), dim=0)
            else:
                self.C = torch.cat((self.C, mat_append), dim=0)
                self.d = torch.cat((self.d, vec_append), dim=0)         

class DynamicConstraints(Constraints):
    def __init__(self, skip_initial_state=True, dt=0.02, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_initial_state = skip_initial_state
        self.dt = dt
        self.constraint_list = []
        
    def build_matrices(self, constraint_list=None):
        """
            Input:
                constraint_list: list of constraints
                    e.g. [('deriv', [0, 2]), ('deriv', [1, 3])] -->
                    x_0[t+1] = x_0[t] + self.dt * x_2[t], x_1[t+1] = x_1[t] + self.dt * x_3[t]                                      (explicit Euler) or
                    x_0[t+1] = x_0[t] + self.dt * (x_2[t] + x_2[t+1]) / 2, x_1[t+1] = x_1[t] + self.dt * (x_3[t] + x_3[t+1]) / 2    (variant of trapezoidal rule)
                    where x_i[t] is the i-th dimension of the state or state-action vectors at time t
            The matrices have the following shapes:
                C: (horizon * n_bounds, transition_dim * horizon)
                d: (horizon * n_bounds)
        """

        if constraint_list is None:
            constraint_list = self.constraint_list
        else:
            self.constraint_list.extend(constraint_list)
        
        for constraint in constraint_list:
            type = constraint[0]
            vals = constraint[1]
            if 'deriv' in type:
                x_idx = int(vals[0])
                dx_idx = int(vals[1])
            
                mat_append = torch.zeros(self.horizon - 1, self.transition_dim * self.horizon, device=self.device)
                vec_append = torch.zeros(self.horizon - 1, device=self.device)

                # Calculate multiplicative factors needed for normalization
                if self.normalizer is not None: 
                    x_min = self.normalizer.mins[x_idx]
                    x_max = self.normalizer.maxs[x_idx]
                    dx_min = self.normalizer.mins[dx_idx]
                    dx_max = self.normalizer.maxs[dx_idx]
                    x_diff = x_max - x_min
                    dx_diff = dx_max - dx_min
                    dx_sum = dx_max + dx_min

                for i in range(self.horizon - 1):
                    if self.normalizer is not None:
                        mat_append[i, i * self.transition_dim + x_idx] = 1 * x_diff
                        mat_append[i, i * self.transition_dim + dx_idx] = self.dt * dx_diff
                        mat_append[i, (i + 1) * self.transition_dim + x_idx] = -1 * x_diff
                        vec_append[i] = - dx_sum * self.dt
                    else:
                        mat_append[i, i * self.transition_dim + x_idx] = 1
                        mat_append[i, i * self.transition_dim + dx_idx] = self.dt
                        mat_append[i, (i + 1) * self.transition_dim + x_idx] = -1
                        vec_append[i] = 0

                if self.skip_initial_state:     # --> Do that in the projection method because it needs the current state. For that, record the relevant rows
                    mat_fix_initial = torch.zeros(1, self.transition_dim * self.horizon, device=self.device)    # Fix the initial state
                    mat_fix_initial[0, x_idx] = 1
                    mat_append = torch.cat((mat_fix_initial, mat_append), dim=0)
                    vec_append = torch.cat((torch.tensor([0], device=self.device), vec_append), dim=0)          # Must be changed to current state in each iteration!

                self.A = torch.cat((self.A, mat_append), dim=0)
                self.b = torch.cat((self.b, vec_append), dim=0)

class ObstacleConstraints(Constraints):

    def __init__(self, skip_initial_state=True, dt=0.02, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_initial_state = skip_initial_state
        self.dt = dt
        self.constraint_list = []

    def build_matrices(self, constraint_list=None):
        """
            Input:
                constraint_list: list of constraints
                    e.g. [('sphere_inside', [0, 2] [-1, 5], 1), ('sphere_outside', [1, 3], [0, 1], 4)] -->
                    (x_0 + 1)^2 + (x_2 - 5)^2 <= 1, x_1^2 + (x_3 - 1)^2 >= 4
                    where x_i is the i-th dimension of the state or state-action vectors at time t

                    A '_dynamic' suffix on the type (e.g. 'sphere_outside_dynamic') means
                    `center` is instead a list of (horizon-1) centers, one per planned
                    timestep t=1..horizon-1 — used for obstacles predicted to move during
                    the horizon. Everything else about the constraint is unchanged.
            Generate the matrix P and the vector q for:
                s_i^T P s_i + q^T s_i <= v
            The matrices have the following shapes:
                C: (horizon * n_bounds, transition_dim * horizon)
                d: (horizon * n_bounds)

            self.P_list / q_list / v_list: one entry per obstacle, each itself a list
            of length (horizon-1) — the (P,q,v) triple to use at each planned timestep.
            Static obstacles just repeat the same triple at every timestep.
        """

        if constraint_list is None:
            constraint_list = self.constraint_list
        else:
            self.constraint_list.extend(constraint_list)

        n_steps = self.horizon - 1

        self.P_list = []
        self.q_list = []
        self.v_list = []
        for constraint in constraint_list:
            type = constraint[0]
            dims = constraint[1]
            radius = constraint[3]

            is_dynamic = type.endswith("_dynamic")
            base_type = type[: -len("_dynamic")] if is_dynamic else type
            centers_per_t = constraint[2] if is_dynamic else [constraint[2]] * n_steps
            if len(centers_per_t) != n_steps:
                raise ValueError(
                    f"'{type}' constraint has {len(centers_per_t)} centers, expected "
                    f"horizon-1={n_steps} (one per planned timestep)"
                )

            P_seq, q_seq, v_seq = [], [], []
            for center in centers_per_t:
                P = np.zeros((self.transition_dim, self.transition_dim))
                q = np.zeros(self.transition_dim)
                v = radius ** 2

                dim_counter = 0
                for dim in dims:
                    if self.normalizer is not None:
                        delta_s = self.normalizer.maxs[dim] - self.normalizer.mins[dim]
                        s_min = self.normalizer.mins[dim]
                        P[dim, dim] = delta_s ** 2 / 4
                        q[dim] = delta_s**2 / 2 + delta_s * (s_min - center[dim_counter])
                        v -= delta_s**2 / 4 + delta_s * (s_min - center[dim_counter]) + (s_min - center[dim_counter]) ** 2
                    else:
                        P[dim, dim] = 1
                        q[dim] = -2 * center[dim_counter]
                        v -= center[dim_counter] ** 2
                    dim_counter += 1

                if base_type == 'sphere_outside':
                    P = -P
                    q = -q
                    v = -v

                P_seq.append(P)
                q_seq.append(q)
                v_seq.append(v)

            self.P_list.append(P_seq)
            self.q_list.append(q_seq)
            self.v_list.append(v_seq)

class ProjectionNormalizer():
    def __init__(self, observation_normalizer=None, action_normalizer=None, goal_dim=0):
        self.observation_normalizer = observation_normalizer
        self.action_normalizer = action_normalizer
        self.goal_dim = goal_dim
        self.get_limits()

    def get_limits(self):
        if self.observation_normalizer is not None and self.action_normalizer is not None:
            x_max_obs = self.observation_normalizer.maxs[:-self.goal_dim] if self.goal_dim > 0 else self.observation_normalizer.maxs
            x_min_obs = self.observation_normalizer.mins[:-self.goal_dim] if self.goal_dim > 0 else self.observation_normalizer.mins
            x_max = np.concatenate([self.action_normalizer.maxs, x_max_obs])
            x_min = np.concatenate([self.action_normalizer.mins, x_min_obs])
        elif self.observation_normalizer is not None:
            x_max = self.observation_normalizer.maxs[:-self.goal_dim] if self.goal_dim > 0 else self.observation_normalizer.maxs
            x_min = self.observation_normalizer.mins[:-self.goal_dim] if self.goal_dim > 0 else self.observation_normalizer.mins
        elif self.action_normalizer is not None:
            x_max = self.action_normalizer.maxs
            x_min = self.action_normalizer.mins
        
        self.maxs = x_max
        self.mins = x_min

    def normalize(self, x):
        x_normalized = (x - self.mins) / (self.maxs - self.mins) * 2 - 1
        return x_normalized
    
    def unnormalize(self, x_normalized):
        x = (x_normalized + 1) * (self.maxs - self.mins) / 2 + self.mins
        return x
                

class CBF(Constraints):
    def __init__(self, skip_initial_state=True, dt=0.02, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_initial_state = skip_initial_state
        self.dt = dt
        self.constraint_list = []

    def build_matrices(self, constraint_list=None):
        """
            Input:
                constraint_list: list of constraints
                    e.g. [('sphere_inside', [0, 2] [-1, 5], 1), ('sphere_outside', [1, 3], [0, 1], 4)] -->
                    (x_0 + 1)^2 + (x_2 - 5)^2 <= 1, x_1^2 + (x_3 - 1)^2 >= 4
                    where x_i is the i-th dimension of the state or state-action vectors at time t
            Generate the matrix P and the vector q for:
                s_i^T P s_i + q^T s_i <= v
            The matrices have the following shapes:
                C: (horizon * n_bounds, transition_dim * horizon)
                d: (horizon * n_bounds)
        """

        if constraint_list is None:
            constraint_list = self.constraint_list
        else:
            self.constraint_list.extend(constraint_list)

        self.P_cbf_list = []
        self.q_cbf_list = []
        self.v_cbf_list = []
        for constraint in constraint_list:
            type = constraint[0]
            dims = constraint[1]
            center = constraint[2]
            radius = constraint[3]

            P_cbf = np.zeros((self.transition_dim, self.transition_dim))
            q_cbf = np.zeros(self.transition_dim)
            alpha = 1
            b_cbf = 0
            
            #Barrier function
            
            dim_counter = 0
            for dim in dims:
                if self.normalizer is not None:
                    delta_s = self.normalizer.maxs[dim] - self.normalizer.mins[dim]
                    s_min = self.normalizer.mins[dim]
                    P_cbf[dim, dim] = alpha * delta_s ** 2 / 4
                    P_cbf[dim-4, dim] = delta_s ** 2 / 4
                    P_cbf[dim, dim-4] = delta_s ** 2 / 4
                    q_cbf[dim] = -alpha * delta_s * center[dim_counter] + alpha * delta_s * s_min + 0.5 * alpha * delta_s**2
                    q_cbf[dim-4] = 0.5 * delta_s**2 + delta_s * s_min - delta_s * center[dim_counter]
                    # q_cbf[dim] = delta_s**2 / 2 + delta_s * (s_min - center[dim_counter])
                    b_cbf = b_cbf + alpha * center[dim_counter]**2 - (2*alpha * s_min * center[dim_counter] + alpha * delta_s *center[dim_counter]) + alpha * s_min**2 + alpha*(delta_s**2)/4 +  delta_s*alpha*s_min 
                else:
                    P_cbf[dim, dim] = (2 + alpha)
                    q_cbf[dim] = -2 * center[dim_counter] * (alpha + 1)
                    b_cbf =  b_cbf + alpha * center[dim_counter] ** 2
                dim_counter += 1

            b_cbf = b_cbf -alpha  * radius ** 2
            if type == 'sphere_outside':
                P_cbf = -P_cbf
                q_cbf = -q_cbf
                b_cbf = -b_cbf

            self.P_cbf_list.append(P_cbf)
            self.q_cbf_list.append(q_cbf)
            self.v_cbf_list.append(b_cbf)  
