from scipy.optimize import minimize, Bounds
import numpy as np

def idm(counts, s, t):
  return (counts + s*t) / (np.sum(counts, axis=1, keepdims=True)+s)

def monotonicity_violation(p, parent_index, parent_card, cases, sign, epsilon):
  deltas = []
  for a, b in np.ndindex(parent_card, parent_card):
    if not (b > a): continue
    delta = sign*(np.cumsum(p[cases[:, parent_index] == b], axis = 1) \
      - np.cumsum(p[cases[:, parent_index] == a], axis = 1)) + epsilon
    deltas.append(delta)
  return np.concatenate(deltas)

def compute_bounds(variable, parents, cardinality, cases, counts, monotonicities, s=1, epsilon=0.001, tolerance=1e-6):
  def constraint(x):
    t = x.reshape(counts.shape)
    p = idm(counts, s, t)
    total = 0
    for name, sign in monotonicities:
      i = parents.index(name)
      delta = monotonicity_violation(p, parents.index(name), 
                cardinality[name], cases, sign, epsilon)
      total += np.sum(np.square(np.maximum(delta, 0)))
    return total 
  
  t = np.ones_like(counts)
  t /= t.sum(axis=1, keepdims=True)
  x0 = t.reshape(-1)
  x0_lower = x0_upper = x0
  bounds = Bounds(np.zeros_like(x0), np.ones_like(x0), keep_feasible=True)

  for i in range(10):
    weight =  10**i
    f_lower = lambda x: np.sum(x) + weight*constraint(x)
    res_lower = minimize(f, x0=x0_lower, bounds=bounds)  
    x0_lower = np.clip(res_lower.x, 0, 1)
    
    if constraint(x0_lower, s) < tolerance:
      break 
  
  for i in range(10):
    weight =  10**i
    f_upper = lambda x: -np.sum(x) + weight*constraint(x)
    res_upper = minimize(f, x0=x0_upper, bounds=bounds)  
    x0_upper = np.clip(res_upper.x, 0, 1)
    
    if constraint(x0_upper, s) < tolerance:
      break 
  
  t_lower = x0_lower.reshape(counts.shape)
  t_upper = x0_upper.reshape(counts.shape)
  return t_lower, t_upper, constraint(x0_lower, s), constraint(x0_upper, s)

