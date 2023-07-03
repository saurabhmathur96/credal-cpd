from scipy.optimize import minimize, Bounds


def learn_cpd(variable, parents, counts, monotonicities, epsilon=0.001, s=1, lower=True, tolerance=1e-6):
  cases = np.array(list(np.ndindex(*parent_card)))
  
  def constraint(x):
    t = x.reshape(counts.shape)
    p = (counts + s*t) / (np.sum(counts, axis=1, keepdims=True)+s)
    total = 0
    for name, sign in monotonicities.items():
      i = parents.index(name)
      for a, b in np.ndindex(cardinality[name], cardinality[name]):
        if not (b > a): continue
        delta = sign*(np.cumsum(p[cases[:, i] == b], axis = 1) - np.cumsum(p[cases[:, i] == a], axis = 1)) + epsilon
        total += np.sum((delta > 0)*np.square(delta))
    return total
    
  t = np.ones_like(counts)
  t /= t.sum(axis=1, keepdims=True)
  x0 = t.reshape(-1)
  bounds = Bounds(np.zeros_like(x0), np.ones_like(x0), keep_feasible=True)
  weight = 1
  for i in range(10):
    weight =  10**i
    if lower:
       f = lambda x: np.sum(x) + weight*constraint(x)
    else:
      f = lambda x: -np.sum(x) + weight*constraint(x)

    res = minimize(f, x0=x0, bounds=bounds)
    res.x = np.clip(res.x, 0, 1)
    if constraint(res.x, s) < tolerance:
      break 
    x0 = res.x
  
  t = res.x.reshape(counts.shape)
  p = (counts + s*t) / (np.sum(counts, axis=1, keepdims=True)+s)
  return p, res.x
