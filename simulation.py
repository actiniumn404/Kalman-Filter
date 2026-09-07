import numpy as np
import matplotlib.pyplot as plt


class State:
    def __init__(self, t, x, y, vx=0, vy=0, ax=0, ay=0):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.ax = ax
        self.ay = ay
        self.t = t

    def get_state(self):
        return np.array([self.x, self.y]).T

    def __repr__(self):
        return f"(t={self.t:.3f}, x={self.x:.3f}, y={self.y:.3f})"


class Kalman:
    def __init__(self, ax, ay, vx0, vy0, x0, y0, initial_state, P0, v_cov, R):
        self.ax = ax
        self.ay = ay
        self.vx0 = vx0
        self.vy0 = vy0
        self.x0 = x0
        self.y0 = y0
        self.prev_state: State = initial_state
        self.P = P0
        self.v_cov = v_cov
        self.R = R

    def get_true_state(self, t) -> State:
        ax = self.ax
        ay = self.ay
        vx = ax * t + self.vx0
        vy = ay * t + self.vy0
        x = 0.5 * ax * t * t + self.x0
        y = 0.5 * ay * t * t + self.y0

        return State(t, x, y, vx, vy, ax, ay)

    def get_prev_state(self) -> State:
        return self.prev_state

    def tick(self, t):
        # Priori Estimate
        s = self.get_true_state(t)
        dt = t - self.get_prev_state().t

        F = np.array([[1, 0], [0, 1]])
        B = np.array([[dt, 0], [0, dt]])

        # u = [vx, vy]
        u = np.random.multivariate_normal(np.array([s.vx, s.vy]).T, self.v_cov)

        priori_state = F @ self.get_prev_state().get_state() + B @ u
        priori_covariance = F @ self.P @ F.T + B @ self.v_cov @ B.T

        z = np.random.multivariate_normal(np.array([s.x, s.y]).T, self.R)
        gain = priori_covariance @ np.linalg.inv(priori_covariance + self.R)
        estimate = priori_state + gain @ (z - priori_state)
        covariance = (np.identity(np.shape(gain)[0]) - gain) @ priori_covariance

        self.prev_state = State(t, estimate[0], estimate[1])
        self.P = covariance

        #print("Estimate: ", self.prev_state, "\t\tTrue:", s, "\t\tPriori: ", priori_state, "\t\tMeasurement:", z)
        return self.P, gain

sim = Kalman(1, 5, 0, 0, 0, 0, State(0, 0, 0),
             np.array([[1, 0], [0, 1]]), np.array([[0.333230, 0.006894], [0.006894, 0.235371]]),
             np.array([[0.784125, 0.397491], [0.397491, 0.405683]]))


times = []
values = [[] for i in range(4)]
gain_values = [[] for i in range(4)]

for i in range(1, 30):
    t = i * 0.1
    cov, gain = sim.tick(t)

    times.append(t)
    values[0].append(cov[0][0])
    values[1].append(cov[1][0])
    values[2].append(cov[0][1])
    values[3].append(cov[1][1])

    gain_values[0].append(gain[0][0])
    gain_values[1].append(gain[1][0])
    gain_values[2].append(gain[0][1])
    gain_values[3].append(gain[1][1])

print(values[0][-1], values[1][-1], values[2][-1], values[3][-1])

x_dense = np.linspace(times[0], times[-1], 300)

plt.plot(x_dense, np.interp(x_dense, times, values[0]))
plt.plot(x_dense, np.interp(x_dense, times, values[1]))
plt.plot(x_dense, np.interp(x_dense, times, values[2]))
plt.plot(x_dense, np.interp(x_dense, times, values[3]))


plt.scatter(times, gain_values[0])
plt.scatter(times, gain_values[1])
plt.scatter(times, gain_values[2])
plt.scatter(times, gain_values[3])
plt.show()