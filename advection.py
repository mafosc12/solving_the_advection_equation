%matplotlib notebook
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Let's define the parameters

# physics
v = 10.0**(-1) # advection speed

# domain
x_min = -5.0
x_max = 5.0

# initial condition
class Gaussian:
    def __init__(self, center = 0.0, width = 1.0, amplitude = 1.0):
        self.center = center
        self.width = width
        self.amplitude = amplitude
    
    def __call__(self, x):
        return self.amplitude * np.exp(- (x - self.center)**2 / (2 * self.width**2))

gaussian = Gaussian() 
    # gaussian is a function that can take a scalar and return a scalar, 
    # or take a numpy array and return a numpy array

# let's see what the initial condition looks like

X_example = np.linspace(x_min, x_max, 10**2, endpoint = True)
f0_example = gaussian(X_example)

fig = plt.figure()
ax = fig.gca()

ax.set_xlim(x_min, x_max)
ax.set_ylim(-0.1, 1.5)

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$f_{0}(x)$')

plt.plot(X_example, f0_example, 'b--')

plt.show()

# Let's define the parameters

# grid
n_points_space = 10**3
dx = (x_max - x_min) / n_points_space
dt = 0.05

def Upwind_explicit(v, n_points_space, dx, dt):
    df_dt = np.zeros(n_points_space, dtype = np.float64)

    def step(f):
        df_dt = -v * (f - np.roll(f,1)/dx

        f[:] += dt * df_dt
    
    return step

# initial condition
X = np.linspace(x_min, x_max, n_points_space, endpoint = True)
f0 = gaussian(X)
f = f0.copy()

# scheme
step = Upwind_explicit(v, n_points_space, dx, dt)

# preparation of the animation
fig = plt.figure()
ax = fig.gca()

ax.set_xlim(x_min, x_max)
ax.set_ylim(-0.1, 1.5)

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$f(x, t)$')

show_f0, = ax.plot(X, f0, 'b--', label = "f0")
show_ft, = ax.plot([], [], 'b', lw = 1, ms = 2, label = "f")
time = ax.annotate(0, xy = (3, 1.4), xytext = (3, 1.4))

# integration
def animate(i):
    step(f)

    show_ft.set_data(X, f)
    
    time.set_text("t / T = %.2f" % (v * i * dt / (x_max - x_min)))

    return show_f0, show_ft

# creation of the animation
anim = animation.FuncAnimation(fig, animate, 10**5, interval = dt * 10, blit = True)

# show the results
plt.show()

def Lax_Wendroff(v, n_points_space, dx, dt):

    def step(f):
        df_dt = (-v*(np.roll(f,-1)-np.roll(f,1))/(2*dx)+(v**2*dt/2)*(np.roll(f,-1)-2*f+np.roll(f,1))/(dx**2))

        f[:] +=  dt*df_dt 
    
    return step

N = [(i + 1) * 10**2 for i in range(10)] + [(i + 1) * 10**3 for i in range(10)]
Errors = [0.0 for i in range(20)]

for i in range(20):
    n = N[i]

    # grid
    n_points_space = n
    dx = (x_max - x_min) / n_points_space
    n_points_time = 2 * n
    dt = 100 / n_points_time

    # initial condition
    X = np.linspace(x_min, x_max, n_points_space, endpoint = True)
    f0 = gaussian(X)
    f = f0.copy()

    # scheme
    step = Lax_Wendroff(v, n_points_space, dx, dt)
    
    # integration
    t = 0.0
    for j in range(n_points_time):
        step(f)
        t += dt
    
    # measure error
    error = np.max(np.abs(f - f0))
    Errors[i] = error

# plot the results
fig = plt.figure()
ax = fig.gca()

ax.set_xlim(N[0], N[-1])
ax.set_ylim(10**(-6), 10**0)

ax.set_xlabel(r'$n$')
ax.set_ylabel(r'$Error \,\,\, \epsilon$')

ax.set_xscale('log')
ax.set_yscale('log')

plt.plot(N, Errors, 'b')

plt.show()

def Spectral(v, n_points_space, dx, dt):
    k_vals = np.fft.fftfreq(n_points_space, dx)
    def step(f):
        fourier_f = np.fft.fft(f)
        df_dt = -1*j*k_vals*v*fourier_f
        fourier_f[:]+= df-dt * dt
        f[:] = np.fft.ifft(fourier_f)
    
    return step
