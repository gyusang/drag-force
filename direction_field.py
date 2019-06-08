import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np
from matplotlib.widgets import Slider, Button

fig = plt.figure('Direction Fields')
ax = fig.add_subplot(111)
plt.subplots_adjust(bottom=0.25)
ax.set_title(r'$\dotv=g-\frac{c_1}{m}v$')
g = 9.8
gamma = 1
v0 = 0


def f(x, t):
    # return x
    return np.array([g - gamma * x[0]])


# Solution curves

t0 = 0
tEnd = 10
dt = 0.001
t = np.arange(t0, tEnd, dt)
# r = RK2(f, np.array([0]), t)
r = odeint(f, np.array([v0]), t)
# print(r)
# ax.plot(t, np.exp(t), color='C%d' % 2)
l, = ax.plot(t, r[:, 0], color='C%d' % 1)
# ax.plot(t, r2[:, 0], color='C%d' % 2)
# ax.plot(t, np.exp(t), color='red')
# Vector field
X, Y = np.meshgrid(np.linspace(0, 10, 20), np.linspace(0, 20, 20))
U = 1
V = g - gamma * Y
# Normalize arrows
N = np.sqrt(U ** 2 + V ** 2)
U2, V2 = U / N, V / N
Q = ax.quiver(X, Y, U2, V2, pivot='mid', color='r', units='height')

plt.xlim([0, 10])
plt.ylim([0, 20])
plt.xlabel(r"$t$")
plt.ylabel(r"$v$")

ax_color = 'lightgoldenrodyellow'
gamma_0 = 1
ax_gamma = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor=ax_color)
ax_v0 = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor=ax_color)
s_gamma = Slider(ax_gamma, r'$c_1/m$', 0.5, 10, valinit=1)
s_v0 = Slider(ax_v0, r'$v_0$', 0, 20, valinit=0)
ax_btn = plt.axes([0.8, 0.15, 0.1, 0.04])
button = Button(ax_btn, 'Hide', color=ax_color, hovercolor='0.975')

showSol = True
def hide(event=None):
    global showSol
    showSol ^= True
    l.set_visible(showSol)
    fig.canvas.draw_idle()
hide()

button.on_clicked(hide)


def update_v0(val):
    v0 = s_v0.val
    l.set_ydata(odeint(f, np.array([v0]), t))
    fig.canvas.draw_idle()


def update_gamma(val):
    global gamma
    gamma = s_gamma.val
    V = g - gamma * Y
    N = np.sqrt(U ** 2 + V ** 2)
    U2, V2 = U / N, V / N
    Q.set_UVC(U2, V2)
    l.set_ydata(odeint(f, np.array([v0]), t))
    fig.canvas.draw_idle()


s_v0.on_changed(update_v0)
s_gamma.on_changed(update_gamma)

plt.show()
