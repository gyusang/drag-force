import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np
from matplotlib.widgets import Slider, Button

fig = plt.figure('ODE')
ax = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

plt.subplots_adjust(bottom=0.25)
ax.set_title(r'$\dotv=-\frac{c_1}{m}v-\frac{c_2}{m}v|v|$')
ax2.set_title(r'$\dotx=v$')
g = 9.8
gamma = 0
gamma_2 = 0
v0 = 10
ratio = 0


def f(x, t):
    # return x
    # return np.array([x[1], g - gamma * x[1]])
    return np.array([x[1], - gamma * x[1] - gamma_2 * x[1] * np.abs(x[1])])


def f_l(x, t):
    return np.array([x[1], -gamma * x[1]])


def f_s(x, t):
    return np.array([x[1], -gamma_2 * x[1] * np.abs(x[1])])


# Solution curves

t0 = 0
tEnd = 10
dt = 0.001
t = np.arange(t0, tEnd, dt)
# r = RK2(f, np.array([0]), t)
r_t = odeint(f, np.array([0, v0]), t)
r_l = odeint(f_l, np.array([0, v0]), t)
r_s = odeint(f_s, np.array([0, v0]), t)
# print(r)
# ax.plot(t, np.exp(t), color='C%d' % 2)
l_t1, = ax.plot(t, r_t[:, 1], color='purple', linewidth=3.0)
l_t2, = ax2.plot(t, r_t[:, 0], color='purple', linewidth=3.0)
l_l1, = ax.plot(t, r_l[:, 1], color='red')
l_l2, = ax2.plot(t, r_l[:, 0], color='red')
l_s1, = ax.plot(t, r_s[:, 1], color='blue')
l_s2, = ax2.plot(t, r_s[:, 0], color='blue')
ax.legend(['All', 'Linear', 'Square'])
# ax.plot(t, r2[:, 0], color='C%d' % 2)
# ax.plot(t, np.exp(t), color='red')
# Vector field
X, Y = np.meshgrid(np.linspace(0, 10, 20), np.linspace(0, 20, 20))
U = 1
V = - gamma * Y - gamma_2 * Y * np.abs(Y)
# Normalize arrows
N = np.sqrt(U ** 2 + V ** 2)
U2, V2 = U / N, V / N
Q = ax.quiver(X, Y, U2, V2, pivot='mid', color='r', units='height')

ax.set_xlim([0, 10])
ax.set_ylim([0, 20])
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$v$")
ax2.set_xlim([0, 10])
ax2.set_ylim([0, 20])
ax2.set_xlabel(r"$t$")
ax2.set_ylabel(r"$x$")

ax_color = 'lightgoldenrodyellow'
gamma_0 = 1
ax_gamma = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor=ax_color)
ax_v0 = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor=ax_color)
ax_gamma_2 = plt.axes([0.2, 0.15, 0.65, 0.03])
ax_ratio = plt.axes([0.2, 0, 0.65, 0.03], facecolor=ax_color)
s_gamma_2 = Slider(ax_gamma_2, r'$c_2/m$', 0, 10, valinit=gamma_2)
s_gamma = Slider(ax_gamma, r'$c_1/m$', 0, 10, valinit=gamma)
s_v0 = Slider(ax_v0, r'$v_0$', 0, 20, valinit=10)
s_ratio = Slider(ax_ratio, r'$c_1/c_2$', 0, 20, valinit=0)
ax_btn_d = plt.axes([0.9, 0.1, 0.1, 0.04])
ax_btn_l = plt.axes([0.9, 0.05, 0.1, 0.04])
ax_btn_s = plt.axes([0.9, 0.15, 0.1, 0.04])
button_d = Button(ax_btn_d, 'Direction', color=ax_color, hovercolor='0.975')
button_l = Button(ax_btn_l, 'Linear', color=ax_color, hovercolor='0.975')
button_s = Button(ax_btn_s, 'Square', color=ax_color, hovercolor='0.975')
showLinear = True
showSquare = True
showDirection = True


def hide_d(event=None):
    global showDirection
    showDirection ^= True
    Q.set_visible(showDirection)
    fig.canvas.draw_idle()


def hide_l(event=None):
    global showLinear
    showLinear ^= True
    l_l1.set_visible(showLinear)
    l_l2.set_visible(showLinear)
    fig.canvas.draw_idle()


def hide_s(event=None):
    global showSquare
    showSquare ^= True
    l_s1.set_visible(showSquare)
    l_s2.set_visible(showSquare)
    fig.canvas.draw_idle()


hide_l()
hide_s()
button_d.on_clicked(hide_d)
button_l.on_clicked(hide_l)
button_s.on_clicked(hide_s)


# def onKeyPress(event):
#     global showLinear, showSquare
#     if event.key == "1": showLinear ^= True
#     if event.key == "2": showSquare ^= True
#     l_l1.set_visible(showLinear)
#     l_l2.set_visible(showLinear)
#
#     l_s1.set_visible(showSquare)
#     l_s2.set_visible(showSquare)
#
#
# fig.canvas.mpl_connect('key_press_event', onKeyPress)


# 선형만, 제곱형만, 둘


def update_v0(val):
    global r, l_t1, l_t2, l_l1, l_l2, l_s1, l_s2
    v0 = s_v0.val
    r_t = odeint(f, np.array([0, v0]), t)
    r_l = odeint(f_l, np.array([0, v0]), t)
    r_s = odeint(f_s, np.array([0, v0]), t)
    # print(r)
    # ax.plot(t, np.exp(t), color='C%d' % 2)
    l_t1.set_ydata(r_t[:, 1])
    l_t2.set_ydata(r_t[:, 0])
    l_l1.set_ydata(r_l[:, 1])
    l_l2.set_ydata(r_l[:, 0])
    l_s1.set_ydata(r_s[:, 1])
    l_s2.set_ydata(r_s[:, 0])
    fig.canvas.draw_idle()


def update_gamma(val):
    global gamma, gamma_2
    gamma = s_gamma.val
    gamma_2 = s_gamma_2.val
    if gamma_2 == 0:
        ratio = 0
    else:
        ratio = gamma / gamma_2
    if ratio != s_ratio.val:
        s_ratio.set_val(ratio)

    V = - gamma * Y - gamma_2 * Y * np.abs(Y)
    N = np.sqrt(U ** 2 + V ** 2)
    U2, V2 = U / N, V / N
    Q.set_UVC(U2, V2)
    update_v0(val)


def update_ratio(val):
    global gamma, gamma_2
    ratio = s_ratio.val
    gamma = gamma_2 * ratio
    s_gamma.set_val(gamma)


s_gamma_2.on_changed(update_gamma)
s_v0.on_changed(update_v0)
s_gamma.on_changed(update_gamma)
s_ratio.on_changed(update_ratio)

plt.show()
