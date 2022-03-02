import numpy as np
import matplotlib.pyplot as plt
import time

#heatmap and custom colors
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import ListedColormap

#
#for animations, sliders, buttons, etc
#%matplotlib widget
import ipywidgets as widgets
from IPython.display import display,clear_output

#to run the dashboard in the front and the simulation in the back
import threading

from models import *


def create_colour_map():
    top = cm.get_cmap('autumn', 128)
    bottom = cm.get_cmap('Blues', 128)

    newcolors = np.vstack((top(np.linspace(1, 0, 128)),
                           bottom(np.linspace(0, 1, 128))))
    black = np.array([0 / 256, 0 / 256, 0 / 256, 1])
    grey = np.array([128 / 256, 128 / 256, 128 / 256, 1])
    white = np.array([256 / 256, 256 / 256, 256 / 256, 1])
    turquoise = np.array([48 / 256, 213 / 256, 200 / 256, 1])
    newcolors[0, :] = white
    newcolors[128, :] = grey
    newcolors[129, :] = turquoise
    newcolors[255, :] = black

    return ListedColormap(newcolors, name='OrangeBlue')


newcmp = create_colour_map()


def create_map_with_heading(plt, room, numbers):
    global newcmp

    ax = sns.heatmap(room, vmin=0, vmax=2, annot=numbers, fmt=".3f", \
                     xticklabels=False, yticklabels=False, cbar=False, cmap=newcmp)

    pos_hlines = np.arange(0, room.shape[0]+1, 3)
    pos_vlines = np.arange(0, room.shape[1]+1, 3)

    # print( room.shape[0], len(room[0]), pos_hlines)

    ax.hlines(pos_hlines, *ax.get_xlim())
    ax.vlines(pos_vlines, *ax.get_ylim())


    for t in ax.texts:
        if t.get_text() == "2.000" or t.get_text() == "1.008" or t.get_text() == "0.992" or t.get_text() == "nan":
            t.set_text("")
    return plt


def create_map(plt, room):
    plt.pcolor(room, cmap='Spectral', edgecolors='k', linewidths=3)
    return plt



stop_thread = True
thread = None
mutex = threading.Lock()


def simulation(dash):
    global stop_thread
    print('new simulation started')

    sleep_time = dash.plot_time
    while not stop_thread:

        try:
            dash.update_grid_one_step(False)
            time.sleep(sleep_time)
        finally:
            pass



class Dashboard:

    def __init__(self, ROOM_HEIGHT, ROOM_WIDTH):
        self.slider_h = widgets.IntSlider(min=ROOM_HEIGHT, max=10, step=1, description='Height', value=ROOM_HEIGHT)
        self.slider_w = widgets.IntSlider(min=ROOM_WIDTH, max=10, step=1, description='Width', value=ROOM_WIDTH)
        self.slider_h.observe(self.on_slider_change, names='value')
        self.slider_w.observe(self.on_slider_change, names='value')

        self.out = widgets.Output(layout=widgets.Layout(height='720px', width='720px', border='solid'))

        # self.shellout=widgets.Output()

        self.btn_st = widgets.Button(description='Show transitions')
        self.btn_st.on_click(self.btn_st_eventhandler)

        self.btn_ss = widgets.Button(description='Show sensor')
        self.btn_ss.on_click(self.btn_ss_eventhandler)

        self.btn_if = widgets.Button(description='Init filter')
        self.btn_if.on_click(self.btn_if_eventhandler)

        self.btn_os = widgets.Button(description='One step')
        self.btn_os.on_click(self.btn_os_eventhandler)

        self.btn_go = widgets.Button(description='Go')
        self.btn_go.on_click(self.btn_go_eventhandler)

        self.btn_sp = widgets.Button(description='Stop')
        self.btn_sp.on_click(self.btn_sp_eventhandler)

        self.input_widgets = widgets.HBox([self.slider_h, self.slider_w])
        self.lhs = widgets.VBox([self.btn_st, self.btn_ss])
        self.rhs = self.out
        self.middle = widgets.HBox([self.lhs, self.rhs])
        self.animation = widgets.HBox([self.btn_if, self.btn_os, self.btn_go, self.btn_sp])
        self.db = widgets.VBox([self.input_widgets, self.middle, self.animation])

        # setup of the initial simulation
        self.room = StateModel(self.slider_h.value, self.slider_w.value)
        self.model = Localizer(self.room)

        self.rows, self.cols, self.head = self.room.get_grid_dimensions()
        self.num_states = self.room.get_num_of_states()
        self.total_error = 0
        self.correct_guesses = 0
        self.nbr_of_moves = 0
        self.initialised = False

        self.plot_time = 0.01 * self.rows * self.cols * self.head

        # transition matrix and observation matrix visualization
        self.transition_step = 0
        self.observation_step = self.rows * self.cols

        self.visualizationroom = np.empty(shape=(self.rows * 3, self.cols * 3))

        if self.head == 4:
            self.visrow_iter = [0, 1, 2, 1]
            self.viscol_iter = [1, 2, 1, 0]
        else:
            self.visrow_iter = [0]
            self.viscol_iter = [0]

    def on_slider_change(self, obj):
        global thread
        global stop_thread
        global mutex

        mutex.acquire()
        try:
            # if you change the dimensions of the room we have to stop the ongoing simulation thread
            if thread != None:
                stop_thread = True
                time.sleep(0.1)
                thread.join()
                thread = None
            # setup a new room and model
            self.room = StateModel(self.slider_h.value, self.slider_w.value)
            self.model = Localizer(self.room)
            # reset the counters for steps, accuracy, etc.

            self.rows, self.cols, self.head = self.room.get_grid_dimensions()
            self.num_states = self.room.get_num_of_states()
            self.num_readings = self.room.get_num_of_readings()

            self.total_error = 0
            self.correct_guesses = 0
            self.nbr_of_moves = 0
            self.initialised = False

            self.plot_time = 0.01 * self.rows * self.cols * self.head

            self.visualizationroom = np.empty(shape=(self.rows * 3, self.cols * 3))

            # transition matrix and observation matrix visualization
            self.transition_step = 0
            self.observation_step = self.num_readings - 1


        finally:
            mutex.release()

    def btn_st_eventhandler(self, obj):
        global mutex
        mutex.acquire()
        try:
            # print('Hello from the {} button!'.format(obj.description))
            self.visualizationroom[:] = np.NaN

            # CHECK HERE!!!
            T_hat = self.model.get_transition_model().get_T()[self.transition_step][:]

            for state in range(self.num_states):
                r, c, h = self.room.state_to_pose(state)
                visrow = r * 3 + self.visrow_iter[h]
                viscol = c * 3 + self.viscol_iter[h]

                self.visualizationroom[visrow, viscol] = T_hat[state]

                if self.transition_step == state:
                    self.visualizationroom[visrow, viscol] = 129 / 256 * 2

            plt.close('all')
            plt.figure(1, figsize=(10, 10))
            plot = create_map_with_heading(plt, self.visualizationroom, True)

            if plot == None:
                pass
            else:
                self.update_plt(plot, True)

            self.transition_step += 1
            if self.transition_step >= T_hat.size:
                self.transition_step = 0
        finally:
            mutex.release()

    def btn_ss_eventhandler(self, obj):
        global mutex
        mutex.acquire()
        try:
            # print('Hello from the {} button!'.format(obj.description))
            self.visualizationroom[:] = np.NaN

            for state in range(self.num_states):
                r, c, h = self.room.state_to_pose(state)
                visrow = r * 3 + self.visrow_iter[h]
                viscol = c * 3 + self.viscol_iter[h]

                self.visualizationroom[visrow, viscol] = self.model.get_observation_model().get_o_reading_state(
                    self.observation_step, state)

                if self.observation_step == self.room.state_to_reading(state) \
                        and h == 0 and self.observation_step != self.rows * self.cols:
                    self.visualizationroom[visrow + 1, viscol] = 129 / 256 * 2

            plt.close('all')
            plt.figure(1, figsize=(10, 10))
            plot = create_map_with_heading(plt, self.visualizationroom, True)
            if plot == None:
                pass
            else:
                self.update_plt(plot, True)

            self.observation_step += 1
            if self.observation_step > self.rows * self.cols:
                self.observation_step = 0
        finally:
            mutex.release()

    def btn_if_eventhandler(self, obj):
        global mutex
        global thread
        global stop_thread

        mutex.acquire()
        try:
            # print('Hello from the {} button!'.format(obj.description))

            # reset the counters for steps, accuracy, etc.
            self.total_error = 0
            self.correct_guesses = 0
            self.nbr_of_moves = 0

            # then we can allow to start a new simulation thread
            stop_thread = False
            self.initialised = True

            self.model.initialise()

            true_x, true_y, h = self.model.get_current_true_pose()

            self.visualizationroom[:] = np.NaN

            for state in range(self.num_states):
                r, c, h = self.room.state_to_pose(state)
                visrow = r * 3 + self.visrow_iter[h]
                viscol = c * 3 + self.viscol_iter[h]
                self.visualizationroom[visrow, viscol] = self.model.get_current_f_vector()[state]

            self.visualizationroom[true_x * 3 + 1, true_y * 3 + 1] = 256 / 256 * 2

            plt.close('all')
            plt.figure(1, figsize=(10, 10))
            plot = create_map_with_heading(plt, self.visualizationroom, False)

            if plot == None:
                print("plot is none")
            else:
                self.update_plt(plot, True)
        finally:
            mutex.release()

    def btn_os_eventhandler(self, obj):
        global mutex
        global thread
        try:
            global stop_thread
            if thread != None:
                # if you change the dimensions of the room we have to stop the ongoing simulation thread
                stop_thread = True
                time.sleep(0.1)
                thread.join()
                thread = None

            if self.initialised:
                self.update_grid_one_step(True)
                stop_thread = False
            else:
                print("initialise filter first!")
        finally:
            pass

    def btn_go_eventhandler(self, obj):
        global mutex
        global thread
        global stop_thread

        mutex.acquire()
        try:
            if not self.initialised:
                print("initialize filter first")

            else:
                if not stop_thread and thread == None:
                    # if it isn't, then start it.
                    thread = threading.Thread(target=simulation, args=(self,))
                    thread.start()

        finally:
            mutex.release()

    def btn_sp_eventhandler(self, obj):
        global mutex
        global thread
        global stop_thread

        mutex.acquire()
        try:

            if thread != None:
                stop_thread = True
                time.sleep(0.1)
                thread.join()
                thread = None

            print("thread stopped")

            if self.initialised:
                stop_thread = False

        finally:
            mutex.release()

    def update_grid_one_step(self, plotting):
        global mutex
        mutex.acquire()

        try:
            sensed, trueR, trueC, trueH, sensedR, sensedC, guessedR, guessedC, total_error, f = self.model.update()


            self.visualizationroom[:] = np.NaN
            for state in range(self.num_states):
                r, c, h = self.room.state_to_pose(state)
                visrow = r * 3 + self.visrow_iter[h]
                viscol = c * 3 + self.viscol_iter[h]
                self.visualizationroom[visrow, viscol] = f[state]

            if sensed:
                self.visualizationroom[sensedR * 3 + 1, sensedC * 3 + 1] = 129 / 256 * 2 #turquoise
            self.visualizationroom[guessedR * 3 + 1, guessedC * 3 + 1] = 127 / 256 * 2 #red
            self.visualizationroom[trueR * 3 + 1, trueC * 3 + 1] = 256 / 256 * 2 #black

            plt.close('all')
            plt.figure(1, figsize=(10, 10))
            plot = create_map_with_heading(plt, self.visualizationroom, False)

            if plot == None:
                pass
            else:
                self.update_plt(plot, plotting)

            if sensed:
                print('true pose = <{}, {}, {}>, sensed position = <{}, {}>, guessed position = <{}, {}>'.format(trueR, trueC, trueH, sensedR, sensedC, guessedR, guessedC))
            else:
                print('true pose = <{}, {}, {}>, sensed nothing'.format(trueR, trueC, trueH))


            # update the total error and correct guesses with something useful for the visualisation

            self.total_error += total_error
            if total_error == 0 :
                self.correct_guesses += 1
            self.nbr_of_moves +=1
            print('nbr of moves: {}, avg error: {}, nbr correct guesses: {}'.format(self.nbr_of_moves, self.total_error/self.nbr_of_moves, self.correct_guesses))



        finally:
            mutex.release()
            pass

    def update_plt(self, plt, plotting):
        with self.out:
            clear_output(wait=True)
            plt.figure(1, figsize=(10, 10))
            if plotting:
                display(plt.show())
