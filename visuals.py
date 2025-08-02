'''
This file is responsible for creating and storing the visuals of the
simulation. It plots all quantaties calculated vs time along with
3d vector trajectories of the velocity, angular momentum, and the
position. Additionally, it can creates an animation that shows the
path of the satelite as a video. All of which is done via
matplotlib.
'''

#imports necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib import cm

#function that creates most time graphs
def create_time_graph(x, y, ylabel, title, filename,xlabel="Time (s)", save_dir="graphs"):
    #sets the style and colors of the plot 
    line_color = "#000014"
    figure_color = "#000014"      
    axes_color = "#BAC1FF"         
    grid_color = "#000014"         
    text_color = "#BAC1FF"
    grid_alpha = 0.3
    font_family = "Rockwell Condensed"
    title_font_size = 21
    label_font_size = 18
    tick_font_size = 16\
    
    #downsamples the plot based on number of points in lists
    n = len(x)
    if n > 99999:
        step = 1000
    elif n > 9999:
        step = 100
    elif n > 999:
        step = 10
    else:
        step = 1
    x = x[::step]
    y = y[::step]

    #creates the filepath to save
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    #adjusts the bounds based on min and max elements
    y_min, y_max = min(y), max(y)
    if y_min >= 0:
        y_lower, y_upper = 0, 1.2 * y_max
    elif y_max <= 0:
        y_lower, y_upper = 1.2 * y_min, 0
    else:
        y_lower, y_upper = 1.2 * y_min, 1.2 * y_max

    #creates the plot and adjusts its colors
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(figure_color)   
    ax.set_facecolor(axes_color)           

    #plots the quantity based on the limits
    ax.plot(x, y, color=line_color, linewidth=2)
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(y_lower, y_upper)

    #configures the plot for text and style across the labels
    ax.grid(True, color=grid_color, alpha=grid_alpha)
    ax.tick_params(axis='both', colors=text_color, labelsize=tick_font_size)
    for label in ax.get_xticklabels() + ax.get_yticklabels(): label.set_fontfamily(font_family)

    #adjusts the labels
    ax.set_title(title, fontsize=title_font_size, color=text_color, fontfamily=font_family)
    ax.set_xlabel(xlabel, fontsize=label_font_size, color=text_color, fontfamily=font_family)
    ax.set_ylabel(ylabel, fontsize=label_font_size, color=text_color, fontfamily=font_family)

    #saves the graph and returns the file path
    plt.tight_layout()
    plt.savefig(filepath, facecolor=fig.get_facecolor())
    plt.close()
    return filepath

#function that creates the time graph for the satelite's radius, has a line for central body radius
def create_radius_time_graph(x, y, central_body_radius, ylabel, title, filename, xlabel="Time (s)", save_dir="graphs"):
    #sets the style and colors of the plot
    line_color = "#000014"
    figure_color = "#000014"       # Exterior
    axes_color = "#BAC1FF"         # Interior
    grid_color = "#000014"         # Grid lines (with alpha below)
    text_color = "#BAC1FF"
    grid_alpha = 0.3
    font_family = "Rockwell Condensed"
    title_font_size = 21
    label_font_size = 18
    tick_font_size = 16

    #downsamples the plot based on number of points in lists
    n = len(x)
    if n > 100000:
        step = 1000
    elif n > 10000:
        step = 100
    elif n > 1000:
        step = 10
    else:
        step = 1
    x = x[::step]
    y = y[::step]

    #creates the filepath to save
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    #adjusts the bounds based on min and max elements
    y_min, y_max = min(min(y), central_body_radius), max(max(y), central_body_radius)
    if y_min >= 0:
        y_lower, y_upper = 0, 1.2 * y_max
    elif y_max <= 0:
        y_lower, y_upper = 1.2 * y_min, 0
    else:
        y_lower, y_upper = 1.2 * y_min, 1.2 * y_max

    #creates the plot and adjusts its colors
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(figure_color)
    ax.set_facecolor(axes_color)

    #plots the postion and the central body radius based on the limits
    ax.plot(x, y, color=line_color, linewidth=2, label="Orbital Radius")
    ax.axhline(y=central_body_radius, color="green", linestyle=":", linewidth=2, label="Central Body Radius")
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(y_lower, y_upper)

    #configures the plot for text and style across the labels
    ax.grid(True, color=grid_color, alpha=grid_alpha)
    ax.tick_params(axis='both', colors=text_color, labelsize=tick_font_size)
    for label in ax.get_xticklabels() + ax.get_yticklabels(): label.set_fontfamily(font_family)

    #adjusts the labels
    ax.set_title(title, fontsize=title_font_size, color=text_color, fontfamily=font_family)
    ax.set_xlabel(xlabel, fontsize=label_font_size, color=text_color, fontfamily=font_family)
    ax.set_ylabel(ylabel, fontsize=label_font_size, color=text_color, fontfamily=font_family)

    #saves the graph and returns the file path
    plt.tight_layout()
    plt.savefig(filepath, facecolor=fig.get_facecolor())
    plt.close()
    return filepath

#function that creates the 3d vector plots
def create_3d_vector_trajectory(array, times, title, filename, save_dir="graphs"):
    #sets the visual styles of the graph
    figure_color = "#000014"
    axes_color = "#BAC1FF"
    text_color = "#BAC1FF"
    font_family = "Rockwell Condensed"
    title_font_size = 28
    label_font_size = 22
    tick_font_size = 14

    #downsamples the results into smaller pieces to proccess
    n = len(array)
    if n > 99999:
        step = 1000
    elif n > 9999:
        step = 100
    elif n > 999:
        step = 10
    else:
        step = 1
    array = array[::step]
    times = times[::step]
    x, y, z = array[:, 0], array[:, 1], array[:, 2]

    #creates the time varying color map
    cmap = mpl.cm.get_cmap('coolwarm')
    norm = mpl.colors.Normalize(vmin=min(times), vmax=max(times))
    colors = cmap(norm(times))

    #opens the directory and the filename
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    #creates the matplotlib graph and changes the style
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor(figure_color)
    ax.set_facecolor(figure_color)

    #create segments for the colors of the graph
    points = np.array([x, y, z]).T
    if np.allclose(points, points[0]):
        ax.scatter(x[0], y[0], z[0], color=colors[0], s=60, label="No variation")

    #fall back in case issue is detected to change the limits
    else:
        segments = np.concatenate([points[:-1].reshape(-1, 1, 3), points[1:].reshape(-1, 1, 3)], axis=1)
        lc = Line3DCollection(segments, colors=colors[:-1], linewidth=2)
        ax.add_collection3d(lc)

        #makes sure plots cannot be insivisble
        ax.auto_scale_xyz(x, y, z)

        #function to slightly expand the limits in case 
        def expand_limits(v):
            vmin, vmax = np.min(v), np.max(v)
            if vmin == vmax:
                return vmin - 1, vmax + 1
            range_ = vmax - vmin
            return vmin - 0.05 * range_, vmax + 0.05 * range_

        ax.set_xlim(expand_limits(x))
        ax.set_ylim(expand_limits(y))
        ax.set_zlim(expand_limits(z))

    #stylizes and configures the panes and axis of the plot
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo["grid"]['color'] = axes_color
        axis._axinfo["grid"]['linewidth'] = 0.8
        axis._axinfo["grid"]['alpha'] = 0.3
        axis._axinfo["axisline"]['color'] = figure_color
        axis._axinfo["tick"]['color'] = text_color
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.set_facecolor(axes_color)
        pane.set_edgecolor(figure_color)

    #makes sure that the aspect ratio is equal
    def set_axes_equal(ax):
        limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
        centers = np.mean(limits, axis=1)
        max_range = np.max(limits[:, 1] - limits[:, 0]) / 2
        ax.set_xlim3d([centers[0] - max_range, centers[0] + max_range])
        ax.set_ylim3d([centers[1] - max_range, centers[1] + max_range])
        ax.set_zlim3d([centers[2] - max_range, centers[2] + max_range])
    set_axes_equal(ax)

    #configures the titles and labels of the figure
    ax.set_title(title, fontsize=title_font_size, color=text_color, fontfamily=font_family)
    ax.set_xlabel("X Component", fontsize=label_font_size, color=text_color, fontfamily=font_family, labelpad=20)
    ax.set_ylabel("Y Component", fontsize=label_font_size, color=text_color, fontfamily=font_family, labelpad=20)
    ax.set_zlabel("Z Component", fontsize=label_font_size, color=text_color, fontfamily=font_family, labelpad=20)
    ax.tick_params(colors=text_color, labelsize=tick_font_size)
    for label in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels(): label.set_fontfamily(font_family)

    #adjusts the specific ticks and fonts and their postion
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        try:
            offset_text = axis.get_offset_text()
            offset_text.set_fontsize(16)
            offset_text.set_color(text_color)
            offset_text.set_fontfamily(font_family)
            offset_text.set_position((offset_text.get_position()[0], offset_text.get_position()[1] - 0.08))
        except Exception:
            pass

    #creates a legend that tells what the colors mean for time
    legend_elements = [Line2D([0], [0], color=cmap(0.0), lw=4, label='Start (Blue)'), Line2D([0], [0], color=cmap(1.0), lw=4, label='End (Red)')]
    legend = ax.legend(handles=legend_elements, loc='upper right', facecolor=axes_color, edgecolor=figure_color, fontsize=18)
    for text in legend.get_texts():
        text.set_color(figure_color)
        text.set_fontfamily(font_family)

    #saves the graph and returns the file path
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05)
    plt.savefig(filepath, facecolor=fig.get_facecolor())
    plt.close()
    return filepath

#function that creates the 3d plot for the position, has the central body as a figure
def create_3d_position_trajectory(array, times, title, filename, central_body_radius, save_dir="graphs"):
    #sets the visual styles of the graph
    figure_color = "#000014"
    axes_color = "#BAC1FF"
    text_color = "#BAC1FF"
    font_family = "Rockwell Condensed"
    title_font_size = 28
    label_font_size = 22
    tick_font_size = 14

    #downsamples the results into smaller pieces to proccess
    n = len(array)
    if n > 99999:
        step = 1000
    elif n > 9999:
        step = 100
    elif n > 999:
        step = 10
    else:
        step = 1
    array = array[::step]
    times = times[::step]
    x, y, z = array[:, 0], array[:, 1], array[:, 2]

    #creates the time varying color map
    cmap = mpl.cm.get_cmap('coolwarm')
    norm = mpl.colors.Normalize(vmin=min(times), vmax=max(times))
    colors = cmap(norm(times))

    #opens the directory and the filename
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    #creates the matplotlib graph and changes the style
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor(figure_color)
    ax.set_facecolor(figure_color)

    #creates the central body as a sphere to put on the figure
    central_body_radius *= 0.97
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    xs = central_body_radius * np.outer(np.cos(u), np.sin(v))
    ys = central_body_radius * np.outer(np.sin(u), np.sin(v))
    zs = central_body_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(xs, ys, zs, color='green', alpha=0.15, linewidth=0, antialiased=True, zorder=1)

    #create segments for the colors of the graph=
    points = np.array([x, y, z]).T
    if np.allclose(points, points[0]):
        ax.scatter(x[0], y[0], z[0], color=colors[0], s=60, label="No variation", zorder=2)
    
    #fall back in case issue is detected to change the limits
    else:
        #adjusts segments based on the points of the satelite's trajectory
        segments = np.concatenate([points[:-1].reshape(-1, 1, 3), points[1:].reshape(-1, 1, 3)], axis=1)
        lc = Line3DCollection(segments, colors=colors[:-1], linewidth=2, zorder=3)
        ax.add_collection3d(lc)
        ax.auto_scale_xyz(x, y, z)

    #changes the limits to have the central body in the origin but also plot the trajectory
    def get_max_extent(*args):
        max_val = max(np.abs(np.concatenate(args)))
        return max(max_val, central_body_radius * 1.05)
    max_extent = get_max_extent(x, y, z)
    margin = 0.1 * max_extent
    ax.set_xlim(-max_extent - margin, max_extent + margin)
    ax.set_ylim(-max_extent - margin, max_extent + margin)
    ax.set_zlim(-max_extent - margin, max_extent + margin)

    #stylizes and configures the panes and axis of the plot
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo["grid"]['color'] = axes_color
        axis._axinfo["grid"]['linewidth'] = 0.8
        axis._axinfo["grid"]['alpha'] = 0.3
        axis._axinfo["axisline"]['color'] = figure_color
        axis._axinfo["tick"]['color'] = text_color
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.set_facecolor(axes_color)
        pane.set_edgecolor(figure_color)

    #makes sure that the aspect ratio is equal
    def set_axes_equal(ax):
        limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
        centers = np.mean(limits, axis=1)
        max_range = np.max(limits[:, 1] - limits[:, 0]) / 2
        ax.set_xlim3d([centers[0] - max_range, centers[0] + max_range])
        ax.set_ylim3d([centers[1] - max_range, centers[1] + max_range])
        ax.set_zlim3d([centers[2] - max_range, centers[2] + max_range])
    set_axes_equal(ax)

    #configures the titles and labels of the figure
    ax.set_title(title, fontsize=title_font_size, color=text_color, fontfamily=font_family)
    ax.set_xlabel("X Position", fontsize=label_font_size, color=text_color, fontfamily=font_family, labelpad=20)
    ax.set_ylabel("Y Position", fontsize=label_font_size, color=text_color, fontfamily=font_family, labelpad=20)
    ax.set_zlabel("Z Position", fontsize=label_font_size, color=text_color, fontfamily=font_family, labelpad=20)
    ax.tick_params(colors=text_color, labelsize=tick_font_size)
    for label in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels(): label.set_fontfamily(font_family)

    #adjusts the specific ticks and fonts and their postion
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        try:
            offset_text = axis.get_offset_text()
            offset_text.set_fontsize(16)
            offset_text.set_color(text_color)
            offset_text.set_fontfamily(font_family)
            offset_text.set_position((offset_text.get_position()[0], offset_text.get_position()[1] - 0.08))
        except Exception:
            pass

    #creates a legend that tells what the colors mean for time
    legend_elements = [Line2D([0], [0], color=cmap(0.0), lw=4, label='Start (Blue)'), Line2D([0], [0], color=cmap(1.0), lw=4, label='End (Red)')]
    legend = ax.legend(handles=legend_elements, loc='upper right', facecolor=axes_color, edgecolor=figure_color, fontsize=18)
    for text in legend.get_texts():
        text.set_color(figure_color)
        text.set_fontfamily(font_family)

    #saves the graph and returns the file path
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05)
    plt.savefig(filepath, facecolor=fig.get_facecolor())
    plt.close()

    return filepath

#function that creates the animation for the satelite's position
def create_3d_position_animation(array, times, title, filename, central_body_radius, save_dir="graphs"):
    #downsamples the positions 
    n = len(array)
    if n > 300:
        step = n // 300
        array = array[::step]
        times = times[::step]
    x, y, z = array[:, 0], array[:, 1], array[:, 2]
    
    #gets the colors of the trajectory
    cmap = cm.get_cmap("coolwarm")
    norm = plt.Normalize(min(times), max(times))
    colors = cmap(norm(times))

    #opens the directory, specified by the user in main
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename + ".mp4")

    #creates the matplotlib 3d plot and configures it
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor("#000014")
    ax.set_facecolor("#000014")

    #plots the central body on the figure
    central_body_radius *= 0.97
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    xs = central_body_radius * np.outer(np.cos(u), np.sin(v))
    ys = central_body_radius * np.outer(np.sin(u), np.sin(v))
    zs = central_body_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(xs, ys, zs, color='green', alpha=0.2, linewidth=0)

    #adjusts the 
    ax.set_title(title, color="#BAC1FF", fontsize=20, fontfamily="Rockwell Condensed")
    ax.set_xlabel("X Position", color="#BAC1FF", fontsize=14, fontfamily="Rockwell Condensed")
    ax.set_ylabel("Y Position", color="#BAC1FF", fontsize=14, fontfamily="Rockwell Condensed")
    ax.set_zlabel("Z Position", color="#BAC1FF", fontsize=14, fontfamily="Rockwell Condensed")
    ax.tick_params(colors="#BAC1FF", labelsize=10)

    #sets the limits
    max_range = max(np.max(np.abs(x)), np.max(np.abs(y)), np.max(np.abs(z)), central_body_radius * 1.2)
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

    trajectory_line, = ax.plot([], [], [], lw=2, color='white')
    point = ax.scatter([], [], [], color='blue', s=20)

    #gets the trajectorys movement and motion
    def init():
        trajectory_line.set_data([], [])
        trajectory_line.set_3d_properties([])
        point._offsets3d = ([], [], [])
        return trajectory_line, point

    #updates the trajectories based on frame
    def update(frame):
        trajectory_line.set_data(x[:frame], y[:frame])
        trajectory_line.set_3d_properties(z[:frame])
        trajectory_line.set_color(colors[frame])
        point._offsets3d = ([x[frame]], [y[frame]], [z[frame]])
        point.set_color(colors[frame])
        return trajectory_line, point

    #creates the animation
    ani = FuncAnimation(fig, update, frames=len(x), init_func=init, blit=True)

    #saves the animation and returns the filepath
    writer = FFMpegWriter(fps=30, metadata=dict(artist='OrbitalSim'), bitrate=1800)
    ani.save(filepath, writer=writer)
    plt.close()
    return filepath
