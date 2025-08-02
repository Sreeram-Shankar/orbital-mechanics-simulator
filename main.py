'''
This file is the main file of the
program that runs processes the
GUI interface. It contains the main
simulation control layout to collect
user inputs and then displays the
results of the simulation. Tkinter is
used as the GUI tool.
'''

#imports necessary libraries
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import time
import jax.numpy as jnp
from simulator import *
from visuals import *
import shutil
from PIL import Image, ImageTk
import os
import pandas as pd
import zipfile
from pathlib import Path
import threading
import sys
import subprocess

#creates and configure the main window
root = Tk()
root.title("Orbital Simulation Control")
root.geometry("1200x600")
root.config(bg = "#000014")
for i in range(96): root.grid_columnconfigure(i, weight=1); root.grid_rowconfigure(i, weight=1)
    
#creates and configures the main label
main_label = Label(text="Orbital Simulation Control", bg="#000014", fg="#BAC1FF", font=("Rockwell Condensed", 34, "bold"))
main_label.grid(row=0, column=0, columnspan=100, pady=20)

#defines the buttons and images for the buttons to be created
images = []; imagetks = []; buttons = []

#list of parameters that open different simulation windows
parameters = ["Central Body Settings", "Satellite Parameters", "Simulation Settings", "J2 Perturbation", "Atmospheric Drag", "Third Body Perturbation", "Reset parameters", "Start Simulation"]

#array of booleans that make sure inouts are valid before starting, make first true since central body is preset
is_valid = [False] * 6; is_valid[0] = True

#defines default settings
central_body_name = "Earth"; central_body_mass = 5.9718e24; central_body_radius = 6.3781e6
satelite_mass = "enter"; satelite_position = ["enter", "enter", "enter"]; satelite_velocity = ["enter", "enter", "enter"]
time_span = "enter"; time_units = "seconds"; time_step = "enter"; solver = "select"
j2_value = "off"
drag_coefficient = "off"; cross_sectional_area = "off"; atmospheric_model = "off"     
third_body_mass = "off";  third_body_position = ["off", "off", "off"]

#function that restarts the program from the beggining
def restart_program():
    main_label.config(text="Restarting simulation...")
    root.update()
    time.sleep(1)
    root.destroy()
    subprocess.call([sys.executable, sys.argv[0]])

#time step conversions
time_units_to_seconds = {
    "years":   31557600,   # 365.25 days
    "weeks":   604800,     # 7 days
    "days":    86400,      # 24 hours
    "hours":   3600,       # 60 minutes
    "minutes": 60,         # 60 seconds
    "seconds": 1
}

#defines the body of the main window with the current parameters
labels = []
def define_labels():
    global labels

    #clears the existing list just in case
    labels.clear()

    #list of existing variables
    label_data = [
        ("Central Body Name:", central_body_name),
        ("Central Body Mass (kg):", central_body_mass),
        ("Central Body Radius (m):", central_body_radius),
        ("Satellite Mass (kg):", satelite_mass),
        ("Satellite Initial Position Vector (m):", satelite_position),
        ("Satellite Initial Velocity Vector (m/s):", satelite_velocity),
        (f"Simulation Time Span ({time_units}):", time_span),
        ("Simulation Time Step (s):", time_step),
        ("Numerical Solver:", solver),
        ("J2 Perturbation Value:", j2_value),
        ("Atmospheric Drag Coefficient:", drag_coefficient),
        ("Satellite Cross-Sectional Area (m²):", cross_sectional_area),
        ("Atmospheric Model:", atmospheric_model),
        ("Third Body Mass (kg):", third_body_mass),
        ("Third Body Position Vector (m):", third_body_position),
    ]

    #creates specific labels
    for label_text, value in label_data:
        label = Label(text=f"{label_text} {value}", bg="#000014", fg="#BAC1FF", font=("Rockwell Condensed", 18), anchor="center", justify="center")
        labels.append(label)

    #displays the labels on the screen
    split_index = (len(labels) + 1) // 2
    for i, label in enumerate(labels):
        col = 0 if i < split_index else 50
        row_offset = i if i < split_index else i - split_index
        label.grid(row=(8 * row_offset) + 10, column=col, columnspan=50, sticky="nsew")

#function that clears the entry text upon clicking
def clear(event, entry, text):
    try: 
        if entry.get() == text:
            entry.delete(0, END)
        entry.config(fg = "#BAC1FF")
    except:
        pass

#function that restores the placeholder text if empty
def restore(event, entry, text):
    try: 
        if not entry.get():
            entry.insert(0, text)
        entry.config(fg = "#BAC1FF")
    except:
        pass

#function to change the settings of the central body
def change_central_body_settings():
    #creates the pop up window and configures it with a frame
    top = Toplevel(root, bg="#000014")
    top.title("Central Body Settings")
    top.geometry("500x500")
    frame = Frame(top, bg="#000014")
    for i in range(10): frame.grid_columnconfigure(i, weight=1); frame.grid_rowconfigure(i, weight=1)
    frame.pack(fill="both", expand=True)

    #creates the main title label for the central body settings
    main_label = Label(frame, text="Central Body Settings", bg="#000014", fg="#BAC1FF", font=("Rockwell Condensed", 24, "bold"), anchor="center", justify="center")
    main_label.grid(row=0, column=0, columnspan=10, sticky="nsew")

    #creates the secondary label for the central body settings
    secondary_label = Label(frame, text="Select a central body or click to enter custom", bg="#000014", fg="#BAC1FF", font=("Rockwell Condensed", 18), anchor="center", justify="center")
    secondary_label.grid(row=1, column=0, columnspan=10, sticky="nsew")

    #function to create the labels and entry fields for the central body settings
    def labeled_entry(frame, row, label_text, default_value):
        label = Label(frame, text=label_text, bg="#000014", fg="#BAC1FF", font=("Rockwell Condensed", 18), anchor="e", justify="right")
        label.grid(row=row, column=0, columnspan=5, sticky="nsew")
        entry = Entry(frame, font=("Rockwell Condensed", 18), bg="#000014", fg="#BAC1FF", highlightthickness=0, relief="flat", insertbackground="#BAC1FF")
        entry.insert(0, str(default_value))
        entry.grid(row=row, column=5, columnspan=5, sticky="nsew")
        entry.bind("<FocusIn>", lambda e, ent = entry, txt = str(default_value): clear(e, ent, txt))
        entry.bind("<FocusOut>", lambda e, ent = entry, txt = str(default_value): restore(e, ent, txt))
        
        return entry
    
    #creates the entries and buttons
    name_entry   = labeled_entry(frame, 3, "Central Body Name:", central_body_name)
    mass_entry   = labeled_entry(frame, 4, "Central Body Mass (kg):", central_body_mass)
    radius_entry = labeled_entry(frame, 5, "Central Body Radius (m):", central_body_radius)

    #changes the entry fields to the selected central body when the button is clicked
    def set_central_body(name, mass, radius):
        mass_entry.delete(0, END)
        mass_entry.insert(0, str(mass))

        radius_entry.delete(0, END)
        radius_entry.insert(0, str(radius))

        name_entry.delete(0, END)
        name_entry.insert(0, name)

    #creates the buttons for the sun, mars, and jupiter, earth, and venu, and the remaining planets and the moon
    planet_data = {
    "Sun":      (1.989e30, 6.9634e8),
    "Mercury":  (3.3011e23, 2.4397e6),
    "Venus":    (4.8675e24, 6.0518e6),
    "Earth":    (5.9718e24, 6.3781e6),
    "Moon":     (7.34767309e22, 1.7374e6),
    "Mars":     (6.4171e23, 3.3895e6),
    "Jupiter":  (1.8982e27, 6.9911e7),
    "Saturn":   (5.6834e26, 5.8232e7),
    "Uranus":   (8.6810e25, 2.5362e7),
    "Neptune":  (1.02413e26, 2.4622e7),
    }

    #creates the buttons for preset planets 
    row_start = 8
    col_per_button = 2
    for i, (planet, (mass, radius)) in enumerate(planet_data.items()):
        row = row_start + i // 5
        col = (i % 5) * col_per_button
        Button(
            frame, text=planet, bg="#000014", fg="#BAC1FF",
            font=("Rockwell Condensed", 18),
            command=lambda p=planet, m=mass, r=radius: set_central_body(p, m, r)
        ).grid(row=row, column=col, columnspan=2, sticky="nsew")

    #function that saves the change by updating the global variables 
    def save_changes():
        global central_body_mass, central_body_radius, central_body_name
        try:
            #checks for value error or issue in type
            new_mass = float(mass_entry.get())
            new_radius = float(radius_entry.get())
            new_name = name_entry.get()

            #checks for invalid inputs
            if new_mass <= 0 or new_radius <= 0:
                secondary_label.config(text="Please enter valid positive values")
                return

            #checks against current satelite position
            if satelite_position != ["off", "off", "off"] and any(isinstance(x, (int, float)) for x in satelite_position):
                sat_pos = jnp.array(satelite_position, dtype=float)
                if jnp.linalg.norm(sat_pos) <= new_radius:
                    secondary_label.config(text="Radius cannot be lower than satelitte position")
                    return

            #checks against current 3d body position
            if third_body_position != ["off", "off", "off"] and any(isinstance(x, (int, float)) for x in third_body_position):
                third_pos = jnp.array(third_body_position, dtype=float)
                if jnp.linalg.norm(third_pos) <= new_radius:
                    secondary_label.config(text="Radius cannot be lower than third body position")
                    return

            #updates the global variables and main labels
            central_body_name = new_name
            central_body_mass = new_mass
            central_body_radius = new_radius
            top.destroy()
            for label in labels:
                label.destroy()
            labels.clear()
            define_labels()
            is_valid[0] = True

        #throws an error message if there is a value error
        except ValueError:
            secondary_label.config(text="Please enter valid numerical values")
            return
        

    #creates the button to save the changes and close the window
    save_button = Button(frame, text="Save Changes", bg="#000014", fg="#BAC1FF", font=("Rockwell Condensed", 20), command=lambda: save_changes())
    save_button.grid(row=10, column=0, columnspan=10, sticky="nsew")

#function to change the settings of the satellite
def change_satelite_parameters():
    #creates and configures the top level through a frame
    top = Toplevel(root, bg="#000014")
    top.title("Satellite Settings")
    top.geometry("500x500")
    frame = Frame(top, bg="#000014")
    for i in range(10): frame.grid_columnconfigure(i, weight=1)
    for i in range(12): frame.grid_rowconfigure(i, weight=1)
    frame.pack(fill="both", expand=True)

    #creates the main label
    main_label = Label(frame, text="Satellite Parameters", bg="#000014", fg="#BAC1FF", font=("Rockwell Condensed", 24, "bold"))
    main_label.grid(row=0, column=0, columnspan=10, sticky="nsew")

    #creates the secondary label
    secondary_label = Label(frame, text="Click to enter satellite parameters", bg="#000014", fg="#BAC1FF", font=("Rockwell Condensed", 18))
    secondary_label.grid(row=1, column=0, columnspan=10, sticky="nsew")

    #function creates the labels and entries
    def labeled_entry(row, label, default_val):
        Label(frame, text=label, bg="#000014", fg="#BAC1FF", font=("Rockwell Condensed", 18), anchor="e").grid(row=row, column=0, columnspan=5, sticky="nsew")
        entry = Entry(frame, font=("Rockwell Condensed", 18), bg="#000014", fg="#BAC1FF", highlightthickness=0, relief="flat", insertbackground="#BAC1FF")
        entry.grid(row=row, column=5, columnspan=5, sticky="nsew")
        entry.insert(0, str(default_val))
        entry.bind("<FocusIn>", lambda e, ent=entry, txt=str(default_val): clear(e, ent, txt))
        entry.bind("<FocusOut>", lambda e, ent=entry, txt=str(default_val): restore(e, ent, txt))
        return entry

    #creates the entries and labels for each user parameter
    mass_entry = labeled_entry(3, "Satellite Mass (kg):", satelite_mass)
    entries = {}
    ref_labels = [
        ("X Position (m):", satelite_position[0]),
        ("Y Position (m):", satelite_position[1]),
        ("Z Position (m):", satelite_position[2]),
        ("X Velocity (m/s):", satelite_velocity[0]),
        ("Y Velocity (m/s):", satelite_velocity[1]),
        ("Z Velocity (m/s):", satelite_velocity[2]),
    ]
    for i, (label_text, default) in enumerate(ref_labels): entries[i] = labeled_entry(4 + i, label_text, default)

    #confirms parameters by updating global variables
    def save_changes():
        global satelite_mass, satelite_position, satelite_velocity
        try:
            #gets parameters as floats to make
            satelite_mass = float(mass_entry.get())
            pos = [float(entries[i].get()) for i in range(3)]
            vel = [float(entries[i+3].get()) for i in range(3)]

            #checks if mass is negative
            if satelite_mass <= 0:
                secondary_label.config(text="Mass must be postive")
                return

            jnp_pos = jnp.array(pos)
            #checks if position is less than radius
            if jnp.linalg.norm(jnp_pos) <= central_body_radius:
                secondary_label.config(text="Position cannot be within central body radius")
                return

            #updates the global variables and labels
            satelite_position[:] = pos
            satelite_velocity[:] = vel
            top.destroy()
            for label in labels:
                label.destroy()
            labels.clear()
            define_labels()
            is_valid[1] = True

        #displays error messages for value error
        except ValueError:
            secondary_label.config(text="Please enter valid numerical values")

    #creates the button to save user inputs
    save_button = Button(frame, text="Save Changes", bg="#000014", fg="#BAC1FF", font=("Rockwell Condensed", 20), command=save_changes)
    save_button.grid(row=11, column=0, columnspan=10, sticky="nsew")

#function to change the setting of the simulation
def change_simulation_parameters():
    #creates and configures the top level through a frame
    top = Toplevel(root, bg="#000014")
    top.title("Simulation Settings")
    top.geometry("500x500")
    frame = Frame(top, bg="#000014")
    for i in range(10): frame.grid_columnconfigure(i, weight=1)
    for i in range(12): frame.grid_rowconfigure(i, weight=1)
    frame.pack(fill="both", expand=True)

    #creates the main label
    main_label = Label(frame, text="Simulation", bg="#000014", fg="#BAC1FF", font=("Rockwell Condensed", 24, "bold"))
    main_label.grid(row=0, column=0, columnspan=10, sticky="nsew")

    #creates the secondary label
    secondary_label = Label(frame, text="Click to enter simulation settings", bg="#000014", fg="#BAC1FF", font=("Rockwell Condensed", 18))
    secondary_label.grid(row=1, column=0, columnspan=10, sticky="nsew")

    #function creates the labels and entries
    def labeled_entry(row, label, default_val):
        Label(frame, text=label, bg="#000014", fg="#BAC1FF", font=("Rockwell Condensed", 18), anchor="e").grid(row=row, column=0, columnspan=5, sticky="nsew")
        entry = Entry(frame, font=("Rockwell Condensed", 18), bg="#000014", fg="#BAC1FF", highlightthickness=0, relief="flat", insertbackground="#BAC1FF")
        entry.grid(row=row, column=5, columnspan=5, sticky="nsew")
        entry.insert(0, str(default_val))
        entry.bind("<FocusIn>", lambda e, ent=entry, txt=str(default_val): clear(e, ent, txt))
        entry.bind("<FocusOut>", lambda e, ent=entry, txt=str(default_val): restore(e, ent, txt))
        return entry

    #creates the entries through labels for each one
    entries = []
    ref_labels = [
        ("Time Span (units): ", time_span),
        ("Time Step (s):", time_step)
    ]
    for i, (label_text, default) in enumerate(ref_labels): entries.append(labeled_entry(3 + i, label_text, default))

    #function to create a labeled dropdown for solver choice
    def labeled_dropdown(row, label, options, default_val):
        #creates the label the dropdown
        Label(frame, text=label, bg="#000014", fg="#BAC1FF", font=("Rockwell Condensed", 18), anchor="e").grid(row=row, column=0, columnspan=5, sticky="nsew")

        #creates a tkiner stringvar to store th answer
        var = StringVar()
        var.set(default_val)

        #configures a dropfown with the options
        dropdown = OptionMenu(frame, var, *options)
        dropdown.config(font=("Rockwell Condensed", 16), bg="#000014", fg="#BAC1FF", highlightthickness=0, relief="flat", activebackground="#1a1a2e", highlightbackground="#000014", borderwidth=0, anchor="w")
        dropdown["menu"].config(bg="#000014", fg="#BAC1FF", font=("Rockwell Condensed", 16))
        dropdown.grid(row=row, column=5, columnspan=5, sticky="nsew")

        return var, dropdown

    #time units dropdown
    time_units_options = ["years", "weeks", "days", "hours", "minutes", "seconds"]
    time_units_var, _ = labeled_dropdown(5, "Time Units:", time_units_options, time_units)

    #solver dropdown that presents solvers available from diffrax
    solver_options = ["Euler", "Heun", "Midpoint", "RK5", "RK8", "TSIT5", "DOP8", "Symp. Euler", "Rev. Heun", "Leapfrog", "Imp. Euler", "KVAERNO3", "KVAERNO5" ]
    solver_var, _ = labeled_dropdown(6, "Numerical Solver:", solver_options, solver)

    #function to save changes
    def save_changes():
        global time_span, time_step, time_units, solver
        try:
            #confirms that the time span and step are numerical
            ts = float(entries[0].get()); dt = float(entries[1].get())
            time_units = time_units_var.get()
            solver = solver_var.get()

            #checks if values is negative
            if ts <= 0 or dt <= 0:
                secondary_label.config(text="Time span must be greater than time step")
                return

            #checks if time span is less than time step
            sec_time = ts*time_units_to_seconds[time_units]
            if sec_time <= dt:
                secondary_label.config(text="Time span must be greater than time step")
                return

            #updates the global variables and labels
            time_step = dt; time_span = ts
            top.destroy()
            for label in labels:
                label.destroy()
            labels.clear()
            define_labels()
            is_valid[2] = True

        #displays error messages 
        except ValueError:
            secondary_label.config(text="Please enter valid numerical values")

    #creates the button to save user inputs
    save_button = Button(frame, text="Save Changes", bg="#000014", fg="#BAC1FF", font=("Rockwell Condensed", 20), command=save_changes)
    save_button.grid(row=11, column=0, columnspan=10, sticky="nsew")

#function to change the j2 perturbation settings
def change_j2_settings():
    #create and configure the top-level window
    top = Toplevel(root, bg="#000014")
    top.title("J2 Perturbation Settings")
    top.geometry("500x500")
    frame = Frame(top, bg="#000014")
    for i in range(10): frame.grid_columnconfigure(i, weight=1)
    for i in range(14): frame.grid_rowconfigure(i, weight=1)
    frame.pack(fill="both", expand=True)

    #creates the main and secondary labele
    main_label = Label(frame, text="J2 Perturbation Settings", bg="#000014", fg="#BAC1FF", font=("Rockwell Condensed", 24, "bold"))
    main_label.grid(row=0, column=0, columnspan=10, sticky="nsew")
    secondary_label = Label(frame, text="Toggle J2 and enter or select a value", bg="#000014", fg="#BAC1FF", font=("Rockwell Condensed", 18))
    secondary_label.grid(row=1, column=0, columnspan=10, sticky="nsew")

    #toggle state for j2 variable
    toggle_var = BooleanVar(value=(j2_value != "off"))
    j2_value_var = StringVar()
    j2_value_var.set("" if j2_value == "off" else str(j2_value))

    #function that toggles between the user turning on and off j2
    def toggle_state():
        if toggle_var.get():
            #puts the preset buttons on if turned on
            j2_entry.config(state="normal")
            for b in planet_buttons: b.grid()
        else:
            #removes the entries if turned off
            j2_entry.delete(0, END)
            j2_value_var.set("off")
            j2_entry.config(state="disabled")
            for b in planet_buttons: b.grid_remove()

    #creates and configures the button for toggling 
    toggle_btn = Checkbutton(frame, text="J2 Perturbation: ON", variable=toggle_var, command=toggle_state, bg="#000014", fg="#BAC1FF", font=("Rockwell Condensed", 18), selectcolor="#000014", activebackground="#000014", onvalue=True, offvalue=False)
    toggle_btn.grid(row=3, column=0, columnspan=10, sticky="nsew")
    toggle_btn.deselect() if j2_value == "off" else toggle_btn.select()

    #creates the lavel and entry for the j2 value display
    Label(frame, text="J2 Value:", bg="#000014", fg="#BAC1FF", font=("Rockwell Condensed", 18), anchor="e").grid(row=5, column=0, columnspan=5, sticky="nsew")
    j2_entry = Entry(frame, font=("Rockwell Condensed", 18), bg="#000014", fg="#BAC1FF",highlightthickness=0, relief="flat", insertbackground="#BAC1FF", disabledbackground="#000014", disabledforeground="#BAC1FF")
    j2_entry.grid(row=5, column=5, columnspan=5, sticky="nsew")
    j2_entry.insert(0, j2_value_var.get())
    j2_entry.config(state="normal" if toggle_var.get() else "disabled")

    #default planet j2 values
    j2_data = {
        "Sun":      0.0,
        "Mercury":  0.0,
        "Venus":    0.0,
        "Earth":    1.08263e-3,
        "Moon":     2.034e-4,
        "Mars":     1.96045e-3,
        "Jupiter":  1.4697e-2,
        "Saturn":   1.6298e-2,
        "Uranus":   3.34343e-3,
        "Neptune":  3.411e-3
    }

    #creates and places the buttons for the default planet j2 values
    planet_buttons = []
    row_start = 11
    col_per_button = 2
    for i, (planet, val) in enumerate(j2_data.items()):
        row = row_start + i // 5
        col = (i % 5) * col_per_button
        btn = Button(frame, text=planet, bg="#000014", fg="#BAC1FF", font=("Rockwell Condensed", 16), command=lambda v=val: j2_entry.delete(0, END) or j2_entry.insert(0, str(v)))
        btn.grid(row=row, column=col, columnspan=2, sticky="nsew")
        planet_buttons.append(btn)

    #makes sure the buttons are removed if toggle if off
    if not toggle_var.get():
        for b in planet_buttons: b.grid_remove()

    #function that saves and exits
    def save_changes():
        global j2_value
        #checks to make sure the toggle is on before pdating
        if toggle_var.get():
            try:
                #checks if input is a float
                j2_value = float(j2_entry.get())

                #chekcs if the j2 value is negative
                if j2_value < 0:
                    secondary_label.config("J2 value cannot be negaive")
                    return
                
            except ValueError:
                #displays an erorr message if in valid
                secondary_label.config(text="Plese make sure J2 value is valid")
                return
        else:
            j2_value = "off"
            is_valid[3] = False

        #saves the global variables and updates the labels
        top.destroy()
        for label in labels:
            label.destroy()
        labels.clear()
        define_labels()
        is_valid[3] = True

    #button that saves the values 
    save_button = Button(frame, text="Save Changes", bg="#000014", fg="#BAC1FF", font=("Rockwell Condensed", 20), command=save_changes)
    save_button.grid(row=13, column=0, columnspan=10, sticky="nsew")

#function to change the settings of atmospheric drag
def change_atmospheric_drag_settings():
    #create and configure the top-level window
    top = Toplevel(root, bg="#000014")
    top.title("Atmospheric Drag Settings")
    top.geometry("500x500")
    frame = Frame(top, bg="#000014")
    for i in range(10): frame.grid_columnconfigure(i, weight=1)
    for i in range(20): frame.grid_rowconfigure(i, weight=1)
    frame.pack(fill="both", expand=True)

    #creates the main and secondary label
    main_label = Label(frame, text="Atmospheric Drag Settings", bg="#000014", fg="#BAC1FF", font=("Rockwell Condensed", 24, "bold"))
    main_label.grid(row=0, column=0, columnspan=10, sticky="nsew")
    secondary_label = Label(frame, text="Toggle atmospheric drag and enter parameters", bg="#000014", fg="#BAC1FF", font=("Rockwell Condensed", 18))
    secondary_label.grid(row=1, column=0, columnspan=10, sticky="nsew")

    #toggle state for atmospheric drag
    toggle_var = BooleanVar(value=(drag_coefficient != "off"))
    p0_val_mem = "" if atmospheric_model == "off" else atmospheric_model.split("*")[0]
    h_val_mem = "" if atmospheric_model == "off" else atmospheric_model.split("/")[1][:-1]

    #creates the toggle button
    toggle_btn = Checkbutton(frame, text="Atmospheric Drag: ON" if toggle_var.get() else "Atmospheric Drag: OFF", variable=toggle_var, bg="#000014", fg="#BAC1FF", font=("Rockwell Condensed", 18), selectcolor="#000014", activebackground="#000014", onvalue=True, offvalue=False, command=lambda: toggle_state())
    toggle_btn.grid(row=3, column=0, columnspan=10, sticky="nsew")

    #creates the label and entry for drag coefficient
    Label(frame, text="Drag Coefficient:", bg="#000014", fg="#BAC1FF", font=("Rockwell Condensed", 18), anchor="e").grid(row=4, column=0, columnspan=5, sticky="nsew")
    drag_entry = Entry(frame, font=("Rockwell Condensed", 18), bg="#000014", fg="#BAC1FF", highlightthickness=0, relief="flat", insertbackground="#BAC1FF", disabledbackground="#000014", disabledforeground="#BAC1FF")
    drag_entry.grid(row=4, column=5, columnspan=5, sticky="nsew")

    #creates the label and entry for cross sectional area
    Label(frame, text="Cross-Sectional Area (m^2):", bg="#000014", fg="#BAC1FF", font=("Rockwell Condensed", 18), anchor="e").grid(row=5, column=0, columnspan=5, sticky="nsew")
    area_entry = Entry(frame, font=("Rockwell Condensed", 18), bg="#000014", fg="#BAC1FF", highlightthickness=0, relief="flat", insertbackground="#BAC1FF", disabledbackground="#000014", disabledforeground="#BAC1FF")
    area_entry.grid(row=5, column=5, columnspan=5, sticky="nsew")

    #creates the label and entry for p0
    Label(frame, text="Atmospheric Density p0 (kg/m^3):", bg="#000014", fg="#BAC1FF", font=("Rockwell Condensed", 18), anchor="e").grid(row=7, column=0, columnspan=5, sticky="nsew")
    p0_entry = Entry(frame, font=("Rockwell Condensed", 18), bg="#000014", fg="#BAC1FF", highlightthickness=0, relief="flat", insertbackground="#BAC1FF", disabledbackground="#000014", disabledforeground="#BAC1FF")
    p0_entry.grid(row=7, column=5, columnspan=5, sticky="nsew")

    #creates the label and entry for scale height
    Label(frame, text="Scale Height H (m):", bg="#000014", fg="#BAC1FF", font=("Rockwell Condensed", 18), anchor="e").grid(row=8, column=0, columnspan=5, sticky="nsew")
    h_entry = Entry(frame, font=("Rockwell Condensed", 18), bg="#000014", fg="#BAC1FF", highlightthickness=0, relief="flat", insertbackground="#BAC1FF", disabledbackground="#000014", disabledforeground="#BAC1FF")
    h_entry.grid(row=8, column=5, columnspan=5, sticky="nsew")

    #default atmospheric parameters per planet
    planet_data = {
        "Sun":      (0.0, 1.0),
        "Mercury":  (0.0, 1.0),
        "Venus":    (65.0, 15900),
        "Earth":    (1.225, 8500),
        "Moon":     (0.0, 1.0),
        "Mars":     (0.020, 11100),
        "Jupiter":  (0.16, 27000),
        "Saturn":   (0.19, 59000),
        "Uranus":   (0.42, 27000),
        "Neptune":  (0.45, 24000)
    }

    #creates the planet buttons
    planet_buttons = []
    row_start = 16
    col_per_button = 2
    for i, (planet, (p0_val, h_val)) in enumerate(planet_data.items()):
        row = row_start + i // 5
        col = (i % 5) * col_per_button
        btn = Button(frame, text=planet, bg="#000014", fg="#BAC1FF", font=("Rockwell Condensed", 16), command=lambda p=p0_val, h=h_val: (p0_entry.delete(0, END), p0_entry.insert(0, str(p)), h_entry.delete(0, END), h_entry.insert(0, str(h))))
        btn.grid(row=row, column=col, columnspan=2, sticky="nsew")
        planet_buttons.append(btn)

    #function that toggles the switch and sets up the panel
    def toggle_state():
        #configures the label acording to the toggle state
        state = "normal" if toggle_var.get() else "disabled"
        for widget in [drag_entry, area_entry, p0_entry, h_entry]:
            #removes the entries if not toggled
            if not toggle_var.get():
                widget.delete(0, END)
            widget.config(state=state)
        toggle_btn.config(text="Atmospheric Drag: ON" if toggle_var.get() else "Atmospheric Drag: OFF")
        
        #removes or resets the buttons depending on the state of the toggle
        for btn in planet_buttons:
            if toggle_var.get():
                btn.grid()
            else:
                btn.grid_remove()
                
        #completes the actual toggling of the switch and its effects on the entries 
        if toggle_var.get():
            if drag_coefficient != "off":
                drag_entry.delete(0, END)
                area_entry.delete(0, END)
                p0_entry.delete(0, END)
                h_entry.delete(0, END)
                drag_entry.insert(0, str(drag_coefficient))
                area_entry.insert(0, str(cross_sectional_area))
                p0_entry.insert(0, p0_val_mem)
                h_entry.insert(0, h_val_mem)

    #starts the toggling of the button
    toggle_state()

    #function to save changes
    def save_changes():
        global drag_coefficient, cross_sectional_area, atmospheric_model
        #checks if the toggle is on before validating inputs
        if toggle_var.get():
            try:
                #gets user inputs as floats
                cd = float(drag_entry.get())
                area = float(area_entry.get())
                p0 = float(p0_entry.get())
                h = float(h_entry.get())

                #checks if any of the inputs are negative
                if cd < 0 or area <= 0 or p0 < 0 or h <= 0:
                    secondary_label.config(text="Please validate inputs as above or at zero")
                    return
                
            except ValueError:
                secondary_label.config(text="Please enter valid numerical values")
                return
        else:
            drag_coefficient = "off"
            cross_sectional_area = "off"
            atmospheric_model = "off"
            is_valid[4] = False

        #updates the global variables and 
        drag_coefficient = cd
        cross_sectional_area = area
        atmospheric_model = f"{p0}*exp(-h/{h})"
        top.destroy()
        for label in labels:
            label.destroy()
        labels.clear()
        define_labels()
        is_valid[4] = True

    #button to confirm and save values
    save_button = Button(frame, text="Save Changes", bg="#000014", fg="#BAC1FF", font=("Rockwell Condensed", 20), command=save_changes)
    save_button.grid(row=18, column=0, columnspan=10, sticky="nsew")

#function to update third body settings
def change_third_body_settings():
    #creates and configures the top level and frame
    top = Toplevel(root, bg="#000014")
    top.title("Third Body Perturbation Settings")
    top.geometry("500x500")
    frame = Frame(top, bg="#000014")
    for i in range(10): frame.grid_columnconfigure(i, weight=1)
    for i in range(20): frame.grid_rowconfigure(i, weight=1)
    frame.pack(fill="both", expand=True)

    #create the main label and the secondary label
    main_label = Label(frame, text="Third Body Perturbation Settings", bg="#000014", fg="#BAC1FF", font=("Rockwell Condensed", 24, "bold"), anchor="center", justify="center").grid(row=0, column=0, columnspan=10, sticky="nsew")
    secondary_label = Label(frame, text="Toggle perturbation and enter or select parameters", bg="#000014", fg="#BAC1FF", font=("Rockwell Condensed", 18), anchor="center", justify="center")
    secondary_label.grid(row=1, column=0, columnspan=10, sticky="nsew")

    #variables to deal with the toggling of the function
    toggle_var = BooleanVar(value=(third_body_mass != "off"))
    flip_state = BooleanVar(value=False)

    #function that creates the entries and labels
    def labeled_entry(row, label_text, default):
        Label(frame, text=label_text, bg="#000014", fg="#BAC1FF", font=("Rockwell Condensed", 18), anchor="e").grid(row=row, column=0, columnspan=5, sticky="nsew")
        entry = Entry(frame, font=("Rockwell Condensed", 18), bg="#000014", fg="#BAC1FF", highlightthickness=0, relief="flat", insertbackground="#BAC1FF", disabledbackground="#000014", disabledforeground="#BAC1FF")
        entry.grid(row=row, column=5, columnspan=5, sticky="nsew")
        entry.insert(0, str(default))
        entry.bind_all("<FocusIn>", lambda e: clear(e, entry, default))
        entry.bind_all("<FocusOut>", lambda e: restore(e, entry, default))
        return entry

    #creates the entries and labels
    mass_entry = labeled_entry(3, "Third Body Mass (kg):", "")
    x_entry = labeled_entry(4, "X Position (m):", "")
    y_entry = labeled_entry(5, "Y Position (m):", "")
    z_entry = labeled_entry(6, "Z Position (m):", "")
    entry_widgets = [mass_entry, x_entry, y_entry, z_entry]

    #loads previous values if third body perturbation was on
    if third_body_mass != "off":
        mass_entry.delete(0, END)
        x_entry.delete(0, END)
        y_entry.delete(0, END)
        z_entry.delete(0, END)
        mass_entry.insert(0, str(third_body_mass))
        x_entry.insert(0, str(third_body_position[0]))
        y_entry.insert(0, str(third_body_position[1]))
        z_entry.insert(0, str(third_body_position[2]))
    
    #third body positions for each of the planets to the sun, the moon to the earth, and the sun to the moon
    presets = {
        "(Sun, Mercury)": [(1.989e30, [5.791e10, 0, 0]), (3.301e23, [-5.791e10, 0, 0])],
        "(Sun, Venus)": [(1.989e30, [1.082e11, 0, 0]), (4.867e24, [-1.082e11, 0, 0])],
        "(Sun, Earth)": [(1.989e30, [1.496e11, 0, 0]), (5.972e24, [-1.496e11, 0, 0])],
        "(Sun, Mars)": [(1.989e30, [2.279e11, 0, 0]), (6.417e23, [-2.279e11, 0, 0])],
        "(Sun, Jupiter)": [(1.989e30, [7.785e11, 0, 0]), (1.898e27, [-7.785e11, 0, 0])],
        "(Sun, Saturn)": [(1.989e30, [1.433e12, 0, 0]), (5.683e26, [-1.433e12, 0, 0])],
        "(Sun, Uranus)": [(1.989e30, [2.877e12, 0, 0]), (8.681e25, [-2.877e12, 0, 0])],
        "(Sun, Neptune)": [(1.989e30, [4.503e12, 0, 0]), (1.024e26, [-4.503e12, 0, 0])],
        "(Sun, Moon)": [(1.989e30, [1.53444e11, 0, 0]), (7.348e22, [-1.53444e11, 0, 0])],
        "(Moon, Earth)": [(7.348e22, [3.844e8, 0, 0]), (5.972e24, [-3.844e8, 0, 0])],
    }

    #function that handles the toggling
    def toggle_state():
        state = "normal" if toggle_var.get() else "disabled"
        for w in entry_widgets:
            w.delete(0, END)
            w.config(state=state)
        for b in planet_buttons:
            b.grid() if toggle_var.get() else b.grid_remove()
        toggle_btn.config(text="Third Body Perturbation: ON" if toggle_var.get() else "Third Body Perturbation: OFF")

    #creates and configures the toggle buttons
    toggle_btn = Checkbutton(frame, variable=toggle_var, command=toggle_state, bg="#000014", fg="#BAC1FF", font=("Rockwell Condensed", 18), selectcolor="#000014", activebackground="#000014", onvalue=True, offvalue=False)
    toggle_btn.grid(row=2, column=0, columnspan=10, sticky="nsew")
    toggle_btn.config(text="Third Body Perturbation: ON" if toggle_var.get() else "Third Body Perturbation: OFF")

    #creates and configures the butons for preset thrid body values
    planet_buttons = []
    for i, (label, data) in enumerate(presets.items()):
        row = 16 + (i // 5)
        col = (i % 5) * 2

        #function that flips the preset if the button is pressed on ttwice
        def make_command(preset=data):
            def apply():
                mass, pos = preset[1] if flip_state.get() else preset[0]
                mass_entry.delete(0, END); mass_entry.insert(0, str(mass))
                for e, v in zip([x_entry, y_entry, z_entry], pos):
                    e.delete(0, END); e.insert(0, str(v))
                flip_state.set(not flip_state.get())
            return apply
        
        #actually creates the buttons for the preset values
        btn = Button(frame, text=label, command=make_command(), font=("Rockwell Condensed", 16), bg="#000014", fg="#BAC1FF")
        btn.grid(row=row, column=col, columnspan=2, sticky="nsew")
        planet_buttons.append(btn)

    #function that removez the buttons if the toggle is off
    if not toggle_var.get():
        for b in planet_buttons: b.grid_remove()

    #function that saves the user's inputs
    def save_changes():
        global third_body_mass, third_body_position
        if toggle_var.get():
            try:
                #makes sure
                third_body_mass = float(mass_entry.get())
                position = [float(x_entry.get()), float(y_entry.get()), float(z_entry.get())]
                
                #checks for valid mass
                if third_body_mass <= 0:
                    secondary_label.config(text="Mass cannot be greater than zero")
                    return
                    
                #checks if position is outside central body
                jnp_pos = jnp.array(position)
                if jnp.linalg.norm(jnp_pos) <= central_body_radius:
                    secondary_label.config(text="Position cannot be outside central body")
                    return
                    
                third_body_position = position
                
            #displays error message if invalid
            except ValueError:
                secondary_label.config(text="Please enter valid numerical values")
                return
        else:
            third_body_mass = "off"
            third_body_position = ["off", "off", "off"]
            is_valid[5] = False
        
        #destroys the top level and updates the main labels
        top.destroy()
        for label in labels:
            label.destroy()
        labels.clear()
        define_labels()
        is_valid[5] = True

    #creates the button to save the code
    save_button = Button(frame, text="Save Changes", bg="#000014", fg="#BAC1FF", font=("Rockwell Condensed", 20), command=save_changes)
    save_button.grid(row=18, column=0, columnspan=10, sticky="nsew")

#function that resets all the paramters
def reset():
    #resets all the parameters like the original
    global central_body_name, central_body_mass, central_body_radius
    central_body_name = "Earth"; central_body_mass = 5.9718e24; central_body_radius = 6.3781e6

    global satelite_mass, satelite_position, satelite_velocity
    satelite_mass = "enter"; satelite_position = ["enter", "enter", "enter"]; satelite_velocity = ["enter", "enter", "enter"]

    global time_span, time_units, time_step, solver
    time_span = "enter"; time_units = "years";time_step = "enter"; solver = "select"

    global j2_value
    j2_value = "off"

    global drag_coefficient, cross_sectional_area, atmospheric_model
    drag_coefficient = "off"; cross_sectional_area = "off"; atmospheric_model = "off"   

    global third_body_mass, third_body_position
    third_body_mass = "off";  third_body_position = ["off", "off", "off"]

    global is_valid
    is_valid = [False] * 6

    #updates the main labels
    for label in labels:
        label.destroy()
    labels.clear()
    define_labels()

#function that confirms that the program is ready to begin and then starts it
def run_simulation():
    #indexes of the is valid boolean array that need to be checked
    check_indexes = [0, 1, 2]

    #adds the correct optional values if on
    if j2_value != "off": check_indexes.append(3)
    if drag_coefficient != "off": check_indexes.append(4)
    if third_body_mass != "off": check_indexes.append(5)

    #checks to make sure that all the parameters have been inputted
    if not all(is_valid[i] for i in check_indexes):
        #issues an error message and then waits and resets it
        main_label.config(text="Please make sure all parameters are inputted")
        root.update()
        time.sleep(3)
        main_label.config(text="Orbital Simulation Control")
        root.update()
        return
    
    #updates time span to seconds
    global time_span
    time_span = time_span*time_units_to_seconds[time_units]

    #gets the array of inputs to send to file
    main_parameters = [central_body_name, central_body_radius, central_body_mass, satelite_mass, satelite_position, satelite_velocity, time_span, time_step, solver, j2_value, drag_coefficient, cross_sectional_area, atmospheric_model, third_body_mass, third_body_position]

    #upates the main labels and screen
    for label in labels: 
        label.grid_remove()
        label.destroy()
    labels.clear()
    for button in buttons: 
        button.grid_remove()
        button.destroy()
    buttons.clear()
    main_label.config(text="Running Simulation\nMay Take Time to Process")
    main_label.grid(rowspan=96)
    root.update()

    #calls the simulator
    try:
        (times, positions, radius, velocities, v_magnitude, angular_momentum, angular_magnitude, angular_drift, percent_angular_drift, 
            kinetic_energy, potential_energy, total_energy, energy_drift, percent_energy_drift, 
            inclination_deg, semi_major_axis, eccentricity, raan_deg, arg_peri_deg, true_anom_deg, crash_index) = simulation_runner(main_parameters)
    
    #if the runner did not work, it throws an error message and allows the user to restart the program
    except Exception as e:
        main_label.config(text=f"The Simulation Encountered an Error:\n{e}")
        root.update()
        
        #creates the button that restarts the program
        restart_button = Button(root, text="Restart Simulation", command=restart_program, bg="#000014", fg="#BAC1FF", font=("Rockwell Condensed", 24, "bold"), relief="flat", activebackground="#000014", activeforeground="#BAC1FF", highlightthickness=0, bd=0, cursor="hand2"); 
        restart_button.grid(row=70, columnspan=96, sticky="nsew")
        return
    
    #updates the label after the simulator is finished
    main_label.config(text="Processing Results\nMay Take Time to Process")
    root.update()

    #a dictionary that contains the correct units for each quantity
    units = {
        "Radius": "m",
        "Position X": "m",
        "Position Y": "m",
        "Position Z": "m",
        "Velocity Magnitude": "m/s",
        "Velocity X": "m/s",
        "Velocity Y": "m/s",
        "Velocity Z": "m/s",
        "Angular Momentum Magnitude": "kg·m²/s",
        "Angular Momentum X": "kg·m²/s",
        "Angular Momentum Y": "kg·m²/s",
        "Angular Momentum Z": "kg·m²/s",
        "Angular Momentum Drift": "kg·m²/s",
        "Percent Angular Drift": "%",
        "Kinetic Energy": "J",
        "Potential Energy": "J",
        "Total Energy": "J",
        "Energy Drift": "J",
        "Percent Energy Drift": "%",
        "Inclination": "deg",
        "Semi Major Axis": "m",
        "Eccentricity": "",
        "RAAN": "deg",
        "Argument of Periapsis": "deg",
        "True Anomaly": "deg"
    }

    #a dictionary that actually holds the value of the results from the simulation
    data_dict = {
        "Position X": positions[:, 0],
        "Position Y": positions[:, 1],
        "Position Z": positions[:, 2],
        "Velocity Magnitude": v_magnitude,
        "Velocity X": velocities[:, 0],
        "Velocity Y": velocities[:, 1],
        "Velocity Z": velocities[:, 2],
        "Angular Momentum Magnitude": angular_magnitude,
        "Angular Momentum X": angular_momentum[:, 0],
        "Angular Momentum Y": angular_momentum[:, 1],
        "Angular Momentum Z": angular_momentum[:, 2],
        "Angular Momentum Drift": angular_drift,
        "Percent Angular Drift": percent_angular_drift,
        "Kinetic Energy": kinetic_energy,
        "Potential Energy": potential_energy,
        "Total Energy": total_energy,
        "Energy Drift": energy_drift,
        "Percent Energy Drift": percent_energy_drift,
        "Inclination": inclination_deg,
        "Semi Major Axis": semi_major_axis,
        "Eccentricity": eccentricity,
        "RAAN": raan_deg,
        "Argument of Periapsis": arg_peri_deg,
        "True Anomaly": true_anom_deg
    }

    #creates all of the graphs for the quantaties over time
    shutil.rmtree("graphs")
    image_filepaths = []
    for name, values in data_dict.items():
        unit = f" ({units[name]})" if units.get(name) else ""
        ylabel = name + unit
        title = f"{name} vs Time"
        filename = f"{name.lower().replace(' ', '_')}_vs_time.png"
        image_filepaths.append(create_time_graph(x=times, y=values, ylabel=ylabel, title=title, filename=filename))

    #creates a variation of the previous graph that includes the radius of the central body for reference
    image_filepaths.append(create_radius_time_graph(x=times, y=radius, central_body_radius=central_body_radius, ylabel="Radius (m)", title="Orbital Radius vs Time", filename="radius_vs_time.png"))

    #creates the 3d vector plots for the velocity and the angular momentum
    image_filepaths.append(create_3d_vector_trajectory(velocities, times, "Velocity Vector Trajectory (m/s)", "velocity_3d_trajectory.png"))
    image_filepaths.append(create_3d_vector_trajectory(angular_momentum, times, "Angular Momentum Vector Trajectory (kg·m²/s)", "angular_momentum_3d_trajectory.png"))

    #creates a variation of the 3d plot for position that includes the central body in the figure
    image_filepaths.append(create_3d_position_trajectory(positions, times, "Satelite Position Trajectory (m)", "satelite_position_trajectory.png", central_body_radius))

    #updates the main label according to the results of the simulation
    main_label.grid(rowspan=1)
    if crash_index is None:
        message = f"Simulation Results, Satelite did not crash into {central_body_name}"
    else:
        message = f"Simulation Results, Satelite crashed into {central_body_name} at time {crash_index*time_step}"
    main_label.config(text=message, font=("Rockwell Condensed", 24, "bold"))

    #a function for every graph that creates a top level which holds the graph 
    def open_image_window(filepath):
        #configures the actual top level
        top = Toplevel()
        top.title(os.path.basename(filepath))
        top.config(bg="#000014")
        
        #uses PIL to proces the image and save it s a label
        image = Image.open(filepath)
        photo = ImageTk.PhotoImage(image)
        label = Label(top, image=photo, bg="#000014")
        label.image = photo
        label.pack(padx=10, pady=10)

    #creates the bottoms for each graph and saves it across the screen
    for idx, filepath in enumerate(image_filepaths):
        button = Button(root, text=os.path.basename(filepath).replace("_", " ").replace(".png", "").title(), command=lambda fp=filepath: open_image_window(fp), bg="#000014", fg="#BAC1FF", font=("Rockwell Condensed", 14, "bold"), relief="flat", activebackground="#000014", activeforeground="#BAC1FF", highlightthickness=0, bd=0, cursor="hand2")
        button.grid(row=16 + (idx // 4)*7, column=(idx % 4) * 24, columnspan=24, sticky="nsew")

    #function that exports the results as files according to the user
    def export_simulation_results():
        #gets the user to give the filepath
        file_path = filedialog.asksaveasfilename(title="Export Simulation Data", defaultextension=".csv", filetypes=[("CSV File", "*.csv"), ("Text File", "*.txt")], initialfile="simulation_results")
        if not file_path: return 

        #processes the data as a pandas data frame
        df = pd.DataFrame(data_dict)
        df.insert(0, "Time (s)", times)

        #saves depending on what the user provided or displays an error or success message
        try:
            if file_path.endswith(".txt"):
                df.to_csv(file_path, sep='\t', index=False)
            else:
                df.to_csv(file_path, index=False)
            main_label.config(text=f"Data sucessfully saved to {file_path}")
            root.update()
        except Exception as e:
            main_label.config(text=f"An error occurred export: {e}")
            root.update()

        root.after(3000, lambda: main_label.config(text=message))

    #function that exports all the graphs in a zip folder
    def export_graphs_as_zip():
        #gets the user to give the filepath
        zip_path = filedialog.asksaveasfilename( title="Export Graphs", defaultextension=".zip", filetypes=[("ZIP File", "*.zip")], initialfile="graphs_export")
        if not zip_path: return

        #saves to a zip file depending on what the user provided or displays an error or success message
        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for png_file in Path("graphs").glob("*.png"):
                    zipf.write(png_file, arcname=png_file.name)
            main_label.config(text=f"Graphs successfully zipped to {zip_path}")
        except Exception as e:
            main_label.config(text=f"Error while zipping: {e}")

        root.after(3000, lambda: main_label.config(text=message))

    #function that creates an animation of the satelite and exports it in parralel so that user can still browse results
    def export_animation_video():
        #updates main label to show processing animation
        file_path = filedialog.asksaveasfilename(title="Export Orbit Animation", defaultextension=".mp4", filetypes=[("MP4 Video", "*.mp4")], initialfile="orbit_animation")
        if not file_path:
            root.after(0, lambda: main_label.config(text=message))
            return

        #saves to a video depending on what the user provided or displays an error or success message
        main_label.config(text="Creating Animation- May Take Time and Freeze")
        root.update()
        try:
            create_3d_position_animation(positions, times, "Orbital Position Animation", file_path, central_body_radius)
            root.after(0, lambda: main_label.config(text=f"Animation saved to {file_path}"))
        except Exception as e:
            root.after(0, lambda: main_label.config(text=f"Error during animation export: {e}"))
        finally:
                root.after(5000, lambda: main_label.config(text=message))

    
    #creates the button that can export the results lists
    export_button = Button(root, text="Export Simulation Results", command=export_simulation_results, bg="#000014", fg="#BAC1FF", font=("Rockwell Condensed", 16, "bold"), relief="flat", activebackground="#000014", activeforeground="#BAC1FF", highlightthickness=0, bd=0, cursor="hand2")
    export_button.grid(row=80, column=0, columnspan=24, sticky="nsew")

    #creates the button that can export the graphs
    zip_button = Button(root, text="Export Simulation Graphs", command=export_graphs_as_zip, bg="#000014", fg="#BAC1FF", font=("Rockwell Condensed", 16, "bold"), relief="flat", activebackground="#000014", activeforeground="#BAC1FF", highlightthickness=0, bd=0, cursor="hand2"); 
    zip_button.grid(row=80, column=24, columnspan=24, sticky="nsew")

    #creates the button that can create and export the animation
    animation_button = Button(root, text="Export Orbit Animation", command=export_animation_video, bg="#000014", fg="#BAC1FF", font=("Rockwell Condensed", 16, "bold"), relief="flat", activebackground="#000014", activeforeground="#BAC1FF", highlightthickness=0, bd=0, cursor="hand2"); 
    animation_button.grid(row=80, column=48, columnspan=24, sticky="nsew")

    #creates the button that restarts the program
    restart_button = Button(root, text="Restart Simulation", command=restart_program, bg="#000014", fg="#BAC1FF", font=("Rockwell Condensed", 16, "bold"), relief="flat", activebackground="#000014", activeforeground="#BAC1FF", highlightthickness=0, bd=0, cursor="hand2"); 
    restart_button.grid(row=80, column=72, columnspan=24, sticky="nsew")

#creates the buttons and loads the images for each parameter
column_counter = 0
for parameter in parameters:
    #opens the image file corresponding to the parameter
    img = Image.open(f"icons/{parameter}.png")
    images.append(img.resize((100, 100)))
    imagetks.append(ImageTk.PhotoImage(images[-1]))

    #creates a button for the parameter with the image
    buttons.append(Button(image=imagetks[-1], bg="#000014", fg="#BAC1FF"))
    buttons[-1].grid(row=95, column=column_counter, rowspan = 12, columnspan=12, sticky="nsew")
    column_counter+= 12

# calls the function to define the labels with the current parameters
define_labels()

#attaches the functions to the buttons
buttons[0].config(command=change_central_body_settings)
buttons[1].config(command=change_satelite_parameters)
buttons[2].config(command=change_simulation_parameters)
buttons[3].config(command=change_j2_settings)
buttons[4].config(command=change_atmospheric_drag_settings)
buttons[5].config(command=change_third_body_settings)
buttons[6].config(command=reset)
buttons[7].config(command=run_simulation)
    
#begins the program
root.mainloop()
